"""Create-once Daytona state snapshots for object-backed recovery storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = Path("/home/daytona/mm-data")
DEFAULT_SNAPSHOT_DIR = Path("/data/muscle-memory-snapshots")
FUSE_VOLUME_ROOT = Path("/data")
MANAGED_ROOTS = ("coordinator", "telemetry", "graph", "assets")
_TRANSIENT_SUFFIXES = ("-wal", "-shm", ".tmp")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DaytonaStateError(RuntimeError):
    """Daytona mutable state or an immutable recovery snapshot is unsafe."""


def reject_fuse_mutable_paths(paths: Iterable[Path]) -> None:
    """Reject SQLite, journals, logs, or PID state on the Mountpoint-S3 volume."""

    fuse_root = FUSE_VOLUME_ROOT.resolve()
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if resolved == fuse_root or resolved.is_relative_to(fuse_root):
            raise DaytonaStateError(f"mutable runtime path must not use /data FUSE: {path}")


def preflight(
    state_dir: Path,
    snapshot_dir: Path,
    *,
    mutable_paths: Iterable[Path] = (),
    require_data_mount: bool = True,
) -> None:
    reject_fuse_mutable_paths((state_dir, *mutable_paths))
    resolved_snapshots = snapshot_dir.expanduser().resolve(strict=False)
    if require_data_mount and not (
        resolved_snapshots == FUSE_VOLUME_ROOT
        or resolved_snapshots.is_relative_to(FUSE_VOLUME_ROOT)
    ):
        raise DaytonaStateError("immutable snapshots must use the /data object volume")
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(state_dir, os.W_OK):
        raise DaytonaStateError(f"sandbox state directory is not writable: {state_dir}")
    if not os.access(snapshot_dir, os.W_OK):
        raise DaytonaStateError(f"snapshot object directory is not writable: {snapshot_dir}")


def export_snapshot(
    state_dir: Path,
    snapshot_dir: Path,
    *,
    revision: str,
    snapshot_id: str | None = None,
) -> str | None:
    """Export one immutable snapshot, writing its manifest only after all objects."""

    if _REVISION.fullmatch(revision) is None:
        raise DaytonaStateError("snapshot revision must be a full lowercase commit SHA")
    sources = tuple(_managed_files(state_dir))
    if not sources:
        return None
    identifier = snapshot_id or _snapshot_id(revision)
    if _SNAPSHOT_ID.fullmatch(identifier) is None:
        raise DaytonaStateError("snapshot id is invalid")
    destination = snapshot_dir / identifier
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DaytonaStateError(f"snapshot already exists: {identifier}") from exc

    artifacts: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(state_dir)
        prepared, cleanup = _prepared_source(source, state_dir)
        try:
            digest, size = _copy_create_once(prepared, destination / "files" / relative)
        finally:
            if cleanup:
                prepared.unlink(missing_ok=True)
        artifacts.append(
            {"path": relative.as_posix(), "sha256": digest, "size": size}
        )

    manifest = {
        "schema_version": 1,
        "snapshot_id": identifier,
        "revision": revision,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
    }
    encoded = _canonical_json(manifest).encode("utf-8") + b"\n"
    _write_create_once(destination / "manifest.json", encoded)
    return identifier


def recover_latest(state_dir: Path, snapshot_dir: Path) -> str | None:
    """Restore the newest complete snapshot only into an empty managed state tree."""

    marker = state_dir / ".recovery-in-progress"
    if marker.exists():
        _clear_managed_state(state_dir)
        marker.unlink()
    if tuple(_managed_files(state_dir)):
        return None

    manifests = sorted(snapshot_dir.glob("*/manifest.json"), reverse=True)
    if not manifests:
        return None
    manifest_path = manifests[0]
    manifest = _load_manifest(manifest_path)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)

    stage = Path(tempfile.mkdtemp(prefix="daytona-recovery-", dir=state_dir))
    marker.write_text(str(manifest["snapshot_id"]) + "\n", encoding="utf-8")
    try:
        for raw in artifacts:
            artifact = _artifact(raw)
            source = manifest_path.parent / "files" / artifact["path"]
            _verify_file(source, artifact["sha256"], artifact["size"])
            target = stage / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            _verify_file(target, artifact["sha256"], artifact["size"])
        for root_name in MANAGED_ROOTS:
            staged_root = stage / root_name
            if staged_root.exists():
                os.replace(staged_root, state_dir / root_name)
        marker.unlink()
    except BaseException:
        _clear_managed_state(state_dir)
        marker.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return str(manifest["snapshot_id"])


def repository_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        raise DaytonaStateError("repository does not resolve to a full commit SHA")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise DaytonaStateError("repository cleanliness could not be verified")
    if status.stdout.strip():
        raise DaytonaStateError("repository is dirty; refusing snapshot provenance")
    return revision


def _managed_files(state_dir: Path) -> Iterable[Path]:
    for root_name in MANAGED_ROOTS:
        root = state_dir / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise DaytonaStateError(f"managed state must not contain symlinks: {path}")
            if path.is_file() and not path.name.endswith(_TRANSIENT_SUFFIXES):
                yield path


def _prepared_source(source: Path, state_dir: Path) -> tuple[Path, bool]:
    if source.suffix != ".sqlite3":
        return source, False
    descriptor, raw_path = tempfile.mkstemp(
        prefix="sqlite-snapshot-",
        suffix=".sqlite3",
        dir=state_dir,
    )
    os.close(descriptor)
    prepared = Path(raw_path)
    try:
        with (
            sqlite3.connect(f"file:{source}?mode=ro", uri=True) as origin,
            sqlite3.connect(prepared) as destination,
        ):
            origin.backup(destination)
    except BaseException:
        prepared.unlink(missing_ok=True)
        raise
    return prepared, True


def _copy_create_once(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        output = destination.open("xb")
    except FileExistsError as exc:
        raise DaytonaStateError(f"snapshot object already exists: {destination}") from exc
    with source.open("rb") as input_stream, output:
        while chunk := input_stream.read(1 << 20):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    checksum = digest.hexdigest()
    _verify_file(destination, checksum, size)
    return checksum, size


def _write_create_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DaytonaStateError(f"snapshot object already exists: {path}") from exc
    try:
        retained = path.read_bytes()
    except OSError as exc:
        raise DaytonaStateError(f"snapshot object is unreadable after upload: {path}") from exc
    if retained != payload:
        raise DaytonaStateError(f"snapshot object failed readback verification: {path}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DaytonaStateError(f"snapshot manifest is invalid: {path}") from exc
    required = {"schema_version", "snapshot_id", "revision", "created_at", "artifacts"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise DaytonaStateError("snapshot manifest fields are invalid")
    if payload["schema_version"] != 1 or payload["snapshot_id"] != path.parent.name:
        raise DaytonaStateError("snapshot manifest identity is invalid")
    if not isinstance(payload["revision"], str) or _REVISION.fullmatch(payload["revision"]) is None:
        raise DaytonaStateError("snapshot manifest revision is invalid")
    if not isinstance(payload["artifacts"], list) or not payload["artifacts"]:
        raise DaytonaStateError("snapshot manifest has no artifacts")
    seen: set[str] = set()
    for raw in payload["artifacts"]:
        artifact = _artifact(raw)
        key = artifact["path"].as_posix()
        if key in seen:
            raise DaytonaStateError("snapshot manifest repeats an artifact path")
        seen.add(key)
    return payload


def _artifact(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size"}:
        raise DaytonaStateError("snapshot artifact fields are invalid")
    raw_path = raw["path"]
    if not isinstance(raw_path, str):
        raise DaytonaStateError("snapshot artifact identity is invalid")
    relative = Path(raw_path)
    digest = raw["sha256"]
    size = raw["size"]
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in MANAGED_ROOTS
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise DaytonaStateError("snapshot artifact identity is invalid")
    return {"path": relative, "sha256": digest, "size": size}


def _verify_file(path: Path, expected_digest: str, expected_size: int) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise DaytonaStateError(f"snapshot artifact is unreadable: {path}") from exc
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise DaytonaStateError(f"snapshot artifact failed verification: {path}")


def _clear_managed_state(state_dir: Path) -> None:
    for root_name in MANAGED_ROOTS:
        target = state_dir / root_name
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)


def _snapshot_id(revision: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{revision[:12]}-{secrets.token_hex(4)}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("preflight", "recover", "export"),
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--mutable-path", action="append", default=[], type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--revision")
    return parser


def main() -> int:
    args = _parser().parse_args()
    state_dir = args.state_dir.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    preflight(
        state_dir,
        snapshot_dir,
        mutable_paths=args.mutable_path,
    )
    if args.operation == "preflight":
        print(f"Daytona mutable state: {state_dir}")
        print(f"Daytona immutable snapshots: {snapshot_dir}")
        return 0
    if args.operation == "recover":
        recovered = recover_latest(state_dir, snapshot_dir)
        print("no recovery required" if recovered is None else f"recovered snapshot {recovered}")
        return 0
    if args.revision is None:
        raise SystemExit("export requires --revision")
    revision = repository_revision(args.repository.expanduser().resolve())
    if args.revision.lower() != revision:
        raise SystemExit("export revision does not match the clean repository checkout")
    exported = export_snapshot(
        state_dir,
        snapshot_dir,
        revision=revision,
    )
    print("no mutable state to export" if exported is None else f"exported snapshot {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
