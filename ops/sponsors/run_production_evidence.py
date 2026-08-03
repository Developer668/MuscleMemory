"""Run and retain one real, human-gated sponsor workflow in production."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import uvicorn

from muscle_memory.api import authenticator_from_env, create_app
from muscle_memory.backend.api_backend import MuscleMemoryApiBackend
from muscle_memory.coordinator.models import canonical_json
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ExecutionPlan,
    PipelineCommand,
    PipelineStep,
)
from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.runtime import build_api_backend

_TERMINAL_PHASES = frozenset({"closed", "failed"})
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProductionEvidenceError(RuntimeError):
    """A production provider path failed to produce durable evidence."""


def _request(
    origin: str,
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: object | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = canonical_json(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{origin.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4_096).decode("utf-8", errors="replace")
        raise ProductionEvidenceError(
            f"{method} {path} returned HTTP {exc.code}: {detail}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ProductionEvidenceError(f"{method} {path} failed ({type(exc).__name__})") from exc
    if not isinstance(decoded, dict):
        raise ProductionEvidenceError(f"{method} {path} returned a non-object response")
    return cast(dict[str, Any], decoded)


def _wait_for_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if server.started:
            return
        if not thread.is_alive():
            break
        time.sleep(0.1)
    raise ProductionEvidenceError("embedded production API did not start")


def _wait_for_episode(
    origin: str,
    episode_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _request(origin, "GET", f"/api/v1/live/episodes/{episode_id}")
        phase = status.get("phase")
        if phase in _TERMINAL_PHASES:
            if phase != "closed":
                raise ProductionEvidenceError(
                    f"live source episode {episode_id} failed ({status.get('error_type')})"
                )
            return status
        time.sleep(0.5)
    raise ProductionEvidenceError(f"live source episode {episode_id} timed out")


def _commands(
    *,
    episode_id: str,
    world_id: str,
    baseline_policy_id: str,
    candidate_policy_id: str,
) -> tuple[PipelineCommand, ...]:
    commands = (
        PipelineCommand.create(
            PipelineStep.VALIDATE_WORLD,
            {"uncertain_physical_properties": False, "world_id": world_id},
        ),
        PipelineCommand.create(
            PipelineStep.RUN_EPISODE,
            {"episode_id": episode_id, "world_id": world_id},
        ),
        PipelineCommand.create(
            PipelineStep.SUMMARIZE_TELEMETRY,
            {"episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.QUERY_GRAPH_MEMORY,
            {"episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.SELECT_CURRICULUM,
            {"curriculum_change_requested": False, "episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.TRAIN_CANDIDATE_POLICY,
            {
                "candidate_policy_id": candidate_policy_id,
                "reward_change_requested": False,
            },
        ),
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "baseline_policy_id": baseline_policy_id,
                "candidate_policy_id": candidate_policy_id,
                "heldout_world_set_id": "heldout-v1",
            },
        ),
        PipelineCommand.create(
            PipelineStep.PROMOTE_OR_ROLL_BACK,
            {"action": "roll_back", "candidate_policy_id": candidate_policy_id},
        ),
    )
    if tuple(command.step for command in commands) != FIXED_PIPELINE:
        raise ProductionEvidenceError("production workflow does not match the fixed pipeline")
    return commands


def _git_revision(expected_revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != expected_revision:
        raise ProductionEvidenceError("checkout does not match the expected deployed revision")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ProductionEvidenceError("production evidence requires a clean exact checkout")
    return revision


def _write_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ProductionEvidenceError(f"immutable evidence already exists: {path}") from exc
    try:
        retained = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProductionEvidenceError(
            f"immutable evidence is unreadable after upload: {path}"
        ) from exc
    if retained != encoded:
        raise ProductionEvidenceError(
            f"immutable evidence failed readback verification: {path}"
        )


def _reserve_output(path: Path, expected_revision: str) -> Path:
    if path.exists():
        raise ProductionEvidenceError(f"immutable evidence already exists: {path}")
    reservation = path.with_name(f"{path.name}.reservation")
    _write_once(
        reservation,
        {
            "evidence_path": str(path),
            "expected_revision": expected_revision,
            "reserved_at": datetime.now(UTC).isoformat(),
            "schema_version": 1,
        },
    )
    return reservation


def _provider_state(payload: dict[str, Any], label: str) -> str:
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ProductionEvidenceError(f"{label} provider evidence is missing")
    state = provider.get("state")
    if not isinstance(state, str):
        raise ProductionEvidenceError(f"{label} provider state is missing")
    return state


def _require_live_guild_review(review: dict[str, Any]) -> None:
    if _provider_state(review, "Guild") != "end_to_end_verified":
        raise ProductionEvidenceError("Guild did not retain end-to-end provider evidence")
    reviews = review.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 3:
        raise ProductionEvidenceError("Guild did not return all three specialist reviews")
    session_ids = [
        item.get("provider_session_id") if isinstance(item, dict) else None
        for item in reviews
    ]
    if any(not isinstance(item, str) or not item for item in session_ids):
        raise ProductionEvidenceError("Guild review is missing a live provider session id")
    if len(set(cast(list[str], session_ids))) != 3:
        raise ProductionEvidenceError("Guild specialist sessions are not distinct")


def _require_live_rocketride_run(run: dict[str, Any], *, expected_steps: int) -> None:
    expected_state = "end_to_end_verified" if expected_steps == len(FIXED_PIPELINE) else "healthy"
    if _provider_state(run, "RocketRide") != expected_state:
        raise ProductionEvidenceError(
            "RocketRide provider state does not match fixed-pipeline completion"
        )
    completed_steps = run.get("completed_steps")
    if not isinstance(completed_steps, list) or len(completed_steps) != expected_steps:
        raise ProductionEvidenceError("RocketRide step count does not match the fixed pipeline")
    for item in completed_steps:
        if not isinstance(item, dict):
            raise ProductionEvidenceError("RocketRide step receipt is malformed")
        receipt = item.get("provider_task_receipt_sha256")
        provider_run_id = item.get("provider_run_id")
        if not isinstance(receipt, str) or _HEX_SHA256.fullmatch(receipt) is None:
            raise ProductionEvidenceError("RocketRide step lacks a provider task receipt")
        if not isinstance(provider_run_id, str) or not provider_run_id:
            raise ProductionEvidenceError("RocketRide step lacks a provider run id")


def _wait_for_human_decision(
    backend: MuscleMemoryApiBackend,
    *,
    requirement_id: str,
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        decision = backend.approval_ledger.decision_for(requirement_id)
        if decision is None:
            time.sleep(0.5)
            continue
        if decision.verdict.value != "approve":
            raise ProductionEvidenceError("the authenticated human rejected the rollback gate")
        return {
            "decided_at": decision.decided_at.isoformat(),
            "decision_id": decision.decision_id,
            "human_subject": decision.human_subject,
            "note": decision.note,
            "plan_digest": decision.plan_digest,
            "requirement_id": decision.requirement_id,
            "verdict": decision.verdict.value,
        }
    raise ProductionEvidenceError("timed out waiting for an authenticated human API decision")


def _callback_outputs(backend: MuscleMemoryApiBackend, run_id: str) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for step, request_sha256, result_json in backend.coordinator.rocketride_callback_results(
        run_id
    ):
        decoded = json.loads(result_json)
        if not isinstance(decoded, dict):
            raise ProductionEvidenceError("durable RocketRide result is malformed")
        output = decoded.get("output")
        if not isinstance(output, dict):
            raise ProductionEvidenceError("durable RocketRide output is malformed")
        outputs.append(
            {
                "step": step.value,
                "request_sha256": request_sha256,
                "output_sha256": decoded.get("output_sha256"),
                "operation_execution": output.get("operation_execution"),
                "provider_state": output.get("provider_state"),
                "storage": output.get("storage"),
                "lesson_ids": output.get("lesson_ids"),
                "policy_evaluation_record_id": output.get(
                    "policy_evaluation_record_id"
                ),
                "action": output.get("action"),
                "target_policy_id": output.get("target_policy_id"),
                "numeric_decision_id": output.get("numeric_decision_id"),
            }
        )
    return outputs


def run(args: argparse.Namespace) -> dict[str, object]:
    revision = _git_revision(args.expected_revision)
    _reserve_output(args.output, revision)
    if args.approval_request.exists():
        raise ProductionEvidenceError(
            f"immutable approval request already exists: {args.approval_request}"
        )
    token = os.environ.get("MM_API_OPERATOR_TOKEN", "").strip()
    if len(token) < 24:
        raise ProductionEvidenceError(
            "MM_API_OPERATOR_TOKEN must be present for the authenticated human gate"
        )
    backend: MuscleMemoryApiBackend | None = None
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    try:
        backend = build_api_backend()
        app = create_app(backend=backend, authenticator=authenticator_from_env())
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=os.environ.get("MM_API_LOG_LEVEL", "info"),
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            name="production-evidence-api",
            daemon=True,
        )
        thread.start()
        _wait_for_server(server, thread)
        origin = f"http://127.0.0.1:{args.port}"
        started_at = datetime.now(UTC)
        health_before = _request(origin, "GET", "/api/v1/health")
        options = _request(origin, "GET", "/api/v1/live/options")
        policies = options.get("policies")
        seeds = options.get("seeds")
        if options.get("enabled") is not True or not isinstance(policies, list):
            raise ProductionEvidenceError("production live simulator is not admitted")
        if not isinstance(seeds, list) or not seeds:
            raise ProductionEvidenceError("production live world catalog is empty")
        baseline_policy_id = DirectGoalPolicy.policy_id
        policy_ids = {
            str(item.get("policy_id"))
            for item in policies
            if isinstance(item, dict)
        }
        if baseline_policy_id not in policy_ids:
            raise ProductionEvidenceError("held-out-evaluated baseline is not live-admitted")
        candidates = sorted(policy_ids - {baseline_policy_id})
        if len(candidates) != 1:
            raise ProductionEvidenceError("production requires one exact candidate policy")
        candidate_policy_id = candidates[0]

        categories: dict[str, list[str]] = defaultdict(list)
        episode_evidence: list[dict[str, object]] = []
        source_episode_id: str | None = None
        for raw_seed in seeds[: args.maximum_source_episodes]:
            seed = int(raw_seed)
            started = _request(
                origin,
                "POST",
                "/api/v1/live/episodes",
                token=token,
                payload={"seed": seed, "policy_id": baseline_policy_id},
            )
            episode_id = str(started["episode_id"])
            terminal = _wait_for_episode(
                origin,
                episode_id,
                timeout=args.episode_timeout_seconds,
            )
            closure = backend.episode_runtime.service.closure_for(episode_id)
            if closure is None or not closure.graph.provider_complete:
                raise ProductionEvidenceError(
                    f"source episode {episode_id} lacks provider-confirmed FalkorDB handoff"
                )
            if not closure.telemetry.provider_complete:
                raise ProductionEvidenceError(
                    f"source episode {episode_id} lacks provider-confirmed LaserData delivery"
                )
            failure_categories = tuple(sorted({item.category for item in closure.failures}))
            for category in failure_categories:
                categories[category].append(episode_id)
            episode_evidence.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "world_id": closure.identity.world_id,
                    "policy_id": closure.identity.policy_id,
                    "success": closure.result.success,
                    "failure_categories": failure_categories,
                    "telemetry_records": closure.telemetry.total_records,
                    "provider_confirmed_records": (
                        closure.telemetry.provider_confirmed_records
                    ),
                    "telemetry_digest": closure.telemetry_digest,
                    "graph_provider_complete": closure.graph.provider_complete,
                    "last_frame_id": terminal.get("last_frame_id"),
                    "video_frames": terminal.get("video_frames"),
                }
            )
            recurring = sorted(
                category for category, episode_ids in categories.items() if len(episode_ids) >= 2
            )
            if recurring:
                source_episode_id = episode_id
                break
        if source_episode_id is None:
            raise ProductionEvidenceError(
                "real baseline episodes did not produce one recurring failure category"
            )
        source_closure = backend.episode_runtime.service.closure_for(source_episode_id)
        if source_closure is None:
            raise ProductionEvidenceError("selected source episode closure disappeared")

        run_id = (
            f"production-sponsor-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        commands = _commands(
            episode_id=source_episode_id,
            world_id=source_closure.identity.world_id,
            baseline_policy_id=baseline_policy_id,
            candidate_policy_id=candidate_policy_id,
        )
        plan = ExecutionPlan.create(run_id, commands)
        bundle = backend.evidence_admitter.reproduce(
            plan,
            world_evidence_id=f"{run_id}.world",
            failure_curriculum_evidence_id=f"{run_id}.curriculum",
            evaluation_evidence_id=f"{run_id}.evaluation",
        )
        review = _request(
            origin,
            "POST",
            "/api/v1/workflows/review",
            token=token,
            payload={
                "run_id": run_id,
                "commands": [
                    {"step": command.step.value, "payload": command.payload}
                    for command in commands
                ],
                "evidence": bundle.model_dump(mode="json"),
            },
            timeout=args.provider_timeout_seconds,
        )
        if review.get("executable") is not True:
            raise ProductionEvidenceError("Guild specialists did not admit the exact plan")
        _require_live_guild_review(review)
        first_run = _request(
            origin,
            "POST",
            f"/api/v1/workflows/{run_id}/execute",
            token=token,
            timeout=args.provider_timeout_seconds,
        )
        if first_run.get("state") != "awaiting_human_approval":
            raise ProductionEvidenceError(
                "RocketRide did not stop at the human rollback gate"
            )
        _require_live_rocketride_run(first_run, expected_steps=7)
        pending = _request(origin, "GET", "/api/v1/approvals/pending")
        pending_items = pending.get("items")
        if not isinstance(pending_items, list):
            raise ProductionEvidenceError("pending approval response is malformed")
        matching = [
            item
            for item in pending_items
            if isinstance(item, dict)
            and item.get("run_id") == run_id
            and item.get("kind") == "policy_rollback"
        ]
        if len(matching) != 1:
            raise ProductionEvidenceError("exactly one rollback approval was not exposed")
        requirement_id = str(matching[0]["requirement_id"])
        _write_once(
            args.approval_request,
            {
                "candidate_policy_id": candidate_policy_id,
                "plan_digest": plan.digest,
                "requirement_id": requirement_id,
                "run_id": run_id,
                "schema_version": 1,
                "summary": matching[0].get("summary"),
                "submit_to": f"/api/v1/approvals/{requirement_id}/decision",
                "verdict_requested": "approve",
            },
        )
        approval = _wait_for_human_decision(
            backend,
            requirement_id=requirement_id,
            timeout=args.approval_wait_seconds,
        )
        completed = _request(
            origin,
            "POST",
            f"/api/v1/workflows/{run_id}/resume",
            token=token,
            timeout=args.provider_timeout_seconds,
        )
        if completed.get("state") != "completed" or len(
            cast(list[object], completed.get("completed_steps", []))
        ) != 8:
            raise ProductionEvidenceError("RocketRide did not complete all eight fixed steps")
        _require_live_rocketride_run(completed, expected_steps=8)
        health_after = _request(origin, "GET", "/api/v1/health")
        numeric = backend.coordinator.numeric_policy_decision_for_run(run_id)
        if numeric is None or numeric.action.value != "roll_back":
            raise ProductionEvidenceError("durable numeric rollback decision is missing")
        current_policy = backend.coordinator.current_policy("stable")
        if current_policy != baseline_policy_id:
            raise ProductionEvidenceError("stable alias does not point to the baseline")
        callbacks = _callback_outputs(backend, run_id)
        if len(callbacks) != 8:
            raise ProductionEvidenceError("eight durable callback results were not retained")
        graph_callbacks = [
            item for item in callbacks if item["step"] == "query_graph_memory"
        ]
        if len(graph_callbacks) != 1 or (
            graph_callbacks[0]["storage"] != "falkordb"
            or graph_callbacks[0]["provider_state"]
            not in {"healthy", "end_to_end_verified"}
        ):
            raise ProductionEvidenceError("workflow graph query was not served by FalkorDB")
        if _git_revision(args.expected_revision) != revision:
            raise ProductionEvidenceError("production revision changed during evidence execution")

        evidence: dict[str, object] = {
            "schema_version": 1,
            "evidence_kind": "production_sponsor_orchestration_with_admitted_artifacts",
            "repository_revision": revision,
            "run_id": run_id,
            "plan_digest": plan.digest,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "robot": verify_mm01_bundle().model_dump(mode="json"),
            "source_episodes": episode_evidence,
            "recurring_failure_categories": sorted(
                category for category, values in categories.items() if len(values) >= 2
            ),
            "laserdata": {
                "health_before": health_before,
                "health_after": health_after,
                "episode_count": len(episode_evidence),
                "provider_confirmed_records": sum(
                    cast(int, item["provider_confirmed_records"])
                    for item in episode_evidence
                ),
            },
            "falkordb": {
                "source_graph_handoffs_complete": all(
                    bool(item["graph_provider_complete"]) for item in episode_evidence
                ),
                "callback_outputs": graph_callbacks,
            },
            "guild": {
                "provider": review.get("provider"),
                "reviews": review.get("reviews"),
            },
            "rocketride": {
                "provider": completed.get("provider"),
                "completed_steps": completed.get("completed_steps"),
                "callback_outputs": callbacks,
            },
            "human_gate": approval,
            "policy_decision": {
                "action": numeric.action.value,
                "decision_id": numeric.decision_id,
                "stable_alias_policy_id": current_policy,
                "target_policy_id": numeric.target_policy_id,
            },
        }
        _write_once(args.output, evidence)
        return evidence
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=60.0)
            if thread.is_alive():
                raise ProductionEvidenceError("embedded production API did not stop cleanly")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="create-once path for the redacted production evidence JSON",
    )
    parser.add_argument("--approval-request", type=Path, required=True)
    parser.add_argument("--approval-wait-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-source-episodes", type=int, default=6)
    parser.add_argument("--episode-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.maximum_source_episodes < 2:
        raise SystemExit("--maximum-source-episodes must be at least 2")
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_revision) is None:
        raise SystemExit("--expected-revision must be a lowercase 40-character commit SHA")
    if args.approval_wait_seconds <= 0:
        raise SystemExit("--approval-wait-seconds must be positive")
    evidence = run(args)
    guild_reviews = cast(dict[str, object], evidence["guild"])["reviews"]
    rocketride = cast(dict[str, object], evidence["rocketride"])
    print(
        json.dumps(
            {
                "evidence_path": str(args.output),
                "guild_review_count": len(cast(list[object], guild_reviews)),
                "rocketride_step_count": len(
                    cast(list[object], rocketride["completed_steps"])
                ),
                "run_id": evidence["run_id"],
                "stable_alias_policy_id": cast(
                    dict[str, object], evidence["policy_decision"]
                )["stable_alias_policy_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
