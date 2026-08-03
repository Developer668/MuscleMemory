"""Immutable live-episode, video, and typed event contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from muscle_memory.policy.observation import NavigationObservation
from muscle_memory.robot.command import TaskCommand
from muscle_memory.worlds.models import TrainingWorld

NUMERIC_TELEMETRY_HZ = 20
VIDEO_FRAME_RATE = 30
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        encoded = json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("selected evaluation evidence is not canonicalizable") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SelectedTaskPolicy(Protocol):
    """Only the three-output, sensor-observation policy surface is reachable."""

    policy_id: str
    policy_hash: str

    def command(self, observation: NavigationObservation) -> TaskCommand: ...


class ValidatedTrainingWorldEnvelope(Protocol):
    """Training-world boundary without importing the A* owning package at runtime."""

    @property
    def world(self) -> TrainingWorld: ...


def require_validated_training_world(envelope: object) -> TrainingWorld:
    """Accept only the concrete result emitted by the strict training validation gate."""
    envelope_type = type(envelope)
    admitted_types = {
        ("muscle_memory.worlds.generation.models", "ValidatedTrainingWorld"),
        ("muscle_memory.live.catalog", "ValidatedRuntimeWorld"),
    }
    if (envelope_type.__module__, envelope_type.__name__) not in admitted_types:
        raise TypeError("live episodes require a validation-gated training world")
    world = getattr(envelope, "world", None)
    if not isinstance(world, TrainingWorld):
        raise TypeError("live episodes require a validation-gated training world")
    return world


class VideoProduct(StrEnum):
    THIRD_PERSON = "third_person"
    LEFT_EYE_RGB = "left_eye_rgb"
    RIGHT_EYE_RGB = "right_eye_rgb"
    STEREO_COMPOSITE = "stereo_composite"
    DERIVED_DEPTH = "derived_depth"
    SIMULATOR_DEBUG_SEGMENTATION = "simulator_debug_segmentation"


class LiveEpisodePhase(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CLOSED = "closed"
    FAILED = "failed"


class LiveEpisodeHealth(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TERMINAL = "terminal"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvaluatedPolicySelection:
    """A task-policy implementation bound to completed held-out evidence.

    Evaluation is not promotion. ``promotable`` truthfully mirrors the measured gate and
    does not grant authority to promote the policy. A learned policy is backed by a
    checkpoint; the deterministic baseline is backed by its exact source artifact.
    """

    policy: SelectedTaskPolicy
    checkpoint_path: Path | None
    checkpoint_hash: str
    evaluation_path: Path
    evaluation_hash: str
    evaluation_evidence_hash: str
    evaluated_episode_count: int
    promotable: bool
    policy_source_path: Path | None = None

    def __post_init__(self) -> None:
        self.verify_integrity()

    def verify_integrity(self) -> None:
        """Fail if either selected artifact changed after evidence admission."""
        if self.checkpoint_hash != self.policy.policy_hash:
            raise ValueError("selected checkpoint hash does not match the loaded policy")
        if _SHA256_PATTERN.fullmatch(self.checkpoint_hash) is None:
            raise ValueError("checkpoint_hash must be a lowercase SHA-256 digest")
        if _SHA256_PATTERN.fullmatch(self.evaluation_hash) is None:
            raise ValueError("evaluation_hash must be a lowercase SHA-256 digest")
        if _SHA256_PATTERN.fullmatch(self.evaluation_evidence_hash) is None:
            raise ValueError(
                "evaluation_evidence_hash must be a lowercase SHA-256 digest"
            )
        if self.evaluated_episode_count <= 0:
            raise ValueError("selected policy requires completed evaluation episodes")
        if (self.checkpoint_path is None) == (self.policy_source_path is None):
            raise ValueError(
                "selected policy requires exactly one immutable implementation artifact"
            )
        implementation_path = self.checkpoint_path or self.policy_source_path
        assert implementation_path is not None
        if _sha256(implementation_path) != self.checkpoint_hash:
            artifact = "checkpoint" if self.checkpoint_path is not None else "source"
            raise ValueError(
                f"selected policy {artifact} bytes changed after policy loading"
            )
        if _sha256(self.evaluation_path) != self.evaluation_hash:
            raise ValueError("selected evaluation evidence changed after policy loading")
        if _canonical_json_sha256(self.evaluation_path) != self.evaluation_evidence_hash:
            raise ValueError("selected evaluation evidence changed after policy admission")

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        evaluation_path: Path,
        expected_evaluation_evidence_hash: str | None = None,
    ) -> EvaluatedPolicySelection:
        """Load an immutable behavior policy only when held-out evidence binds to it."""
        from muscle_memory.policy.network import BehaviorClonedPolicy

        policy = BehaviorClonedPolicy.load(checkpoint_path)
        try:
            evidence = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("selected policy evaluation evidence is unreadable") from exc
        if not isinstance(evidence, dict):
            raise RuntimeError("selected policy evaluation evidence must be an object")
        candidate_results = evidence.get("candidate_results")
        decision = evidence.get("promotion_decision")
        if not isinstance(candidate_results, list) or not candidate_results:
            raise RuntimeError("selected policy has no completed evaluation episodes")
        if not isinstance(decision, dict):
            raise RuntimeError("selected policy evaluation has no promotion decision")
        for result in candidate_results:
            if not isinstance(result, dict):
                raise RuntimeError("selected policy evaluation result is malformed")
            if (
                result.get("policy_id") != policy.policy_id
                or result.get("policy_hash") != policy.policy_hash
                or result.get("world_split") != "held_out"
            ):
                raise RuntimeError("selected policy does not match its held-out evidence")
        candidate = decision.get("candidate")
        if not isinstance(candidate, dict) or (
            candidate.get("policy_id") != policy.policy_id
            or candidate.get("policy_hash") != policy.policy_hash
            or candidate.get("episode_count") != len(candidate_results)
        ):
            raise RuntimeError("selected policy does not match the measured gate summary")
        promotable = decision.get("promotable")
        if not isinstance(promotable, bool):
            raise RuntimeError("selected policy promotion decision is malformed")
        evidence_hash = _canonical_json_sha256(evaluation_path)
        if expected_evaluation_evidence_hash is not None:
            if _SHA256_PATTERN.fullmatch(expected_evaluation_evidence_hash) is None:
                raise RuntimeError("admitted evaluation evidence hash is malformed")
            if evidence_hash != expected_evaluation_evidence_hash:
                raise RuntimeError("selected evaluation evidence was not admitted")
        return cls(
            policy=policy,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=policy.policy_hash,
            evaluation_path=evaluation_path,
            evaluation_hash=_sha256(evaluation_path),
            evaluation_evidence_hash=evidence_hash,
            evaluated_episode_count=len(candidate_results),
            promotable=promotable,
        )

    @classmethod
    def load_baseline(
        cls,
        *,
        evaluation_path: Path,
        expected_evaluation_evidence_hash: str | None = None,
    ) -> EvaluatedPolicySelection:
        """Load the immutable direct-goal baseline only from its measured evidence."""
        from muscle_memory.policy.baseline import DirectGoalPolicy

        policy = DirectGoalPolicy()
        source_file = inspect.getsourcefile(DirectGoalPolicy)
        if source_file is None:
            raise RuntimeError("deterministic baseline source artifact is unavailable")
        source_path = Path(source_file).resolve()
        if _sha256(source_path) != policy.policy_hash:
            raise RuntimeError("deterministic baseline identity does not match its source")
        try:
            evidence = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("baseline evaluation evidence is unreadable") from exc
        if not isinstance(evidence, dict):
            raise RuntimeError("baseline evaluation evidence must be an object")
        baseline_results = evidence.get("baseline_results")
        decision = evidence.get("promotion_decision")
        if not isinstance(baseline_results, list) or not baseline_results:
            raise RuntimeError("baseline has no completed evaluation episodes")
        if not isinstance(decision, dict):
            raise RuntimeError("baseline evaluation has no promotion decision")
        for result in baseline_results:
            if not isinstance(result, dict):
                raise RuntimeError("baseline evaluation result is malformed")
            if (
                result.get("policy_id") != policy.policy_id
                or result.get("policy_hash") != policy.policy_hash
                or result.get("world_split") != "held_out"
            ):
                raise RuntimeError("baseline does not match its held-out evidence")
        baseline = decision.get("baseline")
        if not isinstance(baseline, dict) or (
            baseline.get("policy_id") != policy.policy_id
            or baseline.get("policy_hash") != policy.policy_hash
            or baseline.get("episode_count") != len(baseline_results)
        ):
            raise RuntimeError("baseline does not match the measured gate summary")
        evidence_hash = _canonical_json_sha256(evaluation_path)
        if expected_evaluation_evidence_hash is not None:
            if _SHA256_PATTERN.fullmatch(expected_evaluation_evidence_hash) is None:
                raise RuntimeError("admitted evaluation evidence hash is malformed")
            if evidence_hash != expected_evaluation_evidence_hash:
                raise RuntimeError("baseline evaluation evidence was not admitted")
        return cls(
            policy=policy,
            checkpoint_path=None,
            checkpoint_hash=policy.policy_hash,
            evaluation_path=evaluation_path,
            evaluation_hash=_sha256(evaluation_path),
            evaluation_evidence_hash=evidence_hash,
            evaluated_episode_count=len(baseline_results),
            promotable=False,
            policy_source_path=source_path,
        )


@dataclass(frozen=True, slots=True)
class LiveEpisodeConfig:
    maximum_duration_seconds: float = 30.0
    render_width: int = 320
    render_height: int = 240
    jpeg_quality: int = 82
    realtime: bool = True
    degraded_lag_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not 0.05 <= self.maximum_duration_seconds <= 30.0:
            raise ValueError("live episodes must last between 0.05 and 30 seconds")
        if self.render_width < 64 or self.render_height < 48:
            raise ValueError("live render dimensions must be at least 64 by 48")
        if self.render_width > 1280 or self.render_height > 720:
            raise ValueError("live render dimensions must not exceed 1280 by 720")
        if self.render_width * self.render_height > 921_600:
            raise ValueError("live render area must not exceed 921600 pixels")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be within [1, 100]")
        if not math.isfinite(self.degraded_lag_seconds) or self.degraded_lag_seconds <= 0:
            raise ValueError("degraded_lag_seconds must be positive and finite")


@dataclass(frozen=True, slots=True)
class EncodedVideoProduct:
    product: VideoProduct
    mime_type: str
    width: int
    height: int
    sha256: str
    data: bytes

    def __post_init__(self) -> None:
        if self.mime_type != "image/jpeg":
            raise ValueError("live video products must be JPEG frames")
        if self.width <= 0 or self.height <= 0 or not self.data:
            raise ValueError("encoded video product must be non-empty")
        if hashlib.sha256(self.data).hexdigest() != self.sha256:
            raise ValueError("encoded video product checksum mismatch")

    def metadata(
        self,
        *,
        episode_id: str,
        frame_index: int,
    ) -> VideoProductMetadata:
        encoded_episode_id = quote(episode_id, safe="")
        encoded_product = quote(self.product.value, safe="")
        return VideoProductMetadata(
            product=self.product,
            mime_type=self.mime_type,
            width=self.width,
            height=self.height,
            byte_length=len(self.data),
            sha256=self.sha256,
            stream_url=(
                f"/api/v1/episodes/{encoded_episode_id}/video/{encoded_product}.mjpeg"
            ),
            frame_url=(
                f"/api/v1/episodes/{encoded_episode_id}/video/{encoded_product}"
                f"/frames/{frame_index}"
            ),
        )


@dataclass(frozen=True, slots=True)
class VideoProductMetadata:
    product: VideoProduct
    mime_type: str
    width: int
    height: int
    byte_length: int
    sha256: str
    stream_url: str
    frame_url: str

    def __post_init__(self) -> None:
        if not self.stream_url.startswith("/api/v1/episodes/"):
            raise ValueError("video stream URL must use the direct API service")
        if not self.stream_url.endswith(f"/{self.product.value}.mjpeg"):
            raise ValueError("video stream URL does not match its product")
        if not self.frame_url.startswith("/api/v1/episodes/"):
            raise ValueError("video frame URL must use the direct API service")
        if f"/video/{self.product.value}/frames/" not in self.frame_url:
            raise ValueError("video frame URL does not match its product")


@dataclass(frozen=True, slots=True)
class VideoFrameMetadata:
    frame_id: str
    frame_index: int
    scheduled_time_seconds: float
    captured_time_seconds: float
    telemetry_sequence: int
    products: tuple[VideoProductMetadata, ...]
    transport: str = "direct_video_service"

    def __post_init__(self) -> None:
        if not self.frame_id or self.frame_index < 0 or self.telemetry_sequence < 0:
            raise ValueError("video frame identity and indexes must be non-negative")
        if not math.isfinite(self.scheduled_time_seconds) or not math.isfinite(
            self.captured_time_seconds
        ):
            raise ValueError("video frame times must be finite")
        if self.transport != "direct_video_service":
            raise ValueError("video bytes may only use the direct video service")
        if {item.product for item in self.products} != set(VideoProduct):
            raise ValueError("a video frame must contain all required visual products")


@dataclass(frozen=True, slots=True)
class VideoFrameSet:
    metadata: VideoFrameMetadata
    products: tuple[EncodedVideoProduct, ...]

    def __post_init__(self) -> None:
        if len(self.products) != len(self.metadata.products):
            raise ValueError("video bytes do not match frame metadata")
        for encoded, metadata in zip(self.products, self.metadata.products, strict=True):
            if (
                encoded.product is not metadata.product
                or encoded.mime_type != metadata.mime_type
                or encoded.width != metadata.width
                or encoded.height != metadata.height
                or len(encoded.data) != metadata.byte_length
                or encoded.sha256 != metadata.sha256
            ):
                raise ValueError("video bytes do not match frame metadata")

    @property
    def byte_length(self) -> int:
        return sum(len(item.data) for item in self.products)

    def product(self, product: VideoProduct) -> EncodedVideoProduct:
        for item in self.products:
            if item.product is product:
                return item
        raise KeyError(product)


@dataclass(frozen=True, slots=True)
class ImuEvent:
    pelvis_accelerometer_mps2: tuple[float, ...]
    pelvis_angular_velocity_rad_s: tuple[float, ...]
    pelvis_orientation_wxyz: tuple[float, ...]
    torso_accelerometer_mps2: tuple[float, ...]
    torso_angular_velocity_rad_s: tuple[float, ...]
    torso_orientation_wxyz: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class JointEffortEvent:
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    actuator_effort: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ContactEvent:
    left_force_n: tuple[float, ...]
    right_force_n: tuple[float, ...]
    left_floor_contact: bool
    right_floor_contact: bool


@dataclass(frozen=True, slots=True)
class TrayStateEvent:
    current_tilt_degrees: float
    maximum_tilt_degrees: float


@dataclass(frozen=True, slots=True)
class TactileSlipEvent:
    package_slipped: bool
    pressure_available: bool


@dataclass(frozen=True, slots=True)
class BatteryEvent:
    charge_percent: float | None
    estimated_energy_joules: float
    power_draw_watts: float


@dataclass(frozen=True, slots=True)
class PolicyActionEvent:
    forward_speed_mps: float
    turning_rate_rad_s: float
    stop_probability: float


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    body_collisions: int
    new_body_collisions: int
    current_clearance_m: float
    minimum_clearance_m: float
    falls: int
    new_falls: int


@dataclass(frozen=True, slots=True)
class RewardProgressEvent:
    destination_distance_m: float
    progress_m: float
    tick_reward: float
    cumulative_reward: float


@dataclass(frozen=True, slots=True)
class SimulatorPoseEvent:
    position_x_m: float
    position_y_m: float
    yaw_radians: float
    signal_use: str = "Simulator ground truth"


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    completed: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class LiveTelemetryPayload:
    imu: ImuEvent
    joint_effort: JointEffortEvent
    contacts: ContactEvent
    tray_state: TrayStateEvent
    tactile_slip: TactileSlipEvent
    battery: BatteryEvent
    policy_action: PolicyActionEvent
    collisions: CollisionEvent
    reward_progress: RewardProgressEvent
    simulator_pose: SimulatorPoseEvent
    safety_markers: tuple[str, ...]
    completion: CompletionEvent
    video_frames: tuple[VideoFrameMetadata, ...]
    schema_version: int = 1
    event_type: str = "episode_tick"
    numeric_telemetry_hz: int = NUMERIC_TELEMETRY_HZ
    video_frame_rate: int = VIDEO_FRAME_RATE

    def as_json_value(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LiveEpisodeStatus:
    episode_id: str
    phase: LiveEpisodePhase
    health: LiveEpisodeHealth
    world_id: str
    policy_id: str
    policy_hash: str
    policy_promotable: bool
    simulation_time_seconds: float = 0.0
    wall_elapsed_seconds: float = 0.0
    wall_clock_lag_seconds: float = 0.0
    telemetry_records: int = 0
    video_frames: int = 0
    dropped_video_frames: int = 0
    last_frame_id: str | None = None
    provider_state: str | None = None
    completion_reason: str | None = None
    success: bool | None = None
    failed_reasons: tuple[str, ...] = ()
    graph_provider_complete: bool | None = None
    telemetry_provider_complete: bool | None = None
    error_type: str | None = None
    detail: str | None = None

    def as_json_value(self) -> dict[str, object]:
        return asdict(self)
