"""Aggregate paired evaluations and enforce the documented promotion gate."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from muscle_memory.evaluation.runner import PolicyEpisodeResult

MINIMUM_V1_SUCCESS_RATE = 0.80
MAXIMUM_V1_COLLISION_RATE = 0.10
MINIMUM_V1_MEDIAN_CLEARANCE_M = 0.25
MINIMUM_SUCCESS_RATE_IMPROVEMENT = 0.20
MINIMUM_COLLISION_RATE_REDUCTION = 0.50
MAXIMUM_PATH_EFFICIENCY_REGRESSION = 0.15


@dataclass(frozen=True, slots=True)
class PolicyEvaluationSummary:
    """Comparable measurements for one policy on one fixed world set."""

    policy_id: str
    policy_hash: str
    episode_count: int
    success_count: int
    success_rate: float
    collision_episode_count: int
    collision_rate: float
    collisions_per_episode: float
    total_falls: int
    median_completion_time_seconds: float | None
    median_minimum_clearance_m: float
    maximum_tray_tilt_degrees: float
    package_slip_episodes: int
    median_path_efficiency: float
    energy_per_successful_delivery_joules: float | None
    human_interventions: int


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Every gate check plus the final promotable decision."""

    promotable: bool
    checks: dict[str, bool]
    success_rate_improvement: float
    collision_rate_reduction: float | None
    path_efficiency_regression: float
    baseline: PolicyEvaluationSummary
    candidate: PolicyEvaluationSummary


def summarize_policy_results(
    results: tuple[PolicyEpisodeResult, ...],
) -> PolicyEvaluationSummary:
    if not results:
        raise ValueError("policy summary requires at least one episode")
    policy_ids = {result.policy_id for result in results}
    policy_hashes = {result.policy_hash for result in results}
    if len(policy_ids) != 1 or len(policy_hashes) != 1:
        raise ValueError("policy summary cannot mix policy identities")
    successes = tuple(result for result in results if result.success)
    collision_episodes = sum(result.body_collisions > 0 for result in results)
    completion_times = tuple(
        result.time_to_resident_seconds
        for result in successes
        if result.time_to_resident_seconds is not None
    )
    energy_per_success = (
        float(statistics.mean(result.energy_joules for result in successes))
        if successes
        else None
    )
    return PolicyEvaluationSummary(
        policy_id=next(iter(policy_ids)),
        policy_hash=next(iter(policy_hashes)),
        episode_count=len(results),
        success_count=len(successes),
        success_rate=len(successes) / len(results),
        collision_episode_count=collision_episodes,
        collision_rate=collision_episodes / len(results),
        collisions_per_episode=float(
            statistics.mean(result.body_collisions for result in results)
        ),
        total_falls=sum(result.falls for result in results),
        median_completion_time_seconds=(
            float(statistics.median(completion_times)) if completion_times else None
        ),
        median_minimum_clearance_m=float(
            statistics.median(
                result.minimum_obstacle_clearance_m for result in results
            )
        ),
        maximum_tray_tilt_degrees=max(
            result.maximum_tray_tilt_degrees for result in results
        ),
        package_slip_episodes=sum(result.package_slipped for result in results),
        median_path_efficiency=float(
            statistics.median(result.path_efficiency for result in results)
        ),
        energy_per_successful_delivery_joules=energy_per_success,
        human_interventions=sum(result.human_interventions for result in results),
    )


def evaluate_promotion(
    baseline_results: tuple[PolicyEpisodeResult, ...],
    candidate_results: tuple[PolicyEpisodeResult, ...],
) -> PromotionDecision:
    """Require an exact paired comparison before promotion is even possible."""
    baseline_keys = tuple(
        (result.world_id, result.world_seed, result.robot_checksum)
        for result in baseline_results
    )
    candidate_keys = tuple(
        (result.world_id, result.world_seed, result.robot_checksum)
        for result in candidate_results
    )
    if baseline_keys != candidate_keys:
        raise ValueError("promotion requires identical ordered worlds and robot checksums")
    baseline = summarize_policy_results(baseline_results)
    candidate = summarize_policy_results(candidate_results)
    success_improvement = candidate.success_rate - baseline.success_rate
    collision_reduction = (
        None
        if baseline.collision_rate == 0.0
        else 1.0 - candidate.collision_rate / baseline.collision_rate
    )
    path_regression = (
        0.0
        if baseline.median_path_efficiency == 0.0
        else 1.0
        - candidate.median_path_efficiency / baseline.median_path_efficiency
    )
    measured_improvement = success_improvement >= MINIMUM_SUCCESS_RATE_IMPROVEMENT or (
        collision_reduction is not None
        and collision_reduction >= MINIMUM_COLLISION_RATE_REDUCTION
    )
    checks = {
        "at_least_80_percent_success": candidate.success_rate >= MINIMUM_V1_SUCCESS_RATE,
        "zero_falls": candidate.total_falls == 0,
        "collision_rate_at_most_10_percent": (
            candidate.collision_rate <= MAXIMUM_V1_COLLISION_RATE
        ),
        "median_clearance_at_least_0_25_m": (
            candidate.median_minimum_clearance_m >= MINIMUM_V1_MEDIAN_CLEARANCE_M
        ),
        "measured_improvement": measured_improvement,
        "path_efficiency_regression_at_most_15_percent": (
            path_regression <= MAXIMUM_PATH_EFFICIENCY_REGRESSION
        ),
    }
    return PromotionDecision(
        promotable=all(checks.values()),
        checks=checks,
        success_rate_improvement=success_improvement,
        collision_rate_reduction=collision_reduction,
        path_efficiency_regression=path_regression,
        baseline=baseline,
        candidate=candidate,
    )
