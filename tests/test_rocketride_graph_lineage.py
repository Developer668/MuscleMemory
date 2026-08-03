from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from muscle_memory.backend.rocketride_callback import FixedStepDispatcher
from muscle_memory.graph_memory import (
    EvaluatedPolicyVersion,
    GraphStorage,
    GraphWriteReceipt,
    ProviderState,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def _policy(policy_id: str, marker: str) -> EvaluatedPolicyVersion:
    return EvaluatedPolicyVersion.create(
        policy_id=policy_id,
        checkpoint_hash=marker * 64,
        evaluation_evidence_hash=marker * 64,
        evaluation_split="held_out",
        metrics={"collision_rate": 0.1, "success_rate": 0.85},
        evaluated_at=NOW,
    )


class _Graph:
    def __init__(self) -> None:
        self.policies: list[EvaluatedPolicyVersion] = []
        self.comparisons: list[Any] = []

    def record_evaluated_policy(self, policy: EvaluatedPolicyVersion) -> GraphWriteReceipt:
        self.policies.append(policy)
        return _receipt("evaluated_policy", policy.policy_id, policy.content_hash)

    def record_outperformance(self, comparison: Any) -> GraphWriteReceipt:
        self.comparisons.append(comparison)
        return _receipt("outperformance", "candidate:baseline:evidence", comparison.content_hash)

    def record_policy_evaluation(self, comparison: Any) -> GraphWriteReceipt:
        return _receipt(
            "policy_evaluation",
            "candidate:baseline:evidence",
            comparison.content_hash,
        )


class _Episodes:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _record_training_lineage(self, **kwargs: object) -> tuple[GraphWriteReceipt, ...]:
        self.calls.append(kwargs)
        return (
            _receipt("evaluated_policy", "candidate", "a" * 64),
            _receipt("policy_training", "lesson-1:candidate:evidence", "b" * 64),
        )


def _receipt(kind: str, record_id: str, content_hash: str) -> GraphWriteReceipt:
    return GraphWriteReceipt(
        record_kind=kind,
        record_id=record_id,
        content_hash=content_hash,
        storage=GraphStorage.FALKORDB,
        provider_state=ProviderState.HEALTHY,
        mirrored_to_local_cache=True,
        detail="test",
    )


def test_training_step_records_trusted_lesson_lineage() -> None:
    candidate = _policy("candidate", "c")
    episodes = _Episodes()
    evidence = SimpleNamespace(
        candidate=SimpleNamespace(policy_id="candidate"),
    )
    curriculum = SimpleNamespace(
        graph_query_digest="d" * 64,
        failure_patterns=(
            SimpleNamespace(lesson_ids=("lesson-2", "lesson-1")),
            SimpleNamespace(lesson_ids=("lesson-1",)),
        ),
    )
    bundle = SimpleNamespace(
        evaluation=SimpleNamespace(evaluation_evidence=evidence),
        failure_curriculum=SimpleNamespace(failure_curriculum_evidence=curriculum),
    )
    dispatcher = SimpleNamespace(
        _episodes=episodes,
        _required_evidence=lambda _plan: bundle,
        _checkpoint=lambda _policy_id: candidate,
        _verify_checkpoint=lambda *_args, **_kwargs: None,
        _async_job_completion=FixedStepDispatcher._async_job_completion,
    )

    result = FixedStepDispatcher._train_candidate_policy(  # type: ignore[arg-type]
        cast(Any, dispatcher),
        cast(Any, None),
        {"candidate_policy_id": "candidate", "reward_change_requested": False},
    )

    assert episodes.calls[0]["lesson_ids"] == ("lesson-1", "lesson-2")
    assert episodes.calls[0]["evidence_hash"] == "d" * 64
    assert result["training_lineage_record_ids"] == ["lesson-1:candidate:evidence"]


@pytest.mark.parametrize(
    ("action", "comparison_count"),
    (("promote", 1), ("roll_back", 0)),
)
def test_evaluation_records_outperformance_only_for_a_passing_gate(
    action: str,
    comparison_count: int,
) -> None:
    baseline = _policy("baseline", "b")
    candidate = _policy("candidate", "c")
    graph = _Graph()
    evidence = SimpleNamespace(
        baseline=SimpleNamespace(success_rate=0.55, collision_rate=0.4),
        candidate=SimpleNamespace(success_rate=0.85, collision_rate=0.1),
        heldout_world_set_id="heldout-v1",
        paired_world_count=20,
        proposed_action=action,
    )
    bundle = SimpleNamespace(evaluation=SimpleNamespace(evaluation_evidence=evidence))
    dispatcher = SimpleNamespace(
        _graph_memory=graph,
        _required_evidence=lambda _plan: bundle,
        _checkpoint=lambda policy_id: baseline if policy_id == "baseline" else candidate,
        _verify_checkpoint=lambda *_args, **_kwargs: None,
        _async_job_completion=FixedStepDispatcher._async_job_completion,
    )

    result = FixedStepDispatcher._evaluate_candidate_policy(  # type: ignore[arg-type]
        cast(Any, dispatcher),
        cast(Any, None),
        {"baseline_policy_id": "baseline", "candidate_policy_id": "candidate"},
    )

    assert graph.policies == []
    assert result["policy_evaluation_record_id"] == "candidate:baseline:evidence"
    assert len(graph.comparisons) == comparison_count
    assert (result["outperformance_record_id"] is not None) is (comparison_count == 1)
    if graph.comparisons:
        comparison = graph.comparisons[0]
        assert comparison.success_rate_delta == pytest.approx(0.3)
        assert comparison.collision_rate_delta == pytest.approx(-0.3)
