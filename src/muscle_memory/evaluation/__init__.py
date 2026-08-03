"""Episode evaluation contracts with held-out access loaded on demand."""

from typing import Any

from muscle_memory.evaluation.success import (
    SAFE_DELIVERY_CRITERIA,
    EpisodeOutcome,
    FailureReason,
    SafeDeliveryCriteria,
    SuccessEvaluation,
    Vector2,
    calculate_facing_error_degrees,
    calculate_stop_distance_metres,
    evaluate_safe_delivery,
)

_HELDOUT_EXPORTS = frozenset({"HeldOutBundleError", "ValidatedHeldOutWorld", "load_heldout_worlds"})


def __getattr__(name: str) -> Any:
    """Keep ordinary evaluation imports outside the held-out trust boundary."""

    if name in _HELDOUT_EXPORTS:
        from muscle_memory.evaluation import heldout

        return getattr(heldout, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SAFE_DELIVERY_CRITERIA",
    "EpisodeOutcome",
    "FailureReason",
    "HeldOutBundleError",
    "SafeDeliveryCriteria",
    "SuccessEvaluation",
    "ValidatedHeldOutWorld",
    "Vector2",
    "calculate_facing_error_degrees",
    "calculate_stop_distance_metres",
    "evaluate_safe_delivery",
    "load_heldout_worlds",
]
