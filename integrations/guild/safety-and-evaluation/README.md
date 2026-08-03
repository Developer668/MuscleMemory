# Safety and Evaluation Agent

Owns evaluation-evidence review, numeric promotion-gate enforcement, and rollback
recommendations. It recomputes every gate threshold from paired metrics before asking
the Guild LLM for a conservative evidence review.

It has no tool or approval-write capability. Even a passing `proceed` review carries the
required human promotion or rollback approval kind.

