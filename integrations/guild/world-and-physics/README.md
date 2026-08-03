# World and Physics Agent

Owns obstacle physical-property review, world assembly evidence, and world-validation
evidence. It cannot execute a world, approve a proposal, or change robot configuration.

The agent returns `block` for any failed required world check. Uncertain or
agent-proposed properties are surfaced through the human approval kind while the
recommendation remains constrained by the evidence review.

