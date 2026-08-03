# Failure and Curriculum Agent

Owns recurring-failure review, targeted-world proposals, and curriculum selection.
Its input schema accepts training-split evidence only and has no evaluation-world or
held-out episode field. It cannot train a policy or apply a curriculum proposal.

Any requested curriculum change is returned with the independent human approval kind.

