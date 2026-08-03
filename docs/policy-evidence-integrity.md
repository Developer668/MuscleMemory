# Policy evidence integrity

## Held-out artifact admission

Production accepts exactly two version-1 top-level schemas:

- The immutable legacy five-key schema already stored at
  `evidence/policy/delivery-v1/heldout-evaluation.json`.
- The checkpoint-bound six-key schema emitted by the current one-shot evaluator.

Both paths require an independently configured canonical artifact hash. The
admission service hashes the configured checkpoint itself and requires that hash
to equal the loaded policy identity, every candidate episode result, and the
candidate identity inside the recomputed promotion decision. The legacy file is
therefore supported without rewriting history or weakening the checkpoint
binding. Partial, mixed, coercible, or extended top-level schemas fail closed.

## Development evidence scope

Delivery-v2 development evidence uses the typed evaluation scope
`generated_disjoint_development`. Its worlds are deterministic, disjoint
development inputs created with the training-world generator and validator. The
per-result `world_split: training` value records those generation mechanics; it
does not mean that these measured development episodes were part of the training
dataset. The envelope and lock both state that held-out access was never used.

`evidence/policy/delivery-v2/lock.json` binds the exact repository-relative
development-evidence path, checkpoint path, raw and canonical evidence hashes,
policy and robot identities, paired-world count, scope, provenance, and derived
access state. A caller may provide another lock only when that lock explicitly
binds the exact evidence and checkpoint paths supplied to the verifier.

The lock also binds the real expert dataset, training evidence, and first
development round. Verification hashes each canonical file and cross-checks the
dataset hash and robot identity stored in the dataset, training evidence, and
checkpoint metadata. A self-consistent replacement lock is still not authority
for the one-shot command.

The verifier rejects type coercion and unexpected fields. It reconstructs every
stored episode, validates its terminal outcome from measurements, verifies the
ordered world pairs and immutable identities, and recomputes both policy
aggregates and the promotion preview. The selection status and lock access state
must equal that recomputed decision.

## Import boundary

`ops.policy.evaluate_heldout` loads and verifies the development lock before it
imports any held-out-world module. The checked-in delivery-v2 candidate is
rejected by development evidence, so the default command stops at that gate and
never opens the held-out bundle. Ordinary evaluation imports also leave held-out
exports lazy; explicit callers retain the same public export names.

## Reviewed one-shot boundary

`src/muscle_memory/evaluation/heldout_trust.py` is the explicit human-review
boundary for the checked-in one-shot command. It independently pins the lock,
development evidence, checkpoint, training lineage, frozen world bundle, output,
and consumption-receipt paths and hashes. No training or evaluation command
updates this trust root automatically. Authorizing another candidate requires a
reviewed source change to that module.

The held-out CLI accepts no path overrides. After the reviewed development gate
passes, it atomically records the candidate-checkpoint hash and frozen-world-set
hash at the canonical receipt path before importing held-out code. An existing
receipt rejects every repeat even if another caller intends to use a different
output. The current rejected delivery-v2 candidate never reaches this claim and
therefore has no consumption receipt.
