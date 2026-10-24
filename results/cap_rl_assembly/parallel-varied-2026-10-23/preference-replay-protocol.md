# Training-only sample-efficiency amendment

Registered after five GRPO rounds (40 sampled programs, 20 optimizer updates),
before round-six validation and before any varied held-out evaluation. Greedy
code is identical at the initial policy and checkpoints 1–5; scheduled
validation at rounds 2 and 4 remains 4/8. The exact source and result records
remain preserved. Full environment evaluation costs about 20 minutes per batch;
generation plus optimization costs about 25 seconds.

At the scheduled round-six boundary, retire this GRPO pilot **only if** every
scheduled greedy validation report still has the same programs and strict
outcomes as its baseline. If this condition is false, continue the existing
GRPO budget/selection rules. This is an explicit development-budget amendment,
not a claim of convergence or a held-out-derived stopping decision. Preserve
all six checkpoints, samples, measured failures, baseline/validation reports,
the original 16-round budget, stopping source and hashes. No held-out test is
triggered for the retired baseline-selected run.

The contingent next phase reuses the completed pilot's own certified training
programs. Build every success-versus-failure pair within the same training
layout; deduplicate exact programs before pairing. Never pair different
layouts or use validation/test outcomes as training labels. Layouts with only
successes or only failures remain in the source record but yield no preference
pair. Sample informative layouts uniformly, then pairs uniformly within a
layout. No successful hand-written order, teacher, repairs or synthetic
negative programs are supplied.

Optimize direct preference optimization (DPO; https://arxiv.org/abs/2305.18290)
with summed constrained sequence log-probabilities at temperature 1, beta 0.1,
learning rate 0.0001, four preference pairs per update, gradient accumulation
one pair at a time, 48 optimizer updates maximum, seed 42, dropout disabled and
non-reentrant activation checkpointing. Initialize from the completed GRPO
checkpoint 6; the original pre-GRPO adapter is the frozen DPO reference and the
before-training comparison. This is **GRPO followed by offline preference
optimization**, not 48 new on-policy GRPO batches or 48 new sets of simulations.

Keep the same grammar, complete physical task, reward, varied train/validation/
test seeds, cameras and earlier vectorized backend. Evaluate all eight validation
layouts every eight optimizer updates and save that checkpoint. Earliest maximum
validation wins, with the original baseline eligible. Stop at the first scheduled
8/8 validation boundary, or after 48 updates. The 24 held-out layouts remain
untouched until a final learned policy is frozen. Existing primary, unrestricted
transfer, raw-baseline, model-regeneration and fresh-replay requirements remain.
Record all parent RL cost and subsequent preference/validation cost separately.
