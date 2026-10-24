# Initial-camera conditioning, before held-out evaluation

The completed constant-prompt preference phase used all 48 planned updates.
Validation at updates 8/16/24/32/40/48 was 4/8, 4/8, 2/8, 2/8, 2/8, 2/8.
The original policy remains selected at 4/8. Its training loop took 3,716.40
seconds, including scheduled validation. The automatic final-test gate refused
this selection; the 24 held-out layouts have not been evaluated.

The next development phase changes the language policy's input. Previously the
language model received the identical prompt for every layout; only its supplied
feedback skills saw the changed placement. The new prompt adds initial XY/yaw
estimates and near/far ordering from the existing calibrated RGB estimator.
Known inventory and surveyed support geometry remain explicit priors. The model
receives neither the layout seed nor true object poses, joint outcomes or rewards.
Missing visual estimates stay unknown; no layout is dropped.

This is a language code policy conditioned on a fixed vision frontend's text
output, not training a raw-image VLM or a motor controller. The two-layout
feasibility probe reproduced every estimate from saved PNGs and calibration.
Initial sensor capture performs no simulation step and obtains no task outcome.

The phase is permitted only after the completed 48-update constant-prompt run
finishes below 8/8, with no held-out results. It retains that attempt's records
and accounting separately. Its weights are not used. The warm start is the
same GRPO round-six adapter, and the frozen reference is the original prior.

Training reuses the same 23 certified success/failure pairs within seven
informative training layouts, from the GRPO policy's own 48 samples. Each pair
is augmented with the initial camera estimates for that same layout. Original
programs, outcomes and evidence identities remain unchanged; original prompt
hashes are preserved. These are offline historical examples under a new input
representation, not new on-policy rollouts or externally authored demonstrations.
Identical code in different visual contexts receives separate log probabilities.

The optimizer remains DPO: beta 0.1, learning rate 0.0001, four pairs per update,
one-pair microbatches with gradient accumulation, gradient norm clipping at 1,
seed 42, disabled dropout, non-reentrant activation checkpointing and constrained
sequence log probabilities at temperature 1. Up to 48 optimizer updates are
allowed, with full eight-layout greedy validation every eight updates. Select
the earliest maximum, including the original baseline; stop at a learned 8/8
ceiling. Pair sampling is uniform over informative layouts, then pairs.

Physics, geometry, cameras, the program grammar and every strict acceptance gate
remain frozen on the parent's vectorized backend. The independently optimized
native backend is not substituted into this data-replay experiment. Observation
images, calibration, public robot packets, extracted estimates, prompt fragments
and source receipts are saved and hashed before optimization for all 32 training
and eight validation layouts. Capturing their inputs does not turn the unlabelled
training layouts into additional preference examples.

Both baseline and trained policies receive the same visual-input protocol.
The earlier constant-prompt 4/8 score must not substitute for the new baseline.
Final testing still requires a learned 8/8 validation selection and uses the
unchanged 24 held-out seeds 53001–53024. Held-out observations are collected only
after that selection is frozen. The target remains at least 20/24 strict success;
it is a target, not an observed result. No test-guided tuning or reward relaxation.

Final verification retains source/record auditing, exact checkpoint regeneration,
raw and unrestricted diagnostics, and fresh paired physical replays for seeds
53001, 53012 and 53024. It also reconstructs all visual estimates from saved
sensor bytes and independently re-renders the initial observations for those
same three test seeds. Training-loop cost, continuous allocated GPU time,
discarded attempts and optimizer counts remain separate.
