# Balanced camera-conditioned on-policy continuation

The 48-update visual offline-DPO phase completed without validation improvement:
the original 4/8 policy remains selected. Its loop took 250.24 seconds; all
validation programs reused existing certified physical evidence. No held-out
evaluation was started. The same seven informative layouts supplied every
preference update, despite capturing inputs for all 32 training layouts.

This new development phase gathers on-policy data across the entire training
split. Shuffle the 32 training keys with Python Random(97), divide into eight
rounds of four layouts, and sample four full programs per layout at temperature
3.0. Each layout appears once unless perfect validation stops training early.
That is at most 128 samples, not 128 guaranteed distinct physical executions:
exact duplicates and certified terminal-failure prefixes retain explicit cache
receipts. All eligible new programs undergo full physical simulation.

Use Qwen3.5-4B at the same pinned revision, the original initial adapter as the
frozen reference and fair pre-training baseline, and the original GRPO round-six
adapter as warm start. Neither unsuccessful DPO phase supplies weights. Both
baseline and learned policies receive the same initial RGB-derived text inputs.
The warm-start lineage and the separately spent DPO attempts remain archived.

Use the existing clipped GRPO objective (clip 0.2, KL coefficient 0.005), binary
strict physical success, four optimizer epochs per rollout batch, learning rate
0.0001, seed 97, gradient norm clipping at 1, microbatch 8 and no dropout.
Generate 16 programs together on the GPU; eight single-threaded CPU MuJoCo
workers execute independent episodes. Old/reference log-probabilities overlap
simulation and are frozen across the four updates. Thus eight rollout rounds
mean 32 optimizer updates, not 32 independent environment batches.

Keep the complete physics, RGB feedback skills, three-permutation grammar,
placement variation and acceptance criteria unchanged. In particular, do not
relax nail strength, collision checks, released stability, airborne deformation
or the five-second inverted hold. The native simulator optimization remains a
separately verified backend, preserving this experiment's exact provenance and
existing certified cache.

Save every round's adapter. Evaluate all eight validation layouts after rounds
2, 4, 6 and 8; choose the earliest maximum including the original baseline.
Stop early only at a learned 8/8 score. Then freeze selection and run the same
24-layout held-out comparison, raw/free-form diagnostics, source and reward
audits, checkpoint regeneration and independent physical/sensor replays.
The target is still at least 20/24 strict held-out success. No held-out-guided
updates or retry selection are permitted.
