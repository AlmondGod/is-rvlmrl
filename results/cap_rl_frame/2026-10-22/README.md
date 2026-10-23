# Qwen3.5-4B: pick, transfer, place and release a frame rail

**Training is in progress. These are development validation observations, not held-out test results.**

The isolated MuJoCo task extends the earlier xArm7 rail lift to require carrying the rail to a new XY target, lowering, releasing and withdrawing. It uses the same native articulated robot, full-size free wooden rail, gravity and contact controller. The model receives privileged coordinates; this is not visual or complete frame assembly.

Model: `Qwen/Qwen3.5-4B`, immutable revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Rank-16 all-linear LoRA trains 32,464,896 parameters. No teacher, demonstrations, supervised warm-start or previous task adapter. The method is an independent CaP-RL-style GRPO implementation, not upstream VeRL.

| Validation checkpoint | Physical successes |
|---|---:|
| Original model | 12/64 (18.75%) |
| Update 20 | 57/64 (89.06%) |

The predeclared run has 80 updates, two layouts × eight samples per update, learning rate 1e-4, KL coefficient .02, BF16 and microbatch one on one NVIDIA A40. Evaluation uses identical prompts and greedy decoding. The final test remains untouched. Checkpoint selection uses validation only.

[Task, exact reward and success criteria](../../../experiments/cap_rl/FRAME_TASK.md). The scripted feasibility controller passed all 512 training layouts and is never supplied as model training data. All 48 tests pass on the GPU host. Runtime checks confirm finite nonzero adapter gradients and record BF16 generation/full-forward differences.

Independent macOS replay matched all 64 baseline-validation programs. A partial training replay found a yaw trace difference of about 6.3e-5 rad on an unsafe rail-flip episode: both platforms give zero reward and failure. The strict replay tolerance remains 1e-7; original-runtime replay is being checked. This difference is retained in `frame01/cross_platform_note.json` rather than silently widening tolerances.

Raw metadata, frozen sources, validation programs, runtime checks and environment versions are in `frame01/`. Final test results, audits, weights and videos will be added after the run completes.
