# Shutdown handoff — 2026-10-24

The user requested shutdown preparation at approximately 10:01 UTC. The active
learner and auxiliary simulation were stopped. Four complete rollout rounds,
16 GRPO optimizer updates and 64 sampled programs are preserved. There is **no
verified improvement on the varied full-frame task**. All 24 final test layouts
remain untouched.

## Results across the project

| Task | Model | Held-out before → after | Interpretation |
|---|---|---|---|
| Symbolic pick and place | Qwen2.5-Coder-1.5B | 4/128 → 121/128 | Completed independent execution audit |
| Ordered two-object delivery | Qwen2.5-Coder-1.5B | 3/128 → 103/128 | Completed small symbolic follow-up |
| MuJoCo inertial docking | Qwen2.5-Coder-1.5B | 1/128 → 107/128 | Physical puck task with supplied controller scaffold |
| Articulated single-rail pickup | Qwen2.5-Coder-1.5B | 55/128 → 122/128 | Contact grasp/lift; exact-coordinate inputs and fixed skills |
| Rail placement and release | Qwen3.5-4B | 21/128 → 128/128 | All reported successes also pass pickup-first audit; original training reward had a disclosed ordering loophole |
| Earlier full three-arm pilot | Qwen3.5-4B | 0/4 → 3/4 on completed paired cases | Partial test only; four trained cases and independent verification unfinished |
| New varied-placement full assembly | Qwen3.5-4B | Not evaluated | Latest completed validation is 3/8 versus initial 4/8 |

See the individual dated result directories for checkpoints, generated programs,
videos, task definitions and limitations. Scores from different tasks or splits
must not be combined into one improvement estimate.

## Current task and reward

Three simulated arms assemble four loose wooden members, nail four corners,
place and nail a plywood sheet, then lift and flip the complete structure.
Initial plank placement varies through lane swaps, X ±12 mm, Y ±35 mm and yaw
±0.06 radians. The fixed split is 32 training, 8 validation and 24 test layouts.

The policy receives text derived from reset-time RGB estimates, without hidden
poses or the layout seed. It writes a complete 22-call program using supplied
camera/encoder feedback skills. Constrained decoding learns three permutations:
member placement, corner nailing and panel nailing, for 24³ legal orders. This
is task-sequencing RL, not learned motor control, unrestricted Python, or a
native multimodal policy.

Training reward is exactly 1 for complete strict success and 0 otherwise.
Success requires all eight finite-strength nails intact, all required placements,
released stability before lifting, a 180° flip, at least five seconds inverted
hold, relative displacement below 5 mm, relative angle below 3°, penetration at
most 0.5 mm, and no numerical or prohibited-contact failure. Dense progress is
recorded diagnostically, not optimized. No acceptance threshold was relaxed to
improve this score.

## Latest completed training

Run `varied-online25` uses rank-16 all-linear LoRA on pinned Qwen3.5-4B, clipped
GRPO, four samples per layout, four layouts per round, four optimizer epochs per
round, learning rate 1e-4, KL coefficient 0.005 and sampling temperature 3.
Inference/gradients run in GPU batches; eight independent CPU MuJoCo workers
produce physical rewards. It is a hybrid loop, not GPU-native full simulation.

- Completed: **4 rollout rounds, 16 optimizer updates, 64 samples**.
- Sampled training successes: **35/64**, not a paired improvement estimate.
- Initial greedy validation: **4/8**.
- After 8 updates: **3/8**, comprising one paired win and two paired losses.
- Validation after 16 updates was interrupted by the shutdown request. No final
  selection file exists; checkpoint four is the latest training state, not a
  validated better policy.
- Process occupancy: approximately **1.995 single-GPU hours**, including setup,
  simulator waits and incomplete validation. Loop time through round four was
  6,162.58 seconds (1.712 hours); gradient optimization totaled **133.55 seconds**.
  These are not total billed hours for the project. Earlier certificate creation,
  failed attempts and concurrent benchmarks have separately preserved costs.
- The first two rounds reused prior certified physical results. Their 102.60-second
  loop time is not a fresh-rollout speed measurement.

Earlier varied-task GRPO and both offline-DPO attempts did not beat the 4/8
baseline. A prior process was killed by host-memory OOM. An attempted exact
recovery correctly failed its weight/gradient equality checks; run25 restarted
independently from the original warm adapter. Complete optimizer/RNG snapshots
now accompany its checkpoints, but GPU bitwise replay is not guaranteed.

The no-improvement retirement policy was declared before validation four, but
never resolved: shutdown interrupted that evaluation. This is a user-requested
interruption, not a completed plateau retirement.

## Verified acceleration and unfinished integration

- Vectorized CPU simulator: exact full trajectory, 35.57 → 21.25 minutes under
  shared load. Isolated early-prefix medians improve 45.37 → 26.80 seconds.
- Eight independent worlds: 7.97× throughput on a 20-simulated-second prefix
  benchmark. This is not an eightfold complete RL speedup.
- Integrated fused backend: both successful and failed full trajectories match
  exactly—6,482×155 and 6,309×155 recorded states, plus every non-timing result.
  Times are 1,000.40 → 731.87 seconds and 1,059.86 → 779.86 seconds. These shared-host,
  instrumented comparisons establish observed savings, not isolated throughput.
- Full in-memory snapshot restoration preserves all 6,494×155 states. A different
  program branching after nine completed calls also matches a fresh execution
  exactly, all 6,550×155 states. Its reused tail takes 624.27 seconds versus 841.65
  fresh, with a separately paid 919.14-second origin. Both retain their failures.
- Origin-file hash binding and a real spawned reward-worker initialization check
  pass. The actual Linux fused prefix experiment was interrupted before completing
  its fresh/origin/reused comparison. The subsequent physical queue check never
  started. Neither is claimed as a full passing integration.
- `train_assembly_cohort.py` and `audit_assembly_cohort.py` are **unrun prototypes**.
  They prepare full-layout gradient accumulation: all 32 training layouts, four
  samples each, GPU batches of 16, six bounded prefix workers and eight global
  optimizer epochs per cohort. Components have fixture tests; the composed real
  Qwen/physics learner has not been tested or deployed. The proposed smaller
  panel-only curriculum is also not deployed.
- Warp measured an initial physics kernel only. It omits the full feedback and
  custom-fastener pipeline and is not a training backend.

## Preserved data

[Checkpoint release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-varied-assembly-2026-10-24)
contains `online25-step-0000.tar.gz` through `online25-step-0004.tar.gz`.
Every archive has local per-file verification and a matching GitHub SHA-256.
Step four is 426,268,293 bytes, SHA-256
`c301f18e758c3e9011e9a88d71e40714b9737c97f623985ab26128f92d1f1090`.
Its adapter, AdamW state and Python/NumPy/Torch CPU/CUDA RNG state are included.

The final `cap-rl-shutdown-2026-10-24.tar.gz` supplements those increments with
77 verified payloads: final logs/telemetry, training records, retirement policy,
partial validation evidence, partial prefix experiment and worker tools.
It is 20,986,864 bytes, SHA-256
`e74e87c4182f7129cdb7e8ca8ac5041355eed7fafada93a9f1c24d6bbedcf6af`.
Interrupted episodes remain uncertified and must never become training rewards.

[Speed release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-assembly-speed-2026-10-23)
contains the complete integrated fused proof (278 verified payloads) and local
memory-prefix proof (286 verified payloads), alongside earlier acceleration
experiments and exact binaries/sources.

## Resume priorities

1. Restore online25 increments 0–4 in order, then the shutdown supplement, with
   original cache identities and frozen runtime/source. Keep interrupted entries
   separate; do not overwrite or silently certify them.
2. Re-evaluate checkpoint four on the original validation set and apply an
   explicitly recorded continuation/selection protocol. No final test outcomes
   have been exposed.
3. Complete Linux exact full-branch and certified queue tests in fresh output
   directories. Measure host RAM before choosing concurrent workers; the old pod
   had 24 GB GPU memory, a 31 GB host-memory limit and a 10.2-CPU quota.
4. Review and test the composed all-layout learner with the real model before
   freezing a new run. Its current parent check requires a completed selection;
   the interrupted run deliberately has none and cannot be passed directly.
5. Only after validation-based selection, evaluate the untouched 24 test layouts,
   regenerate programs independently, audit records and freshly replay physical
   before/after episodes. The declared target is at least 20/24 with paired
   improvement. That objective has not been achieved.
