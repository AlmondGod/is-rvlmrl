# Three-arm full assembly: code-policy experiment

This extension executes generated programs in the existing torso-mounted three
UR10e scene: four loose frame members, a loose plywood sheet, two grippers and a
nail tool. It retains the historical environment, RGB feedback skills, finite
breakable nail model and independent acceptance gates without modifying them.
It is separate from the frozen live visual-language benchmark.

The language model generates a whole program before simulation begins. Each
call invokes an existing camera-guided skill. The model receives API semantics
and the assembly requirements, not images, simulator poses, the layout seed,
hidden joint conditions, demonstrations or a completed example. All layouts
have the same prompt. This tests learning a reusable supervisory program, not
visual reasoning, learned motor control or layout-specific code synthesis.

The AST-only API permits at most 32 calls: `position_member(i)`,
`release_member(i)`, `park_grippers()`, `nail_corner(i)`, `feed_panel()`,
`position_panel()`, `release_panel()`, `nail_panel(i)`, `flip_structure()`,
`finish()` and `stop()`. Indices must be literal integers 0–3. Imports, loops,
variables, expressions, arbitrary Python and calls after a terminal instruction
are rejected. Runtime failures terminate the program; there is no repair or
automatic insertion of missing actions.

One continuous MuJoCo state advances at the original 0.5 ms timestep with 50 Hz
control and 10 Hz perception, using the original 1280×960 camera images. Six
seconds of physical settling precede each call, including the first and finish.
This is an explicit fixed scheduling choice, not simulated measured inference
latency. There is no wall-clock pacing, changed physics, object attachment,
runtime workpiece pose injection or success-by-command-count.

## Success and reward

The final `Evaluation.result` from `assembly_benchmark` supplies success:

- Correct released frame and sheet geometry before lifting, including member
  XY error below 6 mm, yaw/tilt below 1 degree, sheet XY below 6 mm and height
  error below 3 mm, with three seconds of released supported stability.
- Eight intact simulated fasteners after flipping; triggers alone do not count.
- At least five seconds inverted, with fixture clearance above 20 mm,
  airborne relative displacement below 5 mm and angle error below 3 degrees.
- Peak contact penetration at most 0.5 mm, no solver/nonfinite failures and no
  unintended robot/workpiece contact detected by the original flip-period gate.
- A completed flip and final `finish()`, with no execution error.

Dense reward is `0.30 * released_members/4 + 0.25 * intact_nails/8 +
0.10 * min(preflip_stability/3, 1) + 0.15 * min(inverted_hold/5, 1) +
0.20 * strict_success`. Member credit requires measured position, orientation,
height and released contact after its release skill. It is latched progress
credit, not the final success criterion. Nail credit uses intact connections,
not tool visits. Numerical or detected unintended-contact violations zero the
whole reward. Syntax errors and truncated model responses also receive zero.
Other runtime errors may retain measured partial-progress reward but never
success. Code length and elapsed task time are not optimized.

The insertion law is an uncalibrated instantaneous ray test and breakable spring
surrogate; it is not validated nail-driving or real-world construction physics.

## Predeclared pilot

Use Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, raw base
initialization, non-thinking mode, rank-16 all-linear LoRA and the existing
GRPO trainer. Initial budget: 32 optimizer updates, two layouts with four sampled
programs each per update, 512 completion tokens, learning rate 0.0001, KL coefficient
0.02, microbatch one, seed 42, validation every eight updates. Any later training
extension must be selected using training/validation only, before held-out test.

Throughput amendment before any optimizer update: the first sequential
preparation run was stopped with zero updates and its partial baseline evidence
retained. Four independent physics workers and two groups of four replace one
group of eight, preserving 256 sampled programs and 32 updates. Duplicate parsed
programs on the same layout are executed once per batch. A contact-driven
sequential/parallel regression verifies states and rewards, not merely counts.

Training seeds 41001–41008, validation 42001–42004 and held-out 43001–43008 are
disjoint from historical benchmark seeds. Layouts randomize feeder-lane order,
XY and yaw using the unchanged environment. Do not remove unsuccessful layouts.
Checkpoint selection prioritizes validation strict success, then mean reward;
the untrained baseline is eligible. Greedy baseline and selected model are
evaluated on the same untouched test layouts. Eight cases constitute a pilot,
not a reliability claim. Full-scene simulation is substantially more expensive
than the earlier single-rail experiment; GPU time and simulation time must be
reported separately.

Identical programs may reuse previously measured results only for the
same seed, complete source/asset hashes, runtime versions and platform. Cache
certificates hash the saved program, result, provenance, XML and state recording.
Programs sharing every call through a previously measured terminal runtime
failure can also reuse that failure: the differing later calls cannot execute.
This does not estimate a continuation or restore an intermediate simulator state.
No reward model or simplified symbolic simulator replaces physical execution.
Partial/interrupted caches are not retried silently.

```bash
# Headless Linux needs a working EGL driver. Install requirements.txt and the
# separate Qwen3.5 training dependencies into the chosen environment.
export MUJOCO_GL=egl
export CAP_RL_ASSEMBLY_CACHE=/absolute/path/to/new/assembly_evidence
python -m unittest experiments.cap_rl.test_assembly
python -m experiments.cap_rl.train train --task assembly_full \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out /absolute/path/to/new/run --steps 32 --prompts-per-step 2 \
  --group-size 4 --microbatch 1 --physics-workers 4 --max-tokens 512 --eval-every 8 --lr 0.0001
# Run once, only after validation-based selection is final:
python -m experiments.cap_rl.train test --out /absolute/path/to/run
```

To execute and record a particular program, use `python -m
experiments.cap_rl.assembly --program PROGRAM.py --seed SEED --out NEW_DIRECTORY
--video`. Development smoke runs can set `--seconds`; scored runs use 1200.
The script in `assembly_benchmark.mock_server` is an infrastructure-only
feasibility reference. Never label it model-generated or use it as an RL target.

## Corrected-support experiment (declared before held-out evaluation)

The contact-only exploratory pilot found a measurement defect: finite panel-to-
frame nails carry part of the panel weight, but the original support predicate
counts only collision forces. An independent GPU replay matches all 6,419 saved
physical states exactly while contact plus measured internal nail support
balances the panel weight. This correction is not a learned improvement.

The next experiment restarts **raw Qwen3.5-4B**, with the same model revision,
splits, prompt and 32-update configuration above. It does not initialize from the
retired contact-only pilot. Use `--support-profile contact_and_nails
--controller-profile southwest_approach --checkpoint-every 1` for BOTH baseline
and trained evaluations. These profiles are included in the evidence identity;
old-profile results cannot supply new-profile rewards. The corrected evaluator
also records `contact_only_acceptance` on every trajectory, preserving the old
classification for comparison. All numerical, geometry, eight-fastener and
inverted-hold thresholds remain unchanged.

The fixed controller profile changes only the southwest corner approach
horizontal component from 0.80 to 0.90 (with the corresponding normalized
vertical component). It was selected using scripted training-layout trials,
never test outcomes. Every model and layout uses the same controller. The
original controller and contact-only evaluator remain the defaults. Factory
injection preserves the original default direction expression exactly.

Independent processes requesting the same cache identity take a per-identity
file lock, so speculative evaluation and a learner can share one measured
execution. Dependencies are checked again after waiting and after physics.
Interrupted results remain evidence but are never certified as training rewards.
Incomplete evidence still requires explicit investigation, not an automatic retry.

Each update publishes an adapter snapshot with file hashes by atomically renaming
a completed temporary directory. These are adapter-only checkpoints, not full
optimizer/RNG resumption state. Validation still selects the final policy;
merely saving an update does not make it eligible for held-out evaluation.
