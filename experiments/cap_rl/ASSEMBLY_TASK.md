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
GRPO trainer. Initial budget: 32 optimizer updates, one layout and eight sampled
programs per update, 512 completion tokens, learning rate 0.0001, KL coefficient
0.02, microbatch one, seed 42, validation every eight updates. Any later training
extension must be selected using training/validation only, before held-out test.

Training seeds 41001–41008, validation 42001–42004 and held-out 43001–43008 are
disjoint from historical benchmark seeds. Layouts randomize feeder-lane order,
XY and yaw using the unchanged environment. Do not remove unsuccessful layouts.
Checkpoint selection prioritizes validation strict success, then mean reward;
the untrained baseline is eligible. Greedy baseline and selected model are
evaluated on the same untouched test layouts. Eight cases constitute a pilot,
not a reliability claim. Full-scene simulation is substantially more expensive
than the earlier single-rail experiment; GPU time and simulation time must be
reported separately.

Exact identical programs may reuse previously measured results only for the
same seed, complete source/asset hashes, runtime versions and platform. Cache
certificates hash the saved program, result, provenance, XML and state recording.
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
  --out /absolute/path/to/new/run --steps 32 --prompts-per-step 1 \
  --group-size 8 --microbatch 1 --max-tokens 512 --eval-every 8 --lr 0.0001
# Run once, only after validation-based selection is final:
python -m experiments.cap_rl.train test --out /absolute/path/to/run
```

To execute and record a particular program, use `python -m
experiments.cap_rl.assembly --program PROGRAM.py --seed SEED --out NEW_DIRECTORY
--video`. Development smoke runs can set `--seconds`; scored runs use 1200.
The script in `assembly_benchmark.mock_server` is an infrastructure-only
feasibility reference. Never label it model-generated or use it as an RL target.
