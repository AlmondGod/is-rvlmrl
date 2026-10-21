# Wuji wooden-frame assembly prototype

Runnable MuJoCo environment using the official **Wuji Hand 2.2 right hand**, with
20 actuated finger joints. The task is a **400 × 300 mm dry-fit wooden frame**,
40 mm-wide and 20 mm-thick rails, with half-lap corners and 0.5 mm shoulder
clearance. These dimensions and joint type are an initial design assumption.

## Run locally

From `/Users/almondgod/Repositories/phyla`:

```bash
.venv-frame/bin/python -m experiments.frame_assembly.run --video
.venv-frame/bin/mjpython -m experiments.frame_assembly.run --viewer
.venv-frame/bin/mjpython -m experiments.frame_assembly.run --viewer --assembled
.venv-frame/bin/python -m experiments.frame_assembly.test_environment
```

`mjpython` is required for the interactive MuJoCo viewer on macOS. The ordinary
Python command renders the demo without opening a window. No GPU server is used.
The viewer exposes MuJoCo's normal controls, including actuator sliders.

## What is simulated

- Each wooden rail is a free rigid body. Two non-overlapping collision boxes per
  rail describe its half-lap profile; no collision approximation fills the notch.
- Nominal wood density is 500 kg/m³; sliding friction is 0.55. These values and the
  compliant contact parameters are **uncalibrated assumptions**.
- Two 6 mm-high fixture supports hold only the horizontal rails. The vertical
  rails transfer their weight through the lap faces. Four corner pins mark the
  work area. This checks stability **in the fixture**, not a free-standing frame.
- The vendor's finger actuator limits, gains, inertias, meshes and collision
  exclusions are preserved. A free wrist is driven by an ideal external carrier.
  There is no arm model, reachability test or arm torque limit.
- Physics runs at 1 kHz; commands and exported state run at 50 Hz. Contacts use
  MuJoCo's elliptic friction cone and implicit-fast integration.

## What the demo does

`run --video` produces a **scripted, assisted transport demonstration**. It opts
into temporary board-to-carrier welds, lifts and lowers each rail, releases the
weld, and lets gravity and contact seat the piece. The hand follows the carrier
and stays clear of the wood. Its fingers are controllable but **do not grasp the
boards in this demo**. This is not a trained policy or evidence of dexterous
assembly capability. The video explicitly labels the assistance.

The regular environment and interactive viewer disable board assistance by
default. A policy must use wrist and finger actions to move the pieces.

Success requires all four rails within 2 mm horizontally, 1 mm vertically and
2 degrees of their targets; each lap transferring at least 0.01 N; low linear
and angular velocity; less than 1 mm contact penetration; no active board carry
constraints; and no hand contact. These conditions must persist for 2 seconds.
The temporary transport weld is not counted as a completed joint.

There is **no glue, fastener, material fracture, wood grain mechanics, elastic
bending, calibrated finger-pad friction or hardware validation**. Dry half-lap
pieces can separate when lifted or disturbed. A successful fixture episode does
not establish structural strength or sim-to-real performance.

## One-rail residual RL pilot

`rail_rl.py` provides a Gymnasium task for the east rail: lift its **lowest
corner at least 5 cm**, maintain a stable, unsupported grasp for 2 seconds,
translate 12 cm, then place and release it. Placement requires position within
15 mm horizontally and 3 mm vertically, orientation within 10 degrees, low
velocity and no hand contact for 1 second. Episodes last at most 16 seconds.
All manipulation uses actual hand contacts; board carry assistance stays off.

The frozen comparison uses an engineered feedback controller, that controller
plus residual PPO, and direct PPO. A 100-rollout search on one nominal case
(55,000 control steps) tunes the feedback controller before held-out evaluation.
Each learned method trains two seeds for 262,144 steps per seed, with four CPU
simulation workers. PPO uses a 128-by-128 MLP. Both learned methods receive the
same 164-dimensional privileged observation, reward, randomization, and motor
limits. Both also observe the nominal command; the residual method additionally
adds that command to the policy output. This isolates the additive control prior,
not access to the nominal controller's information.

Actions update wrist translation, rotation and 20 finger targets at 50 Hz.
Translation is capped at 0.25 m/s, rotation at 1.5 rad/s and each finger target
change at 0.06 rad per control step. Lift scoring rejects table/fixture support,
penetration of 1 mm or more, and excessive slip relative to the wrist.

Training varies initial XY by 15 mm, yaw by 0.1 rad, mass by 20%, and wood friction
from 0.4 to 0.8. Harder held-out cases use 30 mm, 0.2 rad, 40%, friction 0.3–0.9,
and a 0.9–1.1 frame geometry scale. Hand friction varies proportionally because
varying only the lower-friction wood would not change the effective hand contact
friction under MuJoCo's contact mixing rule. Each policy is evaluated on the same
20 standard and 20 harder cases, using its final fixed-budget checkpoint.

From the project root:

```bash
# Install requirements-rl.txt into a Python 3.12 environment with the vendor assets.
.venv-frame/bin/python -m experiments.frame_assembly.test_rail_rl
# These commands write to outputs/frame_assembly/rl; preserve existing results first.
.venv-frame/bin/python -m experiments.frame_assembly.rail_experiment optimize
.venv-frame/bin/python -m experiments.frame_assembly.rail_experiment evaluate --mode base --episodes 20 --video
.venv-frame/bin/python -m experiments.frame_assembly.run_rail_pilot
.venv-frame/bin/python -m experiments.frame_assembly.report_rail_pilot
```

The current local environment reuses PyTorch from the project's `.venv` through
`phyla_shared_runtime.pth`; a fresh installation should install the pinned
`requirements-rl.txt` instead. No cloud resources are required for this pilot.

`outputs/frame_assembly/rl/` preserves the source hashes, controller parameters,
training settings, episode logs, final policy checkpoints and observation
normalizers. `pilot_results.json` contains every evaluation case; `summary.json`
and `pilot_results.png` summarize the frozen comparison. Each seed-0 controller
also has videos and state trajectories for the first predetermined case in each
split. Their inherited bottom HUD describes the original frame assembly task;
use the top rail-task caption and the evaluation JSON for this pilot's scoring.
The `invalid_reset_*` and `superseded_friction_*` directories are archived debugging
runs and excluded from the results.

This is a small simulation pilot, with two training seeds, privileged state and
an ideal wrist carrier. The historical Astra trials are qualitative context,
not a matched full-task baseline. It does not test visual perception, real-time
Astra inference or real hardware.

### Completed pilot results

The four training runs completed 1,048,576 steps in total. Training plus learned
policy evaluation took approximately 9.5 minutes locally, excluding preparation
and baseline work. Each row below uses the same 20 cases in each split.

| Controller | Standard lift / hold | Harder lift / hold | Full placement, both splits |
|---|---:|---:|---:|
| Feedback baseline | 7/20 | 7/20 | 0/40 |
| Residual PPO, seed 0 | 0/20 | 1/20 | 0/40 |
| Residual PPO, seed 1 | 5/20 | 6/20 | 0/40 |
| Direct PPO, seed 0 | 0/20 | 0/20 | 0/40 |
| Direct PPO, seed 1 | 0/20 | 0/20 | 0/40 |

Neither residual seed improved the baseline's stable-hold count. The stronger
residual seed raised the rail higher on average, but height alone did not predict
stable holding or placement. No training episode achieved the two-second hold;
deterministic evaluation removes action-sampling noise and performed better for
one residual seed. This is consistent with exploration disrupting grasps, but
does not isolate exploration from reward design, training budget or other causes.

**Contact validity caveat:** the primary hold metric limits penetration during
the qualifying interval, not throughout the episode. Every scored lift had more
than 1 mm penetration somewhere else in its episode. None survive a stricter,
post-hoc episode-wide 1 mm filter. Some failed episodes reached approximately
15 mm penetration. Treat the results as controller-development diagnostics,
not validated physical grasps. Contact-model validation should precede a larger
RL run. A subsequent, separately registered experiment could test smaller
residual actions and a lift/hold curriculum before adding placement.

### Corrected zero-initialized residual experiment

`rail_residual_v2.py` keeps the original physics, reward, and nominal controller
fixed, and writes separately to `outputs/frame_assembly/rl_zero_init/`. Its PPO
output weights and bias are exactly zero at initialization. Physical residuals
are capped at 10% of each action dimension: at most 0.5 mm per wrist translation
axis, 0.003 rad per rotation axis and 0.006 rad per finger target per control step,
before the existing total-action limits. Raw policy standard deviation is fixed
by a schedule from 0.05 to 0.30 over 65,536 steps (effective correction noise
0.005–0.03). Learning rate is reduced to 0.0001 and entropy bonus is zero.

Before training, complete zero-output episodes are checked against the baseline
state for state. A separate no-learning comparison evaluates the baseline, old
exploration noise, and the starting/ending new noise levels on 20 paired cases.
Noise draws are paired by seed. These probes isolate noise; the learning
comparison changes initialization, residual bounds, exploration, and learning
rate together, so it cannot attribute changes to any one of those choices.

Two seeds each train for 262,144 steps. For each seed, the baseline and checkpoints
at 131,072 and 262,144 steps compete on 20 separate standard validation cases.
The selection rule prioritizes full-task successes, then stable lifts; a strict
improvement is required to replace the baseline. The choice is saved before
evaluation on 20 fresh standard and 20 harder test cases. Final learned policies
are also reported even if validation retains the baseline. This fallback does
not guarantee non-regression on unseen cases. No grasp curriculum or contact
physics changes are included in this experiment.

```bash
.venv-frame/bin/python -m experiments.frame_assembly.test_rail_residual_v2
# Refuses to overwrite an existing rl_zero_init directory.
.venv-frame/bin/python -u -m experiments.frame_assembly.rail_residual_v2 run
.venv-frame/bin/python -m experiments.frame_assembly.report_rail_residual_v2
```

The pre-run manifest records the exact case IDs, selection rule, and source
hashes. Results retain all evaluation cases, training logs, initial/mid/final
models and corresponding normalizers. The existing whole-episode penetration
caveat remains applicable; stricter episode-wide lift counts are also reported.

The completed corrected pilot verified exact zero-policy/baseline equality in
three full episodes. With no learning, the baseline held successfully in 4/20
validation cases, old exploration noise in 0/20, and both smaller-noise settings
in 5/20. Both smaller-noise settings retained all four baseline successes; old
noise lost all four. During training, the two policies achieved 64 and 83 stable
holds across 324 episodes each. No full placements occurred.

All four learned checkpoints tied the baseline's validation lift count of 4/20,
so both selection decisions retained the baseline. On **fresh** test cases:

| Controller | Standard lift / hold | Harder lift / hold | Full placement |
|---|---:|---:|---:|
| Baseline | 6/20 | 0/20 | 0/40 |
| Final residual, seed 0 | 5/20 | 0/20 | 0/40 |
| Final residual, seed 1 | 6/20 | 0/20 | 0/40 |
| Validation-selected controller, either seed | 6/20 | 0/20 | 0/40 |

The corrected training avoided the previous severe degradation, but did not
show a learned success-rate improvement. Neither the 5/20 noise-only result nor
the one-case difference between final policies supports a general superiority
claim. These test cases differ from v1; do not compare the aggregate percentages
as if the cases were paired across versions. No successful lift passed the
stricter episode-wide 1 mm penetration check. The corrected experiment took
approximately 5.8 minutes locally, excluding implementation and reporting.

## Environment API

```python
from experiments.frame_assembly.environment import FrameAssembly, Config

env = FrameAssembly(Config())  # Board assistance disabled.
obs = env.reset(seed=4)
obs = env.step(
    wrist_position=[0.4, 0.28, 0.43],  # Absolute world coordinates, metres.
    wrist_quaternion=[1, 0, 0, 0],     # wxyz; normalized by the environment.
    finger_angles=env.data.ctrl.copy(),  # 20 angle targets, radians.
)
print(obs['metrics'])
```

Wrist translation is limited to 0.25 m/s; orientation commands are interpolated
towards their target. Finger commands are clipped to the vendor actuator limits.
Actuator names and joint ordering are available from `env.model`. The observation
contains privileged board poses and finger states; it is not a Gymnasium wrapper.
Use `mujoco.Renderer(env.model)` for RGB or depth policy observations.

## Saved outputs

`outputs/frame_assembly/demo/` contains:

- `assembly.mp4`, `initial.png`, `assembled.png`: rendered physics episode.
- `result.json`: explicit scope, release/contact diagnostics, warnings, runtime,
  dependency versions and scene hash.
- `episode.npz`: time, `qpos`, `qvel`, `ctrl`, `mocap_pos`, `mocap_quat`,
  `eq_active`, phase labels, and 64 world-space material-point trajectories at
  the corners of the eight rail collision boxes. These are exact simulator
  points, including internal/occluded points, **not estimated visible tracks**.
- `video_frame_times`: in the episode archive; matches each rendered video frame
  to simulation time. State is saved at 50 Hz and video at 20 fps.
- `scene.xml` and `config.json`: generated scene and all physical assumptions.
  The XML references local vendor meshes by absolute path; regenerate it after
  moving the project.

The dataset uses assisted transport. Do not treat it as a set of unassisted
Wuji grasping demonstrations, or as evidence for the earlier H2 experiment.

## Recreate dependencies

The existing `.venv-frame` is isolated from the earlier robotics experiments.
For a new machine, create a Python 3.12 environment and install `requirements.txt`.
Fetch the official assets with these commands from the project root:

```bash
git clone --filter=blob:none --sparse https://github.com/wuji-technology/wuji-description.git third_party/wuji-description
git -C third_party/wuji-description sparse-checkout set hand2/hand2_2/body/mjcf hand2/hand2_2/body/meshes/right
git -C third_party/wuji-description checkout 829c9aa27cb12c3a49d5b8d63c9869dd8f656702
```

The vendor license is retained at `third_party/wuji-description/LICENSE`.
Configuration and scene construction live in `environment.py`; the assisted
controller and recording code live in `run.py`.

## Astra control experiment

`agent_control.py` exposes persistent observations and bounded action chunks.
The requested GPT-6 Astra model runs as a delegated agent in the current Codex
session. It selects wrist and finger targets, executes them through this bridge,
and inspects the returned state and images before adjusting its commands.
There is no separate API key or model server in the Python environment.

This experiment provides **privileged simulator state** (board poses, fingertip
positions, joint limits and contact forces) alongside two rendered views. It is
not a vision-only policy or real-time 50 Hz model inference. A chunk holds the
chosen target for 0.02–3 simulation seconds while the normal controller and
physics continue. Simulation pauses between chunks while the model reasons;
this does not test real-time inference latency. The ideal wrist carrier remains in use; board transport
assistance is disabled throughout.

```bash
# Create a fresh trial (existing trials cannot be overwritten).
.venv-frame/bin/python -m experiments.frame_assembly.agent_control init --trial new_trial
# Observe without advancing simulation.
.venv-frame/bin/python -m experiments.frame_assembly.agent_control observe --trial new_trial
# Example hand-only action; omitted joint/pose targets remain unchanged.
.venv-frame/bin/python -m experiments.frame_assembly.agent_control act --trial new_trial --json '{"seconds":0.5,"wrist_position":[0.35,0.28,0.4],"reason":"approach"}'
# Independently replay the exact action log, produce video and audit manipulation.
.venv-frame/bin/python -m experiments.frame_assembly.agent_control replay --trial trial01
.venv-frame/bin/python -m experiments.frame_assembly.test_agent_control
```

Trials live in `outputs/frame_assembly/astra/`. Each preserves the original scene
hash, all selected actions and reasons, observations and screenshots. Replay
checks final-state agreement and measures contact, displacement and continuous
airborne contact. An airborne interval requires all rail corners more
than 3 mm above the table, positive hand contact and negligible non-hand support.
This can include slipping. A separate stable-carry metric additionally requires
low translation/rotation relative to the wrist and submillimetre penetration.
The normal full-frame assembly criteria remain unchanged. See each trial's
`result.json` for the measured outcome; controller motion alone is not task success.
