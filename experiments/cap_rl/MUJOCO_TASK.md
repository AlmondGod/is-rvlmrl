# MuJoCo code-as-policy RL: inertial docking

The model writes a short program that applies forces to a 1 kg planar slider,
stops at station 1, and then stops at station 2. This is a deliberately small
continuous-dynamics task in **MuJoCo 3.3.7**, independent of the repository's
robotic frame-assembly benchmark. The slider has two translational joints and
two force motors. It is not an articulated robot, contact-rich manipulation,
vision task, or evidence of real-world transfer.

## Program and physics

- `thrust(fx, fy)` applies `(0.4*fx, 0.4*fy)` N for 0.5 seconds; each argument
  must be a literal integer in [-4, 4]. Both axes act simultaneously.
- `dock()` checks the next station: Euclidean position error <= 1.5 cm and
  Euclidean speed <= 2.5 cm/s. It then coasts for 0.25 seconds and checks position
  again. The call never snaps position, attaches the puck, or zeros velocity.
- The program has at most ten calls. A typical solution uses six: accelerate,
  brake, dock, accelerate, brake, dock. Python AST validation rejects imports,
  attributes, assignments, expressions, loops, and arbitrary execution.
- MuJoCo integrates each 5 ms timestep with RK4. There is no gravity, friction,
  damping, automatic position servo, or state reset between calls. The model
  receives exact initial coordinates, zero initial velocity, both target
  coordinates, and the known force/duration calibration in its prompt.
- A boundary breach at any timestep (|x| or |y| > 0.48 m) terminates execution.
  Walls outside that boundary are visible, but valid trajectories do not need
  collision handling. Final success also requires a passive 0.5 second coast
  ending within the final position and speed thresholds.

`mj_step` determines all trajectory states and reward measurements. The prompt
documents the analytic constant-force motion and the displacement of an
equal/opposite pulse pair. After an unsuccessful initial pilot, it also supplies
the six-call accelerate–brake–dock structure for both legs. This is a fixed
controller scaffold, not a completed program for the supplied layout: the model
must compute the four pairs of force arguments. The experiment tests using that
known calibration and scaffold in generated programs; it does not test discovering
the control strategy or unknown physics. The executor still permits up to ten
calls, so alternate valid programs are accepted.

## Reward

For each station j, while it is the next undocked station:

```
P_j = max over simulated timesteps of max(0, 1 - distance_to_station / initial_leg_distance)
S_j = max over simulated timesteps of max(0, 1 - distance_to_station / 0.05)
                                    * max(0, 1 - speed / 0.1)
D_j = 1 if dock() successfully accepted that station, otherwise 0
reward = sum_j (0.10*P_j + 0.05*S_j + 0.30*D_j) + 0.10*strict_success
```

Distances are meters and speeds are meters/second. The first leg starts at the
initial position; the second leg starts at station 1. Reward ranges from 0 to 1.
Docking only at station 1 earns 0.45 but fails the task. Runtime errors retain
credit earned by the executed prefix, stop execution, and prohibit success.
Syntax errors execute nothing and earn zero. Token-truncated completions earn
zero in the trainer, even if their executed prefix made progress. No reward is
given for prose, matching a reference program, or merely producing valid syntax.

## Data and training protocol

Seed 20261025 draws 704 unique ordered triples of distinct coordinates from a
5 × 5 lattice spaced 10 cm apart (each axis -0.2 to 0.2 m). The fixed split is
512 training, 64 validation, and 128 test layouts. Full layouts are disjoint;
individual relative motions can recur across splits. This measures new layout
combinations within one known dynamics family, not distribution shift.

Training starts from Qwen2.5-Coder-1.5B-Instruct, without either symbolic-task
adapter. The independent CaP-RL-style GRPO implementation uses on-policy sampled
programs, execution rewards, rank-16 LoRA, and KL to the immutable base model.
No demonstration training, teacher outputs, or oracle programs are used.
The test-only `oracle` confirms feasibility; the trainer never calls it.
Checkpoint selection uses validation success, then validation reward. Test
evaluation happens once after selection, with identical greedy decoding for
the base and selected model. All programs and physical traces are retained.

## Reproduce

Python 3.12 with CUDA-enabled PyTorch 2.8.0:

```bash
pip install -r experiments/cap_rl/requirements-mujoco.txt
# Physics does not need OpenGL. Render separately with an available GL backend.
export MUJOCO_GL=disable
export HF_HUB_ENABLE_HF_TRANSFER=0
python -m unittest experiments.cap_rl.test_mujoco -v
python -m experiments.cap_rl.train train --task mujoco_docking \
  --out /path/to/new_run --steps 120 --eval-every 20 --max-tokens 128 \
  --lr 0.0001 --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a
python -m experiments.cap_rl.train test --out /path/to/new_run
python -m experiments.cap_rl.audit /path/to/new_run
python -m analysis.cap_rl_audit_records /path/to/new_run
```

The audit requires the recorded MuJoCo version. It re-executes generated programs
and compares flags, rewards, trajectories, final physical state, and aggregate
results, with 1e-8 tolerance for cross-platform floating-point roundoff. Replay
verifies execution/accounting; it does not independently prove model provenance.
Model checkpoints, frozen source, hashes, and generation configuration are
retained to make that provenance inspectable.

For a side-by-side video rendered by MuJoCo, with exact generated code:

```bash
unset MUJOCO_GL  # local supported graphics; use egl on suitably configured Linux
python -m analysis.cap_rl_mujoco_video /path/to/new_run --index 0 --out /path/to/video
```

The first test case is the default, not a selected best case. The video freezes
a terminated episode at its actual final state and labels its error. Positions
and velocities are re-executed from the saved source program; frames are actual
MuJoCo renders, not a learned or invented animation.
