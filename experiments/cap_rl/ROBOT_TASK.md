# Articulated robot: grasp and lift one frame rail

This isolates the first manipulation operation in wood-frame assembly: center-grasp a
91.44 × 3.8 × 6.4 cm wooden member and lift it by the requested 6, 8, 10, or 12 cm.
It is a new small benchmark, not a result on the full frozen assembly benchmark.

The scene reuses the repository's native xArm7 asset with seven articulated arm
joints, its actuated two-finger linkage, and a fixed base. The rail is a free body
with density 600 kg/m³, gravity and contact friction. Its rectangular geometry
approximates the frame member; bevels, joining, other arms and vision are omitted.
There are no wood welds, pose writes after reset, external assistance forces, or
ideal wrist carriers. The native gripper has three internal linkage equalities.
MuJoCo 3.3.7, 2 ms implicit integration, 50 Hz joint servo; friction and mechanics
are provisional simulation parameters, not hardware-calibrated measurements.

## What the language model controls

The prompt provides privileged exact rail-center coordinates and initial gripper
coordinates. The model writes bounded API calls: `move(dx,dy)` in relative cm,
`lower()`, `close()`, `lift(h)` in relative cm. A fixed, engineered IK/joint servo
executes them through motors. Lowering uses the calibrated support-plane height;
it does not correct the XY choice. Closing applies the native gripper command;
it does not attach the wood. The prompt supplies the four-call sequence.
The learned problem is parameter selection/arithmetic and valid code generation,
not learning perception, motor control, or discovering a grasp strategy.

## Reward and independent success

Across completed API calls and the final hold, maintain these maxima:

* alignment: `max(0, 1 - XY_tcp_to_wood_center_distance / 0.06)`;
* contact: finger-pad normal force >5 N AND central error ≤12 mm;
* elevation: clipped fraction of requested rail lift, only while central and in contact.

Reward is **0.15 alignment + 0.20 contact + 0.15 elevation + 0.50 success**.
Strict success requires every 20 ms observation of the final 0.5 seconds to have:
central error ≤12 mm, rail height within 8 mm of the requested height, tilt ≤0.10 rad,
linear speed ≤0.02 m/s, angular speed ≤0.10 rad/s, and finger contact >5 N.
Any physics warning or peak penetration ≥3 mm disqualifies success; penetration
also zeros reward. Parser failures and truncated generations receive zero.
Runtime errors preserve earned progress but cannot succeed. No sequence-only
success: the actual rail must remain physically held at the specified height.

## Evaluation plan

Seed 20261026 fixes 256 training, 64 validation, and 128 test layouts. Gripper XY
is sampled from integer centimeters [-3,3]² and rail XY from [-4,4]², relative to
the fixed reference (0.38,-0.21) m. Layout tuples are disjoint; component values
can recur across splits. The oracle is only a feasibility check, never training
data. Select adapters on validation only, then run one paired greedy held-out test.
Keep source hashes, robot asset hashes, sampled programs and execution traces.

Run from the repository root with the MuJoCo training dependencies:

```sh
python -m unittest experiments.cap_rl.test_robot
python -m experiments.cap_rl.train train --task robot_lift --out /opt/cap-rl/robot01 \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --steps 80 --lr 0.0001 --max-tokens 96 --eval-every 20
python -m experiments.cap_rl.train test --out /opt/cap-rl/robot01
```

Audit every saved result and render a paired test episode:

```sh
python -m experiments.cap_rl.audit /opt/cap-rl/robot01
python -m analysis.cap_rl_audit_records /opt/cap-rl/robot01
python -m analysis.cap_rl_robot_video /opt/cap-rl/robot01 --index 0 --out robot-before-after
```

Headless physics can use `MUJOCO_GL=disable`; rendering requires a working OpenGL
backend (macOS offscreen rendering was verified). The recording uses actual
MuJoCo joint states at 10 Hz, with real-time playback and terminated episodes
frozen. The audit allows 1e-7 absolute trace roundoff, because `acos` near zero
tilts differs by about 2e-8 rad across CPU platforms; reward tolerance is 1e-9.
These are replay comparison tolerances and do not change task success gates.
Identical deterministic programs on the same layout can share one audit execution;
every stored record is still checked, with unique executions and cache checks
reported separately. Robot XML, meshes and license files are hashed too.
