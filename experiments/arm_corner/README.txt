ARTICULATED, MARKER-FREE CORNER ASSEMBLY
Local simulation experiment; no RunPod. 2026-10-17.

DELIVERED TASK
Three physically articulated, fixed-base xArm7 chains replace the floating
Cartesian tool carriers. Arms0/1 retain the native parallel grippers; arm2
carries a mounted, parked nailer proxy. Two free 38 x 64 x 914.4 mm members
are aligned into one unfastened butt corner, with the reference member initially
supported by the table and gripped by arm1. Arm0 picks, reorients, approaches,
lowers and releases the other member; arm1 subsequently releases. The crate
scale remains 3 x 3 x 2 ft. There is no complete crate or fastening claim.

ROBOT PROVENANCE AND PHYSICAL LIMITS
Existing repository asset:
third_party/3-PoinTr/simulation_assets/ufactory_xarm7/xarm7.xml
Its README identifies it as the MuJoCo Menagerie simplified xArm7 description,
derived from UFACTORY's public URDF. BSD-3-Clause license remains with the assets.
Primary source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7
Vendor specifications: https://docs.xarm.ufactory.cc/8.technical_specifications.html
These links identify the model family, not calibration of this simulation.

Retained: arm inertias, joint ranges, collision meshes, joint friction/damping,
actuator force ranges, native gripper four-bar linkages and pad contacts.
The only free joints belong to the two wood bodies. Six native hand linkage
constraints remain; none attach to wood. Robot links are joint-actuated; no
external body force, mocap weld, object snap or post-reset pose editing is used.
Bolted arm bases are excluded from world contact; articulated link/world,
self-contact and workpiece contact remain active under native filtering.

Modified: three name-prefixed copies at measured-in-scene but uncalibrated mount
locations; the third hand replaced by a 1.9 kg nailer-shaped rigid tool; finger
pad colors made gray; contact softness/impedance carried from prior work.
Arm0 base was moved from x0.90 to x0.80 m after inverse-kinematics residuals showed
its original layout could not remove the final ~3 mm approach error. This is a
physical fixture-layout intervention, not a policy-score improvement alone.

Joint position targets use damped inverse kinematics, limited target slew
(0.7 rad/s, TCP target0.07 m/s and0.3 rad/s), robot-model gravity/bias compensation
through the existing motors, and bounded integral joint tracking correction
(Ki200; integral term limited to8 Nm). Motor force ranges are unchanged in the
final model. No actual hardware servo identification has been performed. Slew
limits constrain commands, not a certified bound on physical transient speed.
The gripper command is180/255; its mechanics convert actuator effort to pad force.
Native actuator limits do not identify real jaw pressure or wood crushing limits.
An unsuccessful development attempt reduced the hand effort limit to3 and failed
to close against its linkage/friction. That change was reverted, not hidden.

CAMERA POLICY BOUNDARY
Runtime inputs: ordinary RGB, fixed camera calibration, nominal wood dimensions,
robot joint encoders/forward kinematics, commands and timestamps. No rendered
segmentation, true depth, latent workpiece pose, evaluator contact forces, true
joint gap or success flag is passed to the camera controller or Astra.
Ground truth is logged solely for diagnostics/evaluation. Oracle runs are labeled.
The shared process/filesystem is not a security sandbox; APIs and explicit agent
instructions define the observation boundary. A successful interface test is not
proof against arbitrary malicious code reading hidden files.

The fast marker-free estimator uses a calibrated overhead RGB camera. Two oblique
cameras are also supplied to Astra for visual inspection. It thresholds wood
appearance, fits the projected cuboid outline using known dimensions, and uses
a face-up tabletop prior at initial acquisition. Under a gripper the center is
occluded but long edges and ends remain visible. Close to the corner, the nearby
ends overlap in projection: the fit uses the far end and long edges, extrapolating
the known length. After a visually verified probe lift, height during end occlusion
is propagated from measured gripper motion; this is a grasp-retention assumption,
not a direct visual height measurement or hidden simulation-state input. RGB is
still needed for lateral position/yaw; missing RGB never returns a ground-truth
fallback. Reference height remains at the known support plane when not lifted.
A slip in height during this occlusion may not be detected. That is a transfer gap.

There are no colored workpiece markers. This is nevertheless a narrow appearance
baseline: simple rectangular brown wood, fixed lighting/background, ideal camera
calibration and known dimensions. It is not general bare-wood perception under
arbitrary texture, dust, clutter, warping, lighting or partial visibility. No neural
vision model, behavior-cloning policy or residual RL was trained here.

CAUSAL DEVELOPMENT RECORD (ALL FAILED RUNS PRESERVED)
- diagnostic_v1: native arm physically picked/held the member; alignment timed out
  with loaded tracking error. Added bounded joint-feedback integral correction.
- diagnostic_v2: overly small hand effort cap failed the visually checked lift.
  Restored imported hand limit and lowered desired closure from210 to180.
- diagnostic_v3: privileged-state diagnostic physically completed the corner.
  This establishes feasibility in the proxy, not camera competence.
- camera_v1: pickup succeeded; reference/moving outlines overlapped and the
  estimator rejected an apparent height inconsistency.
- camera_v2: naive masking of the other member corrupted endpoint fits and
  produced unstable estimates; added far-end/long-edge fitting during overlap.
- camera_v3: motion completed but outer faces were ~17 mm off. Bright top-face
  pixels saturated red and failed the r/g threshold, leaving only a thin side-face
  stripe. The corrected color gate includes those visible top pixels.
- camera_v4: geometry estimate improved; final IK residual exposed reach-boundary
  trouble. Moved arm0 mounting base 10 cm toward the operation.
- camera_v5: final geometry passed, but the runner ended before the required
  uninterrupted released-stability interval was met. Added a fixed three-second
  terminal observation period for every baseline, not an evaluator-controlled
  extension or a changed acceptance threshold. Original failure remains archived.

ACTUAL ASTRA CONTROL VERSUS BASELINES
run.py uses a scripted skill sequence to isolate motor/perception feasibility.
It is explicitly NOT an Astra rollout. live.py has no automatic task sequence.
A real GPT-6-Astra controller agent reads public RGB and robot readings, chooses
one operation at a time, writes a command, waits for completion, inspects fresh
images, then decides again. Skills available: stabilize, pick, align_corner,
release, stop and finish. The model chooses sequencing and visual acceptance;
the geometric alignment and joint motor control are hand-written feedback skills.
It does not output individual joints or synthesize the fast controller online.

Astra's public instructions, exact commands, reasons, observation indices, camera
images and robot readings are archived under astra_live_v1/controller. The private
evaluator receives no replies from the controller. The agent is explicitly told
that primitive 'done' is not joint acceptance. Object-level skill commands have
consecutive IDs and reject stale observation references. Physics advances while
Astra reasons; the fast joint servos hold the setpoints. End-to-end decision timing
includes scheduling, image loading and tool execution; it is not isolated model
inference latency. Wall-clock lag is recorded, so local continuous simulation
is not misrepresented as hard real-time execution.

ACCEPTANCE, TESTS, REPRODUCTION
protocol.json was created before new evaluation: minimum50 mm pickup clearance,
gap0-5 mm, outer-face flush error<5 mm, squareness<2 degrees, height difference
<3 mm, released stability>=2 s, overlap<0.5 mm and no solver warnings. Numerical
and policy outcomes are separate. Final gap is an evaluator centerline metric;
full face mating, crushing, nail holding and adhesive strength are not evaluated.
Small initial XY/yaw perturbations are used, not arbitrary frame designs.
Development seeds0-2; reserved seeds7100-7104. Do not tune on reserved outcomes.

Five tests check: articulated chains and only two free workpieces/no marker geoms;
native linkage constraints do not attach wood; finite contact dynamics and motor
force limits with no external force writes; black RGB has no fallback; probe lift
requires observed object rise. See test_contracts.py. Additional outcome and
half-timestep checks are recorded separately, not assumed from these tests.

From /Users/almondgod/Repositories/phyla with .venv-frame activated:
  python -m unittest experiments.arm_corner.test_contracts
  python -m experiments.arm_corner.run --out outputs/new_camera_run --condition camera --seed 0
  python -m experiments.arm_corner.run --out outputs/new_diagnostic --condition oracle --seed 0
  python -m experiments.arm_corner.run --out outputs/new_halfstep --condition camera --seed 0 --dt 0.0005
  python -m experiments.arm_corner.live --out outputs/new_live_run --seconds 600
The live runner requires an actual external controller using its public directory;
it never fabricates a model answer. All output directories must be new.
Dependencies are recorded in ../crate_corner/requirements-lock.txt; local assets
remain under their original license. This was executed on the local Mac.

Raw baseline rows: time, qpos[nq], qvel[nv], ctrl[nu], actuator_force[nu]. NPZ files
include dimensions. trace.json contains commands, observed robot state, camera
estimates, hidden evaluator state, contacts and IK residuals, clearly separated by
field. controller_rgb.npz contains every overhead image at10 Hz; calibration.json
specifies projection. Video shows all three views. Live public per-second images
and10 Hz video preserve what the agent could see, with private raw logs/metrics.
All development failures and videos remain in outputs/arm_corner.

REAL-WORLD LIMIT
The model is now articulated, but neither that nor a successful video establishes
real-world transfer. The intended robot purchase, actual dimensions/mass, jaw pads,
wood variability, reach/clearance, actuator tracking and cameras still require
measurement. Start with ../crate_corner/PHYSICAL_PROTOCOL.txt, add actual arm joint
tracking, joint torque/payload logs and tool-center/hand-eye calibration. Probe
wood slip explicitly when the far end is occluded; do not assume the height model
is observable from overhead RGB alone. A real matched lift/placement experiment
and independent gap/squareness metrology are required. Physical validation=false.

ACTUAL ASTRA TRIAL RESULT (astra_live_v1, development seed1)
Accepted six image-driven commands: stabilize arm1 at29.90 s; pick arm0 at58.70 s;
align_corner arm0 at84.20 s; release arm0 at121.14 s; release arm1 at142.30 s;
finish at166.38 s. Agent inspected new views between commands. No primitive failed.
Hidden final evaluation: gap1.567 mm; outer-face mismatch2.503 mm;
squareness0.04858 degrees; height mismatch0.667 mm; released stability23.80 s.
Peak contact penetration0.0563 mm; zero motor saturation substeps and zero solver
warnings. Pickup and strict final acceptance both pass. One trial, no rate claim.
Physics was not paused for model reasoning. Maximum wall-clock scheduling lag
was0.231 s, with4208 cycles behind their nominal schedule. Simulation reached
166.38 s; total wall time including final log/video serialization was169.06 s.
The command timestamps include image inspection and orchestration, not isolated
model inference. Video is continuous; preview_4x.mp4 is explicitly4x playback.
This demonstrates camera-supervised sequencing with hand-written visual servo
skills on this proxy. It does not establish general Astra grasp planning, learned
motor competence, fastening strength or real-world reliability.

FROZEN EVALUATION COMPLETED: IMPORTANT FAILURE EVIDENCE
Scripted camera baseline: 2/5 reserved cases passed the exact acceptance audit.
Wilson95 interval: 11.8%-76.9%. This is not an Astra success-rate estimate.
7100: failed outer flush8.83 mm; 7101: failed outer flush7.19 mm;
7102: passed; 7103: lowering timeout with44.90 mm flush error and tilted member;
7104: passed. All five achieved a verified pickup; no carried grasp loss>=100 ms
was detected at50 Hz. This is not proof of absence of micro-slip or rotation.
At first release in7100, image-estimated flush error was7.38 mm and evaluator
error7.78 mm. In7101, those were5.31 mm and5.80 mm. Thus most of those errors
were already visible before jaw opening, rather than arising only after release.
The lower skill terminates on height without a final lateral-convergence gate;
the scripted sequence then releases. This is an identified implementation limit.
7103 tilted approximately several degrees under contact. Final estimated
center height40.3 mm versus evaluator66.1 mm demonstrates failure of the face-up/
grip-height assumption during corner interaction. The lower timeout preserved
the failure. Additional cameras are available to Astra, but the fast estimator
does not yet fuse them to resolve roll/pitch. These findings are from raw logs;
causal effects of proposed fixes have not been tested. Do not label this solved.
The held-out controller was not changed or retested to improve these scores.
Next revision should gate release on fresh image-based gap/flush/squareness,
correct errors before or during lowering, and detect unexpected tilt using the
oblique views. It requires a new frozen test set; these five seeds are retired.
Final development camera and oracle runs passed. Halving physics timestep
from1 ms to0.5 ms preserved success; final moving-member position differed
0.412 mm. All source hashes remained unchanged through the frozen suite.
The exact audit requires>50 mm lowest-corner clearance held1 second, both
grippers released for>=2 seconds, and no detected carried grasp loss>=100 ms.
See outputs/arm_corner/acceptance_audits.json and audit_results.py. This audit
implements the predeclared requirements more explicitly than the runner summary.
Status: articulated software and continuous live Astra sequence implemented;
narrow numerical checks passed; one camera-only Astra example passed; reserved
baseline repeatability insufficient; reliable assembly and real transfer unproven.
