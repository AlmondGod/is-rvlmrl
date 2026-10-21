# Torso-mounted continuous frame/wall assembly

This is a new embodiment, not a cosmetic reparenting of validated trajectories.
All three articulated UR10e models descend from one fixed pillar. The earlier
xArm7 model could not reach the far side from a torso mount. Link dimensions,
inertias, joint ranges and actuator force limits come from the checked-in
MuJoCo Menagerie UR10e model. The custom pillar, adapter plates, retained xArm
parallel grippers and nail tool are provisional hardware proposals, not an
approved mechanical integration. Foundation/column stress is not simulated.

The four members forming a 0.9144 m outer frame and 0.9744 m plywood sheet are all free
bodies from the initial scene. Frame support is at z=0.12 m. The plywood waits
on a visible side shuttle. Its motor translates the shelves; friction carries
the sheet. Shelves retract only after the grippers lift it. This is explicit
fixture automation, not autonomous pickup from an arbitrary lumber pile.

Run locally from the repository root:

    .venv-frame/bin/python -m experiments.torso_assembly.run --out outputs/torso_assembly/NEW_NAME
    .venv-frame/bin/python -m unittest experiments.torso_assembly.test_contracts

`--oracle` is a privileged-state diagnostic. It cannot establish camera-policy
performance. The default is a fixed camera-feedback controller, not Astra.
Neither variant trains a model. One MuJoCo state and monotonic simulation clock
span all stages. Runtime workpiece pose writes, welds and applied external
forces are absent. Nail visits are pose-gated abstract events: they do not add
structural strength. No glue curing or physical nail validation is claimed.

The controller uses known dimensions, camera calibration, RGB silhouettes,
robot encoders and fixture commands. Grasped-height propagation assumes no
hidden slip. Collision planning uses a separate robot/fixture CAD model populated with visually
observed workpiece poses. It rejects missing member estimates; nominal/reset
workpiece coordinates are not fallbacks. Before the sheet is observed, its
feeder has a conservative surveyed occupancy envelope. Initialization of robot
joints is permitted only at reset.

Each attempt preserves scene XML, source snapshot, raw joint/object trajectories,
actuator commands/forces, evaluator metrics, video and failure reason. Development
failures must remain distinct from passing regressions. A successful video is
not physical validation or evidence of a general assembly policy.

## Current corrections and checks

The final geometry uses the explicit native `a0_xarm7` / `a1_xarm7` gripper
child defaults. `test_gripper_geometry_defaults_preserved` compares compiled
collision types, sizes and friction against the original model. Earlier runs
with rounded inherited gripper collision shapes have a MODEL_LIMITATION.json
annotation and are development evidence only.

The raised central plinth is 0.92 m square. Eight separate small feeder
supports hold the initially loose lumber at the same level while leaving the
sheet-grasp corridors clear. The sheet carrier is fork-shaped: its connecting
crossbar remains outside the worktable legs throughout its travel.

`preflight.py` checks all four approach/descent paths and both open sheet grasps
against nominal geometry. This is kinematic diagnostic evidence, not a policy
trial. The runtime always executes routes with force-limited motors.

The camera baseline uses three calibrated RGB views, known dimensions, surveyed
support height and robot encoders. Global wood components plus two-view shape
fitting initialize the members without reset positions. Observed side and length
assign task roles, allowing long/short feeder lanes to swap within each side. It checks a vertical grasp probe using the
visible far endpoint. A low front view looks beneath the parked tool to observe
the north member. It uses a stationary-feeder consistency prior and a
no-slip encoder prior after verified grasp; these assumptions can miss hidden
slip or unexpected workpiece motion. The north member uses a 0.235 m carry
height to preserve visibility while retaining roughly 19 mm nominal clearance
over the existing members. Other members use 0.250 m. Diagnostic comparisons
can explicitly use `--north-lift-height .25` to reproduce the original nominal
carry height. These are development choices, not physically calibrated limits.

Runs with `--resume-diagnostic` are explicitly privileged checkpoint tests and
set `continuous: false`. Only fresh runs starting near time zero qualify as
continuous end-to-end demonstrations. `audit.py` checks event counts, the
75 mm virtual nail rays, unassisted sheet lift, final release stability, normal
force balance and numerical limits. Virtual rays do not establish nail holding
strength. `replay.py` renders saved states with brighter presentation lighting;
it never steps physics, changes object poses relative to recorded states, or
stitches episodes.

## Side-staging layout and work camera

The current reset has two parallel loose planks on each side at x=±0.68 m
and ±0.84 m, all initially oriented along Y. Their destination orientations
remain the frame blueprint. Separate side feeder tables support the planks;
the central plinth starts empty. The plywood shuttle starts 1.55 m out to
leave clearance for the right gripper. Post-release parking uses a checked
joint route instead of a direct return to the old feeder pose.

The presentation camera is a closer elevated front view aimed at the work
surface. The inset uses a separate whole-torso view instead of the occluded
overhead sensor. This changes the display camera, not the camera baseline's
calibration or sensor placement. Side-staged camera-policy competence is not
established by a privileged diagnostic demonstration.

Side-staged transport uses a collision-checked joint route. During planning
only, the held cuboid follows a rigid no-slip grasp approximation derived
from the input pose estimate and robot tool frame. Finger-pad contact is
allowed; arm self-contact and carried-object collisions are rejected. This
approximation never constrains the runtime workpiece: the executed simulation
still uses the original free joint and finite-force contacts. Runtime pose
feedback corrects the final placement. A 156-position kinematic sweep checks
the extended plywood shuttle against the side-table fixtures.

## Actual Astra task control

`live.py --out outputs/torso_assembly/NEW_NAME` runs the external model bridge.
A connected `gpt-6-astra` agent reads only `controller/instructions.json`,
numbered RGB observations and robot/fixture encoders. It writes one action
at a time after inspecting current images. Its commands choose members,
release decisions, tool visits, sheet operations and when to stop. The
underlying motion/perception skills remain handwritten. This is a hybrid
controller, not an end-to-end learned motor policy or evidence of training.
The public observation whitelist is tested; access separation is an instruction
contract, not an operating-system sandbox. The `evaluator/` directory retains
object poses, contact diagnostics and acceptance metrics privately.

Physics runs while the model reads and reasons. Planning and image processing
can make the local simulation slower than wall time; measured wall lag and
observation-to-command delays are recorded. These delays include agent tools
and scheduling and are not isolated model inference benchmarks. The baseline
and model harness now both use 10 Hz image updates and 50 Hz motion updates.

The frame tracker propagates a verified grasp with encoder rotation and a
post-jaw-closure RGB/support-plane anchor. New RGB edges correct transverse
position. At the feeder, a previously observed plank has a stationary
longitudinal prior when endpoints are occluded; grasp transport has a no-slip
prior. Those assumptions can conceal longitudinal disturbances, tilt and slip.
Natural wood-color thresholding, known dimensions and known feeder identities
are also strong development assumptions. Camera input alone does not imply
unrestricted perception or real-world readiness.

The early October 19 camera development trials below must not be described as successful
end-to-end assembly. In actual Astra trial `astra_rgb_live55`, the model chose
five image-inspected commands: place/release each left-side member, then place
the rear member. The first two released placement errors were about 1.0 and
2.8 mm, but rear-member lowering hit the 0.5 mm penetration gate. No nail or
panel operations completed. Trial 48 also failed. Source snapshots and all
failures remain preserved under their individual output directories.

The archived trials 56/57 tested an experimental low-view cuboid silhouette fit that estimated perspective
depth and visible long-axis tilt while the rear member is clear of the frame.
During the final descent below 0.23 m, it propagates the last visible grasp
with encoders instead of interpreting merged wood edges as fresh position.
That experimental variant marked `contact_descent_no_slip` in the estimate. It was rejected after unstable pose corrections and has been removed from the active tracker. It cannot
observe hidden slip during that interval and is not an independent sensor
confirmation of contact placement. Ground-truth audits remain mandatory.

The trial58 camera experiment moved camera0 below the torso beam to
(0.35, -0.20, 1.25) m with an 85-degree vertical field of view. This exposes
the rear rail that the original overhead view could not see. At that point the simpler
tracker from actual Astra trial55 was restored; archived depth-fit experiments
are not silently promoted. A 25 mm-radius hypothetical camera housing was
checked against recorded collision geometries; the bracket, lens distortion,
continuous swept clearance and physical calibration remain unvalidated.

The earlier overhead fit (`overhead_pose.py`, superseded near contact) uses natural RGB wood silhouette,
known cuboid dimensions, calibrated perspective and image-estimated yaw.
It rejects fits above 1.5 pixels maximum silhouette residual or 35 mm
translation change instead of trusting a contradictory image. This rejection
is not a calibrated statistical confidence interval. It is a development
consistency check; passing it alone does not prove pose accuracy. An invalid
rear-member estimate makes the motion skill hold and terminate the trial.

Astra commands compound skills, not individual joints or novel grasp poses.
The skill implementation already contains grasp/placement geometry and
motion logic. Claims should say "Astra supervises RGB-feedback skills" rather
than implying that Astra independently learned the manipulation controller.

Previous checkpoint: actual Astra trial `astra_rgb_live59`. Five model commands were issued;
four completed (position/release members 0 and 1). Their final XY errors were
0.653 mm and 1.916 mm. The rear-member placement stopped on an inconsistent
visual estimate, with peak penetration 0.097 mm. No nail or sheet operations
occurred. The full-task audit fails. Three actual model trials in this
iteration produced zero complete assemblies. The latest change improves
failure detection; it does not establish end-to-end competence. See
`outputs/torso_assembly/astra_camera_report.txt` for the evidence and limits.

## Contact-pose correction development (after trial 59)

`multiview_pose.py` fits a known beveled member to two calibrated RGB images.
It uses long-axis extent plus interior cross-sections away from the joining
ends and central gripper occlusion. Connected-component filtering removes
small neighboring-board fragments. The fit estimates translation and rotation;
roll around the long axis is weakly observable and regularized to the previous
verified-grasp/encoder orientation. No evaluator pose, depth buffer or object
segmentation ID enters the fit. It retains residual rejection limits.

The controller uses gain 1 for this pose fit instead of gain 3 to reduce
amplification of camera noise. This is part of the intervention, not an
unchanged-controller comparison. Tests include synthetic tilted RGB geometry,
a neighboring member, central finger occlusion, and rejection when the second
view is absent. Recorded failures provide development error checks only;
they are not held-out data. Acceptance and numerical limits remain unchanged.


## Released-member reacquisition (after trial62)

Trial62 placed and released all four members correctly by independent geometry
checks, but first nailing was blocked because estimates of members0/1 had drifted
while the other arm crossed the view. Partial occlusion shifted the accepted
edge, then the moving search window accumulated the error. The feeder reference
had never been replaced after placement, preventing recovery.

The tracker now records the last observed camera/encoder pose on the logical
held-to-released transition, with Z on the known support plane. Each subsequent
image search starts from this fixed observed release pose. It does not copy the
blueprint goal or evaluator pose. Fresh RGB must produce a consistent estimate;
corrections exceeding4mm or0.03rad are rejected, not used to shift the search.
These are provisional development thresholds, not calibrated uncertainty bounds.
The released reference clears when a new verified grasp begins.

The recorded trial62 sensor replay reacquires both boards after occlusion and
passes the unchanged visual frame check. The replay uses simulator snapshots
only to reconstruct the original camera/encoder observations; evaluator truth
is used only to score the output. This is a diagnostic replay of development
failures, not a fresh policy episode. The output is
`outputs/torso_assembly/stationary_recovery_replay.json`.

This assumes released members remain stationary. It is not a general disturbance
or slip detector. In a synthetic check a20mm transverse shift aliased a truncated
ROI edge and yielded a2.9mm apparent shift; hidden longitudinal motion is also
weakly observable at joined ends. Real deployment needs active inspection,
additional observable features/views and measured uncertainty. The regression
checks missing-image rejection, recovery, and rejection for an80mm shift outside
the search region; it does not claim to detect all smaller shifts.

The nominal and half-timestep fixed-order camera baselines61/63 both pass the
independent full-task audit. They predate this released-reference change and are
not actual Astra trials. Trial64 is the next fresh actual Astra episode; its own
source snapshot and evaluator report determine its outcome.


## Front contact and closed-loop bias (after trials64/65)

Actual64 exposed the same near-contact bias on the front member: roughly10mm
inward placement made the next member perch on its end. The numerical gate stopped
the trial at a gripper-base collision; no limit was relaxed. The two-view fit now
covers front and rear members near their goals, with three of four visible profile
sections required per camera so one fully occluded section does not discard an
otherwise useful view. Side-member observation is unchanged.

Actual65 then showed accurate front translation but slow approach convergence
and drift in the weakly observable long-axis rotation. The roll regularizer now
uses a fixed camera/encoder grasp-relative orientation, propagated with encoders,
rather than recursively adopting the last fitted roll as a new independent prior.
A bounded XY integral (gain0.5/s, +/-3mm) compensates persistent servo error when
an accepted two-view pose is available in align/lower/settle. Final acceptance,
force limits, scene geometry and the0.5mm numerical gate are unchanged.

Baseline66 and actual67 use this revised runtime. Baseline68 is a predeclared
160-second, half-timestep sensitivity check of the four-member manipulation phase;
its expected deadline termination is not a full-assembly success. Do not mix these
purposes or versions into an IID success-rate estimate. See the final report and
per-run source snapshots, result.json and audit.json for outcomes.


## Final development result: actual Astra trial67

The uninterrupted final-code actual Astra episode passed the independent audit.
It used20 individually selected model commands, including finish; all four loose
members were placed/released, four corner tool visits completed, the plywood was
physically fed/lifted/placed/released, and four panel visits completed. Final member
XY errors were0.718,0.933,0.565,0.395mm, panel XY2.314mm, and peak penetration
0.211mm. All eight abstract nail rays intersected both intended materials. No
physical nail bonds or real-world validation are implied.

This is one nominal development episode, not a held-out reliability estimate.
The gain came from perception and feedback-controller changes. Astra still selects
and visually accepts compound handwritten skills. Physics continued during model
reasoning, but12.76 simulated minutes took16.73 wall minutes; this is not a real-time
execution claim. Snapshot-to-command age had median10.13s and p95=14.53s, including
model, image-reading and tool overhead rather than isolated neural inference time.

See `outputs/torso_assembly/contact_pose_report.txt`,
`outputs/torso_assembly/astra_contact_final_summary.json`, and
`outputs/torso_assembly/astra_rgb_live67/evaluator/audit.json`.
The final four-member half-timestep check68 and baseline66 use identical core
runtime hashes. Thirteen synthetic/contract tests pass. All preceding failed
trials are preserved and described in the report.

## Blind starting-pose evaluation

`blind_pose_protocol.txt` freezes the bounded layout distribution and acceptance
criteria. `--layout-seed` is evaluator-owned and is never put in the public
Astra observation packet. Randomization includes within-side lane permutation,
±12 mm X, ±35 mm Y, and ±0.06 rad yaw. Workpieces remain face-up on surveyed
supports; arbitrary piles and unknown support heights are outside this test.

Active nail targets are computed in `tool_skills.py` from visually estimated
members and the sheet plane. Sheet center height is recovered from calibrated
RGB PnP (including the top-face-to-center offset), then propagated from the
visually initialized grasp. A no-slip assumption remains. Hidden joint strength
is not inferred from this motion demonstration.

Run the boundary and image regressions:

    .venv-frame/bin/python -m unittest experiments.torso_assembly.test_blind_pose experiments.torso_assembly.test_multiview_pose experiments.torso_assembly.test_contracts

Replay the development setup with a NEW output directory:

    .venv-frame/bin/python -m experiments.torso_assembly.run --out outputs/torso_assembly/NEW_BLIND_RUN --layout-seed 3 --north-lift-height .25 --seconds 650

`run` chooses a fixed skill sequence and is not an Astra trial. `live` requires
an actual external model reading the controller images and issuing individual
commands. Model file access is governed by an instruction contract, not an OS
sandbox. The observation/controller code and evaluator share a process, so this
is an audited information-flow boundary, not a security isolation guarantee.

## Optional breakable nails and experimental flip

`--physical-nails` enables ray-gated instantaneous insertion and finite-strength
point-spring connections. These are paired internal loads, not welds or object
attachments. This is not calibrated nail driving, bending, wood damage or glue.
`--flip` appends a two-gripper regrasp/lift/180-degree-turn/hold test. The original
continuous run73 failed during initial lifting: one connection broke and contact
penetration exceeded the unchanged 0.5 mm gate. That run remains a failure; see the repair evidence below.

    .venv-frame/bin/python -m experiments.torso_assembly.run --out outputs/torso_assembly/NEW_NAIL_FLIP --layout-seed 3 --physical-nails --flip --seconds 650 --north-lift-height .25
    .venv-frame/bin/python -m experiments.torso_assembly.audit_flip outputs/torso_assembly/NEW_NAIL_FLIP

The live runner accepts `--physical-nails --flip` and exposes `flip_structure`,
but this new skill has not passed an Astra-controlled trial. Physics/evaluator
state remains outside the public sensor packet, including latent nail condition.
See `nail_flip_protocol.txt` for provisional parameters, assumptions, tests and
physical measurements needed. Detailed evidence is in
`outputs/torso_assembly/physical_nails_flip73/REPORT.txt`.


## Flip repair development (2026-10-19)

The `clearance-stack` profile (default with `--flip`) grasps the frame and sheet
stack, uses a support with side access slots, and selects 1.10 m shoulder-center
spacing on the same torso. The previous spacing was 0.68 m. **This is a fixture
and mounting-layout change as well as a controller change.** Native arms, jaws,
friction, actuator limits, and every original nail-strength parameter remain
unchanged. `--flip-profile legacy` retains the former geometry/profile by default;
`--shoulder-half-span` explicitly selects a mounting width for diagnostics.

The controller lifts clear of the support, translates clear of torso/feeder
obstructions, raises the assembly and rotates both wrists with encoder feedback.
Airborne waypoint tolerances account for measured servo compliance; they do not
relax the final retention/penetration gates. There is no runtime workpiece state
input, hidden rigid attachment, nail healing, or physical glue in this test.

The final continuous camera-feedback baseline `physical_nails_flip83` passed the
independent audit from loose parts through eight fasteners and a 180-degree flip.
All eight original-strength connections survived; peak penetration was 0.1997 mm.
Maximum relative motion throughout airborne movement was 0.9465 mm / 1.6089 deg.
This is a scripted skill sequence, not a new Astra-controlled trial.

Panel transfer also required an 18 mm carry lift to clear shoulder brackets,
a consistent grasp-time image height reference, and collision-checked joint-space
retreat after release. The clearance-stack profile now defaults to a 0.5 ms step.
Matched checkpoint tests82/81/84 at 1/0.5/0.25 ms respectively failed/passed/passed;
the 1 ms test exceeded the unchanged penetration gate. The two finer steps agreed
within 0.091 mm and 0.042 deg in final relative member poses. Unfastened and 1 N
negative controls85/86 lost members before floor impact.

Earlier checkpoints24/25 passed a hold-only audit but briefly exceeded the
3-degree airborne retention criterion. Their stricter audits are failures;
original hold-only audits are retained. Trial74 failed panel transfer. All trials
are development evidence on one layout, not an IID reliability study.

Reproduce the matched positive checkpoint with a fresh output directory:

    .venv-frame/bin/python -m experiments.torso_assembly.flip_diagnostic --source outputs/torso_assembly/panel_clearance_checkpoint79 --start-time 423.20 --out outputs/torso_assembly/NEW_FLIP_DIAGNOSTIC --grasp stack --smooth --clearance --fixture-half-width .38 --shoulder-half-span .55 --dt .0005
    .venv-frame/bin/python -m experiments.torso_assembly.audit_flip outputs/torso_assembly/NEW_FLIP_DIAGNOSTIC --checkpoint

Use `--dt .00025` for refinement, or `--condition unfastened` / `--condition weak`
for negative controls. Checkpoint initialization is privileged and labeled.
`diagnose_flip_sweep` is a privileged kinematic check, never a physics success.
The legacy profile is not a byte-exact old runtime; run73 preserves that source.

Run the complete regression suite:

    .venv-frame/bin/python -m unittest experiments.torso_assembly.test_flip experiments.torso_assembly.test_fasteners experiments.torso_assembly.test_blind_pose experiments.torso_assembly.test_multiview_pose experiments.torso_assembly.test_contracts

See `nail_flip_protocol.txt` and `outputs/torso_assembly/flip_repair_trials.json`.
Nail laws, pad grip/moment capacity, rigid plywood, and mounting structure still
need physical characterization. Passing this simulation does not validate a real
nailed joint or establish that Astra can independently complete the task.
