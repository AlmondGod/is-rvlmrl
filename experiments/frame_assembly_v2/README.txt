FOUR-MEMBER FRAME + FOUR ABSTRACT NAIL EVENTS

Scope
Flat frame, 0.9144 m outside square; 38 x 64 mm cross section.
Two 914.4 mm horizontal members; two 834.4 mm vertical members, leaving
nominal 2 mm gaps at four butt joints. All four are separate dynamic bodies.
This is a 3-foot flat frame milestone, not an upright four-panel crate.

Embodiment
Two fixed-base xArm7 chains with native parallel grippers, one xArm7 with a
mounted nailgun-shaped tool. Original Menagerie asset/license is retained at
third_party/3-PoinTr/simulation_assets/ufactory_xarm7.
Handling bases moved to opposite diagonal corners for two-piece reach each.
Nailer base sits inside the open frame for access to all corners. This is a
simulation cell-layout assumption, not a proposed validated production cell;
removal of a finished frame around the central arm is outside this milestone.
All joint and actuator limits and collision geometry remain enabled.
No wood pose changes after reset, no hidden object attachments, no welds.

Abstract nails
One command per southwest, southeast, northeast, northwest corner. A tool
approach, aiming, seating dwell and retraction are physically executed by the
articulated arm. The proposed diagonal axis crosses from the short member
into the long member, representing a toe-nailing placement. The controller
records an event only after visual frame alignment, no gripped members, tool
position/orientation convergence, and dwell. Duplicate corner events fail.
There is NO nail body, drive resistance, recoil, bond, glue, or joint strength.
The event changes a bookkeeping record only. The frame remains four free
pieces held on the table by gravity/friction. Never infer structural assembly
or physical fastening success from the nail counter or video.

Control / information boundary
live.py: actual Astra external decisions from saved current RGB views and
robot encoders. Model selects each pick/place/release/nail/finish separately.
Physics runs during reasoning. Model gets only the controller directory.
Handwritten Skills execute bounded 50 Hz motion feedback; this is not an
Astra-trained motor policy. The controller uses calibrated RGB outlines,
known rectangular sizes, face-up and table-support priors and feeder layout.
The visible far end avoids forearm occlusion; lift is visually verified from
perspective change during a straight vertical probe, then encoder motion
propagates height and longitudinal position while RGB corrects transverse
position/yaw. Once joint ends merge in the image, the tracker retains the
last longitudinal position on release under a stationary-support assumption.
It cannot reliably detect later longitudinal slip with these occluded ends. This can fail under tilt/slip/occlusion; not general vision.
Placement converges in XY/yaw before release, with 4 mm nominal clearance to
avoid pushing down on the table under height-estimation uncertainty. The
small final settling motion is gravity-driven, not snapping.
run.py --condition camera: fixed sequence using the same RGB skills, NOT Astra.
run.py --condition oracle: diagnostic use of true object pose, explicitly labeled.
Evaluator logs contain true poses/contacts; they are not controller inputs.

Parameters / provenance
Dimensions: user approximate 3 x 3 foot footprint; section sizes are design
assumptions. Density 600 kg/m3, wood/table friction 0.5, pad friction 0.65,
0.5 mm bevel, contact solref and servo gains are uncalibrated inherited
simulation parameters. Native arm inertias/limits come from existing asset.
No physical force, slip, nailing, glue or camera measurements were supplied.
No real-world transfer claim is supported.

Reproduce from repository root
.venv-frame/bin/python -m unittest experiments.frame_assembly_v2.test_contracts -v
.venv-frame/bin/python -m experiments.frame_assembly_v2.run --out outputs/frame_assembly/unique_camera_run
.venv-frame/bin/python -m experiments.frame_assembly_v2.run --condition oracle --out outputs/frame_assembly/unique_diagnostic_run
.venv-frame/bin/python -m experiments.frame_assembly_v2.live --out outputs/frame_assembly/unique_astra_run
The live runner needs an actual external camera-based model controller;
starting it alone does not generate Astra actions. See controller/instructions.json.
All computation is local; no RunPod.

Tool routing
plan_tool.py generates collision-checked joint-space routes against known
robot/bench geometry and the nominal blueprint. Its setup uses known initial
robot configurations, not rollout-derived object poses. Joint-space paths
avoid the self-collision seen with naive straight Cartesian approaches.
Runtime uses synchronized bounded joint targets through force-limited motors.
The nailer has a provisional extended narrow nose for clearance. Pose-gated
firing is near the surface; it is not a force-controlled nail-driving action.

Handling-arm clearance
Release retracts first to the outside feeder region, then uses a collision-
checked joint-space route to the known initial park configuration. Planning
uses measured robot joints and camera-estimated member poses, not evaluator
object state. These routes avoid the self-collisions of naive Cartesian
homing and clear the nailer's workspace. Planning latency is logged.

Idle stability
An unchanged Cartesian target reuses its IK solution; joint servos then hold
that fixed joint target. Re-solving a redundant arm at every idle tick caused
slow joint-posture drift and self-contact in development. A regression test
checks fixed idle joint goals. The interrupted first live Astra run is kept
separately and was stopped for this infrastructure repair.

V2 CONTROL ARRANGEMENT
A position command composes the existing camera pickup, verified lift,
alignment and lowering skills without a model-reasoning pause while carried.
The model still selects each member, observes placement, chooses release,
and selects every nail visit. This specifically addresses drift/slip during
the 60+ second suspended hold observed in astra_live_v2. It does not fix
material modeling, add grasp attachments, change the nail model or read
evaluator state. The v1 held-out results remain frozen and unmodified.

Corner measurement correction
Gap and outer-face mismatch are signed distances to the actual rotated face
planes, measured at the corner. The earlier world-axis coordinate difference
ignored face rotation and incorrectly rejected the v1 half-step result. A
rigid-rotation invariance test guards this correction. Original v1 results
and metrics remain preserved; corrected replay results are labeled post hoc.
