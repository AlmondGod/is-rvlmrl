WALL PANEL STATION — CAMERA/ENCODER CONTROL, ABSTRACT FASTENING

This extends the four-member frame experiment. A 974.4 x 974.4 x 12 mm
plywood proxy covers the 914.4 mm square frame, leaving 30 mm overhang for
side-gripper access. Density 600 kg/m3 gives mass 6.836 kg, provisional.
The sheet is rigid; bending, nail resistance, glue and structural strength
are not simulated. One abstract nail event per underlying member is allowed.
Events never attach or weld objects; lifting the resulting wall would not
hold its parts together. This is positioning and tool-sequencing evidence.

Two native xArm7 parallel grippers clamp opposite sheet edges. The frame
rests on an explicit 120 mm plinth to give fingers/forearms clearance. Two
visible motor-driven staging shelves support the incoming sheet above the
frame and withdraw after visual lift verification. A third inverted xArm7
carries the nailgun on a motor-driven vertical overhead carriage. Geometry,
contacts, joint limits and bounded motor actuation are enabled. No workpiece
pose is changed after reset and no hidden forces or attachments are used.

BOUNDARY OF THIS DEMONSTRATION
The frame is initialized from the saved final poses of astra_live_v3 in
outputs/frame_assembly, translated onto the raised fixture. Its previous
assembly is a separate completed camera/Astra run. Transferring that frame
and supplying the sheet to this station are not simulated. This is NOT a
continuous from-loose-lumber-to-wall demonstration. The fixture and overhead
mount differ from the original frame station and are explicitly evaluated
as engineering changes, not a pure policy improvement.

CONTROL AND PERCEPTION
The actual Astra live mode selects position, release, four separate nail
visits and finish after viewing current camera images. Fast motion skills
are handwritten; no model training or residual RL occurs. Physics continues
during reasoning. Fixed-sequence camera baseline and privileged diagnostics
are labeled separately. Runtime policy gets no simulator object poses.

Known camera calibration and sheet dimensions are used to fit its natural
plywood outline in RGB. A hue threshold and near-horizontal pose prior are
narrow assumptions, not general perception. RGB provides XY and yaw. After
clamping on the calibrated level rack, wrist encoders propagate sheet height
and tilt; a visible lift is required before withdrawing support. This can
miss hidden grip slip. Encoder propagation was necessary because monocular
outline depth had a 5–18 mm bias in development. Recorded evaluator poses
are used only to measure that error, never for camera control. The diagnostic
runner intentionally uses true sheet pose and must not be scored as policy.

Tool paths use known robot/fixture geometry, joint/rail encoders and estimated
sheet pose in a separate planning model. Nominal frame blueprint locations
are priors; covered frame members are not visually reobserved through the
sheet. Pose-gated firing is near the surface, not a force-controlled driving
process. A virtual 75 mm ray can verify intersection with sheet and member.

REPRODUCE FROM REPOSITORY ROOT
.venv-frame/bin/python -m unittest experiments.wall_panel.test_contracts -v
.venv-frame/bin/python -m experiments.wall_panel.run --out outputs/wall_panel/unique_camera
.venv-frame/bin/python -m experiments.wall_panel.run --live --out outputs/wall_panel/unique_live
.venv-frame/bin/python -m experiments.wall_panel.diagnostic --out outputs/wall_panel/unique_diagnostic
The live runner requires an actual external image-reading model controller;
it does not generate model decisions itself. Its public controller directory
contains the task, allowed inputs, schema, camera frames and observations.
Private evaluator directories contain raw state/actuator arrays, trace,
source snapshots and hashes. The file boundary is instruction-enforced,
not an OS sandbox. All work is local; no RunPod.

Parameters inherited from native Menagerie assets retain their license.
Wood friction, pad friction, contact compliance, servo tracking, panel density
and fixture geometry are provisional. No hardware measurements were supplied.
The next physical work must measure real sheet mass/flatness, grasp access,
grip/slip and tracking at load, and camera calibration before transfer claims.
Fastening strength needs separate physical characterization.

Acceptance and held-out budget are in protocol.json. Success must include
independent lift, released stability, alignment, all four tool visits,
force balance and numerical checks; a convincing video alone is insufficient.
