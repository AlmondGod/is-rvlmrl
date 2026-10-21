# Full-box fixture and online Astra camera control

This run is local; no RunPod is used. It replaces the earlier guided-joint video
as the assembly-control experiment. All five panels have free joints and collide
with the table, hand and fixture. No pick/attach/teleport action is exposed.

## Fixture correction and measured numerical result

The original beveled panel meshes used single-point convex contact, making thin
flat panels unstable on the bench. `multiccd=enable` provides a multi-point
manifold without replacing the bevels, changing friction, or attaching panels.
The numerical regression covers separately staged panels and an explicitly
labelled diagnostic assembled reset, settling and physical fixture retraction,
at 1, 0.5 and 0.25 ms. Six cases passed: no warnings, all finite, maximum observed
panel penetration 0.06149 mm. Diagnostic resets are never counted as assembly.
Report: `outputs/box_assembly/fixture_fix_v1/report.json`.

Tool-stage rotational damping is now computed from the actual tool inertia;
the previous fixed damping could be unstable for such a light body. The bounded
stage can compensate the nail driver's own reaction in the same substep. This
is virtual hardware, not a validated industrial-arm/nailer model.

## Actual model decisions, not a stored assembly script

`astra_camera_bridge.py` owns the simulator and evaluator. A separately spawned
**gpt-6-astra** agent receives only the `controller/` packet directory and the
client `camera_bridge_client.py`. That agent's initial context contains no prior
simulation state or evaluator logs. It examines camera PNGs and chooses each
motor-command chunk online. Commands include the model's written reason. The
server interpolates/rate-limits motors; it contains no pick/place or assembly
controller. Each chunk lasts at most two simulated seconds. Physics pauses while
the model reasons; this is not a realtime-frequency claim.

The public packet is constructed from:
- Four RGB views and their camera calibration.
- Robot wrist pose, finger motor states, fingertip forward kinematics, motor
  targets and fixture encoder positions.
- Tool pose, nose contact switch, driver stroke/force and dispenser encoder.
- Known part dimensions and desired assembly coordinates from the blueprint.

It contains no current panel poses, object velocities, segmentation/depth,
contact force arrays, rewards, bond strength or success labels. Ideal driver
force is tool telemetry, not the contact solver's object-force output. Current
part location must be inferred from pixels. A visual 100 mm tabletop grid is
provided as a calibration aid; this is an instrumented, known-geometry workcell.

The agent is explicitly prohibited from reading simulator source, evaluator
files, previous hidden-state experiments or other chats. This is an audited
information-flow protocol, not an adversarial filesystem sandbox: agents share
filesystem capabilities. Do not claim OS-enforced isolation.

## Outputs and interpretation

Current run directory: `outputs/box_assembly/astra_online_v2`.
The first attempt (`astra_online_v1`) is preserved separately: 17 online commands,
21.6 simulated seconds, no visually confirmed pickup. It was stopped after an
independent empty-hand audit found wrist oscillation at 1 ms. At 0.5 and 0.25 ms,
the same hold test has negligible angular velocity and no wrench saturation.
The second attempt uses 0.5 ms. Static panel-contact regression at 1 ms remains
valid in its limited scope; it did not establish robot-carrier numerical validity.
See `fixture_fix_v1/wrist_hold_audit.json`.
- `controller/`: each exact observation, camera PNGs, static configuration,
  action replies and latest packet.
- `evaluator/`: hidden initial/final scores, per-action physical metrics and
  exact action requests, never supplied to the controller.
- `astra_camera_attempt.mp4`: actual rendered views from action execution.

This is a development attempt, not a held-out success-rate evaluation. Assembly
success requires the evaluator's physical criteria, not a model's verbal claim
or apparent alignment in an image. Full-box fastening remains unavailable: the
fastening extension covers only one front-left joint. No complete nailed/glued
box result may be reported from this implementation.

To run a new local attempt, choose a fresh directory:

```bash
.venv-frame/bin/python -m experiments.box_assembly.astra_camera_bridge --out outputs/box_assembly/astra_online_NEW
```

The server waits for actions. It does not itself call an API, start a remote
machine, create another agent or rent compute. Model authorization/dispatch is
managed in the conversation. Stop through the public client with a reasoned
`{"stop":true,"reason":"..."}` action to finalize the video.

## Completed v2 development attempt

Astra executed 21 online motor chunks, ending at 21.9 simulated seconds including
the initial one-second settle. The 20.9-second video omits reasoning pauses.
The bottom probe and two opposing-edge wall lift tests failed to retain a panel.
Image-derived measurement shows approximately 14 mm of planar wall motion during
the stronger grasp, not a successful pickup. All five panels remained separate;
no nail or glue operation was reached.

The evaluator recorded finite states and zero solver warnings, but maximum
penetration reached 1.1914 mm, exceeding the predeclared 0.5 mm criterion.
**Active manipulation therefore fails numerical acceptance.** The static fixture
contact fix and wrist hold regression do not establish full manipulation validity.
This attempt cannot isolate model shortcomings from remaining contact-model
limitations, and is not a physically validated robotics result.

Exact actions and hidden metrics, public image packets, the controller's own
camera-only diagnosis, source snapshots and runtime provenance are preserved.
See `outputs/box_assembly/astra_online_v2/report.json`. Remaining work is forceful
contact validation and reliable thin-panel pickup, followed by assembly;
complete-box fastening still requires the remaining joints to be implemented.
