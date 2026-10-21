# Wooden box assembly workcell — uncalibrated prototype

The deliverable includes a dry-fit MuJoCo workcell and an optional single-joint
nail/glue extension. See [FASTENING_PROTOCOL.md](FASTENING_PROTOCOL.md) for the
implementation audit, constitutive laws, measurement protocol and acceptance
criteria. Offline tests, fixture regressions and a local guided-joint diagnostic have run;
see below for scope. **No cloud simulation, autonomous robot assembly success, or
hardware-accuracy result has been established.** The existing frame experiments
and camera controller are preserved. The old camera controller designed for one rail remains separate. A new online
GPT-6-Astra controller now selects motor commands from four RGB views and robot
proprioception for this five-panel task; see [ASTRA_ONLINE.md](ASTRA_ONLINE.md).

## Online camera control (2026-10-17)

A separately initialized GPT-6-Astra agent now controls the full five-free-panel
workcell through four calibrated RGB views and robot/tool readings. It chooses
each short motor-command chunk online; physics pauses during model reasoning.
The default integration step is now 0.5 ms: a new empty-hand regression exposed
wrist oscillation at 1 ms, while 0.5 and 0.25 ms agree. There are 39 passing offline
and local wrist-regression tests. Successful autonomous assembly remains unproven.
Details and preserved development attempts: [ASTRA_ONLINE.md](ASTRA_ONLINE.md).

## Local video diagnostic (2026-10-17)

Following the explicit request to run without RunPod, local simulation is now
available through `allow_local=True`. `fastening_video.py` runs a smaller guided
joint fixture using the same `JointProcess`: front panel fixed in a test jig,
side panel on a one-axis guide, colliding tool proxies with bounded stage forces.
It is a scripted mechanism diagnostic, not a free-body or autonomous box rollout.
The first full-workcell diagnostic exceeded penetration tolerance (about 6.6 mm).
That fixture issue is now corrected by enabling multi-point convex contact: six
settling/retraction cases at three timesteps peak at 0.06149 mm. See
[ASTRA_ONLINE.md](ASTRA_ONLINE.md) for the regression and online camera-control run.

```bash
.venv-frame/bin/python -m experiments.box_assembly.fastening_video --out outputs/box_assembly/nail_glue_video_NEW
```

The diagnostic applies glue, closes the guide clamp, drives a nail, increases
clamp preload after an interrupted shot, retries without resetting the nail,
then holds and cures. Chemistry alone is accelerated after bond formation.
The video labels its time-lapse, slowed driving and evaluator-only cure display.
Its nail close-up is an explicitly labeled diagnostic cutaway, not a camera
observation supplied to a controller. Parameters remain uncalibrated; a successful
guided test does not establish real joint strength or solve the full fixture.

## Concrete starting design

Confirmed construction: **separate wooden panels joined with nails and wood
glue**. Dimensions, butt joints, nail type/layout, driving tool and glue product
are still provisional or unspecified. `fastening_spec.json` records the intended
operations and missing calibration inputs. Fastening is implemented only in
`FasteningBoxAssembly`; the default `BoxAssembly` remains the dry-fit baseline.

Assumed open-top wooden box: **300 × 200 × 150 mm**, with **9 mm panels**.
Four butt-jointed walls sit on a full-size bottom. Front/back panels span the
full length; left/right panels fit between them with an explicit 0.3 mm gap at
each end. This is a design choice pending the user's actual box, not a claim
about its construction. A 0.25 mm convex bevel represents softened edges.
Known dimensions/masses can override individual panels; nominal assembly datums
stay fixed so manufacturing mismatches create real gaps or interference.

All five parts are free bodies. The initial panels lie flat in staging areas;
the robot must lift, rotate, align and place them. Collision geometry is the
actual convex beveled panel envelope, and MuJoCo derives mass/inertia from that
geometry and density unless a panel mass is supplied. No invisible latches,
object carry constraints, snapping or magnetic joints are added. The default
dry-fit environment applies no object-directed assistance. The optional fastening
extension applies explicit equal/opposite tool and joint forces. There are no physical screw holes, finger joints or rabbets in this
butt-joint design; a different joint requires different geometry.

## Fixture and robot changes

- Four guided corner fixtures have real inside jaw pads and outside stops.
  Finite-force prismatic actuators raise both pads and stops to release the box.
  Collision geometry stays present throughout retraction. Guides are rigid
  approximations; guide compliance and actual fixture geometry remain unknown.
- Retract the fixture first to place the bottom beneath the jaws. Engage it
  for wall positioning, then retract it to test the released arrangement.
  This is a proposed station sequence, not a demonstrated feasible policy.
- The official Wuji Hand 2.2 finger mechanics, force limits and inertias are
  retained. Finger target speed and command delay are explicit configuration.
- The old unlimited wrist weld is replaced by a **bounded six-axis positioning
  stage**. Cartesian PD applies a capped force/torque to the robot mount only.
  This allows tracking error and saturation under load. It is **not a real
  arm**: joint reach, singularities, arm collision, gearing and dynamics need
  the actual arm model. Default gains, 35 N/3 Nm limits and 40 ms delay are
  uncalibrated assumptions, not vendor specifications.
- Explicit wood/wood, wood/bench, wood/hand and wood/fixture contact pairs avoid
  silently inheriting the maximum friction coefficient from unrelated defaults.
  Contact coefficients and solver parameters remain calibration variables.

## Vision boundary

Two calibrated RGB views and robot proprioception use the previous camera
experiment's packet format. No object pose, reward, contact force, segmentation,
depth or success label is included. Exact object state is restricted to the
evaluator. Current images are ideal simulated cameras; measured distortion,
exposure, latency, noise and appearance variation still need to be added and
validated. Camera frame conventions follow the
[MuJoCo documentation](https://mujoco.readthedocs.io/en/3.3.3/modeling.html).

## Nails and glue: intended assembly sequence

The proposed process is to dispense glue on mating faces, position and clamp
each panel, drive nails while the fixture supports the joint, and retain support
for the glue product's specified handling interval. Nail access to the bottom
and vertical corners must be checked; reorientation or a different fixture may
be required. The optional extension adds colliding nailer/nozzle proxies on separate bounded
virtual stages; no actual industrial arm or manufacturer-specific tool is modelled.

The single-joint extension implements bounded quasistatic nail feed, seating,
withdrawal/lateral response, finite-strength failure and reaction forces. Glue
application, coverage, bondline gap, continuous clamping, aging and irreversible
bond failure use explicit reduced-order laws. Their parameters remain unmeasured.
Physical repair tools and a complete fastening controller are not implemented.
Do not create bonds merely because panels reach target poses. Do not treat the
existing 0.3 mm corner gap as a validated glue joint: revise it against the selected
adhesive's requirements and actual panel tolerances. Actual strengths and cure times remain unknown. The runnable constitutive model
uses explicitly labeled placeholder values, never values fitted to policy success.

Report nailing completion, handling stability and cured load-test acceptance
separately. Final acceptance needs specified loads and failure limits; no such
limits or hardware measurements have been supplied yet.

## What counts as success

The evaluator separates three statements:

1. **Dry-fit placement:** all panels at the design datums, low velocity, walls
   supported by the bottom, the hand released, no simulator warnings, and no
   excessive penetration anywhere in the episode, maintained for three seconds.
2. **Released unfastened stability:** the same geometry remains stable after
   the real fixture jaws/stops retract, with no fixture contact. This does not
   establish joint strength or durability.
3. **Finished box:** unavailable. `full_assembly_success` is always false until
   all required joints, the actual fastener/adhesive operations and physical
   acceptance tests are implemented and validated. Panels touching is insufficient evidence of fastening.

An assembled initialization is only a diagnostic for checking support, contact
and release. It is labelled explicitly and must never count as a policy rollout.

## Accuracy requires measurements

`calibration_template.json` lists the missing observations and acceptance tests.
The critical inputs are actual panel CAD/dimensions/edge profiles, individual
masses, joint/fastener design, the fixture, robot arm, hand response and cameras.

Use repeated real measurements to estimate a parameter range. Test that range
on held-out loads, speeds and contact configurations. Randomization is a
robustness test, not a replacement for physical calibration. MuJoCo's regularized
contact parameters are solver parameters, not a direct wood Young's modulus;
its documented contact behaviour includes possible gradual slip.
[MuJoCo contact modelling](https://mujoco.readthedocs.io/en/3.3.3/modeling.html)

Wood properties, moisture effects, dimensional change, fastenings and adhesive
behaviour depend on material and construction. The
[USDA Wood Handbook](https://research.fs.usda.gov/fpl/wood-handbook) is a reference
for which effects matter, not a measurement of these particular parts.

Unmodelled effects include panel bending/warp, crushing and splintering,
anisotropic friction, impact-tool dynamics, plastic nail bending/head pull-through,
adhesive fluid flow and environment-dependent cure kinetics.
If insertion or fastening depends on those effects, the present rigid-body
model is insufficient; fit and validate a suitable joint/material model first.

## Validation and execution

Initial work was limited to source/static and offline checks. The later explicit
video request authorizes local MuJoCo compilation, simulation and rendering.
No training has been performed for fastening. The new online Astra development
attempt is recorded separately from the scripted mechanism diagnostic.

On the authorized cloud host, after installing the camera experiment's
requirements and providing the pinned Wuji assets:

```bash
PHYLA_CLOUD_RUN=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.box_assembly.cloud_validate
```

The cloud entrypoint requires Linux and the cloud flag. Local entrypoints
explicitly opt in through `allow_local=True`; they do not use RunPod.
Validation compares 1, 0.5 and 0.25 ms integration, checks model attachments,
bounded wrist force, support loads, penetration, release and camera output. It
records results rather than assuming the fixture/contact configuration works.
Timestep convergence needs explicit comparison; completing these diagnostics
does not by itself establish numerical convergence or hardware accuracy.

Geometry-only commands, which do not load the physics model:

```bash
python -m unittest experiments.box_assembly.test_design
python -m experiments.box_assembly.design
```

These generate `outputs/box_assembly/design/design.json` and `box_design.svg`.
The SVG is a nominal geometry schematic, not a physics rendering. Physical
validation has not run. RunPod is excluded by user instruction; no alternative
cloud host has been supplied. No external compute is launched.
