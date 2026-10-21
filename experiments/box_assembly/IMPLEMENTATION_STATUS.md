# Full fixture and online camera control — 2026-10-17

Multi-point convex contact corrects the thin-panel support instability. Six
settling/retraction cases (separated and diagnostic assembled initializations at
1, 0.5 and 0.25 ms) are finite, warning-free and peak below 0.062 mm penetration.
This is a numerical fixture regression, not autonomous assembly. All 39 tests (including the local wrist hold regression) pass after the changes.
A separate robot-carrier audit found empty-hand oscillation at 1 ms. The default
is now 0.5 ms and agrees with 0.25 ms in that hold test. This is distinct from
the panel/fixture contact regression. The first online attempt is preserved;
a fresh second attempt uses the corrected timestep.

A separately initialized GPT-6-Astra controller is now making online commands
using four RGB cameras, known blueprint/calibration and robot/tool proprioception.
Panel states and scores stay in evaluator files. Physics pauses while the model
reasons. The run starts with five free, separated panels. No attachment or
teleport command is exposed. The corrected attempt finished after 21 online commands with no verified pickup
or assembly. Strong grasp contact reached 1.1914 mm penetration, failing the
0.5 mm numerical criterion despite finite states and zero solver warnings.
Thus active-manipulation validity remains unresolved. No fastening was reached.
Video and exact evidence are in `outputs/box_assembly/astra_online_v2/`.
See ASTRA_ONLINE.md.

# Local simulation update — 2026-10-17

The later request explicitly authorized running locally without RunPod. A real
MuJoCo guided-joint run and 38.16-second video are complete:
`outputs/box_assembly/nail_glue_video_v1/nail_and_glue.mp4`.

One panel is fixed in a visible test jig; the other is constrained to a linear
clamp guide. The same nail/glue constitutive model is coupled to actual colliding
tool proxies. A shot stopped at a 0.5002 mm mating gap; increasing clamp preload
and retrying seated the nail to 25 mm without resetting it. All four adhesive
patches bonded after the required continuous hold and reached about 95% modeled
cure under explicitly accelerated chemical time. No solver warnings; maximum
contact penetration 0.355 mm in this rig. No physical validation or autonomous
camera-control claim. The full box fixture compiled/rendered but exceeded its
penetration limit in the initial assembled run; that initial failure is superseded by the fixture correction below.

Runtime, source hashes, logs and assumptions are preserved beside the video.
The nail-driving inset is a labeled diagnostic cutaway; cure is evaluator-only
telemetry. Twenty-one constitutive tests passed after the logging changes.

## Earlier pre-runtime record

# Fastening implementation status — 2026-10-17

Implemented one front/left box joint, not complete-box assembly. Source:
`fastening.py`, `fastening_environment.py`, `decisions.py`.

- **Code:** reduced-order driving, tool reactions, seating, finite nail/glue
  connectors, application, open time, clamping, cure, irreversible failure,
  evidence-based decision baseline, calibration and held-out comparison tools.
- **Offline validation:** 38 tests passed. Python/XML/API static checks passed.
  Prescribed drive/withdrawal histories at 1, 0.5 and 0.25 ms agree on seating and
  failure displacement. Drive energy residual <=2.7e-15 J. These numbers describe
  placeholder laws, not measured joints or coupled robot dynamics.
- **Coupled runtime:** unexecuted. No MuJoCo model compilation, tool-contact
  trajectory, rendered camera verification or robot rollout was performed.
- **Physical validation:** blocked; zero measurements and no exact wood/nail/glue/
  tool product selection. The held-out report explicitly returns false.
- **Compute:** no RunPod and no external compute started. Local execution was
  limited to source/static checks and offline mathematical/unit tests.

The defaults are not material recommendations. The current drive model is
quasistatic; it must not be advertised as an impact nailer model. A simulated
ready flag does not authorize hardware release. Repair does not erase failure:
extraction, surface cleaning and replacement tooling remain unimplemented.

See `FASTENING_PROTOCOL.md` for decisions, approximation limits, the exploratory
72-specimen measurement matrix, predeclared tolerances and data formats.
Machine-readable results are in `outputs/box_assembly/fastening_status.json`,
`outputs/box_assembly/fastening_numeric_v2/`, and `outputs/box_assembly/fastening_static/`.
