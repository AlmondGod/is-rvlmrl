# One joint: decisions, evidence, and validation

Scope: one front/left butt joint within the existing five-panel box. One nail
and a 9 × 30 mm nominal adhesive region sampled at four patches. These dimensions
are provisional, especially the small nail edge distance in 9 mm stock. This is
not a validated construction detail. Do not infer structural suitability.

## Decisions and present audit

| Decision | Existing workcell | Added implementation | Remaining evidence/mechanism |
|---|---|---|---|
| Position and clamp | Free panels, collision, finite-force fixture | Joint-relative mating gates; actual contact-derived pressure estimate | Real fixture geometry/compliance; local pressure distribution |
| Drive nail | Absent | Contact/alignment-gated, bounded-force, bounded-energy feed; equal/opposite tool reaction | Actual nail/tool/wood force traces; impact dynamics if using a nail gun |
| Judge seating | Absent | Limited driver stroke; nail head visualization; separate inspection decision | Camera head-depth measurement and actual driver sensor accuracy |
| Wait for adhesive | Absent | Finite application, open time, continuous clamp history, exponential cure | Product-specific kinetics and handling strength measurements |
| Release support | Fixture retracts; only dry-fit criteria | Latent model readiness distinct from observed proof-test result | Calibrated proof load, actual release/creep tests |
| Retry | No fastening state | Contact-loss retry can continue an intact partially driven nail | Validate allowed retries and tool capacity |
| Repair/reject | Absent | Damage/expired glue irreversible; conservative reject/repair decision | Extraction, surface cleaning, new nail location and repair tooling are NOT implemented |

No whole-box fastening success is exposed. No industrial arm is added. The Wuji
hand and tool carriers are finite-wrench virtual stages. There is no controller
that autonomously completes this operation yet. No RunPod access is used.

## Constitutive approximation and parameters

`fastening.py` is simulator-independent. `fastening_environment.py` adds it to the
original workcell as `FasteningBoxAssembly`; `BoxAssembly` remains the dry-fit
baseline. The four ablations are `unfastened`, `nails_only`, `glue_only`, `combined`.
All numerical defaults in `parameter_provenance.json` are unmeasured placeholders,
not vendor specifications or literature-derived measurements.

Nail feed is a quasistatic resistance law:

`F = R0 + k_drive * depth + c_drive * depth_rate`.

Constant-force substeps are integrated analytically, then capped at nail length,
remaining driver work budget, and shot duration. Firing requires actual nose
preload, actual tool alignment, and mating geometry. A nail creates no connection
until it penetrates the receiving member. It then acts as an isotropic spring
with separate withdrawal/lateral strength limits scaled by embedded length.
Relative displacement or mixed-mode load can break it permanently. This is a
lumped joint law, not a simulated metal shank penetrating deformable wood. It
omits splitting, plastic nail bending, head pull-through, impact/hammer inertia,
fatigue and nail mass. Driving to a fixed length also omits overdriving damage.
Those omissions require rejection criteria or new models if measurements show
they determine decisions. A quasistatic press is not evidence for an impact gun.

Glue is deposited only where the actual nozzle reaches the nominated surface
patches, within a distance/orientation window. Flow and excess waste are counted.
Full patch coverage, permitted gap/alignment and continuous local clamp estimate
are required before a bond forms. Unbonded glue expires after open time; applying
more does not clean or repair it. There is no pre-gel tack force in this version.
Cure fraction is `1-exp(-ln(20)*age/t95)`; stiffness and strength grow with it.
There is no temperature/humidity kinetics model yet: only a fixed reference
condition is supported. Each patch is a finite-strength connector; broken bonds
never heal when panels return to their target poses. Separate and measure initial
handling, full cure, and load capacity rather than assuming a time threshold
establishes quality.

Tool housing/nose contacts are simulated. Glue film and nail shank do not have
separate collision geometry. Nail heads and glue deposits are rendered as simple
visible markers, with appearance independent of latent strength/cure/damage.
Glue can only be applied before the mating panel obstructs nozzle access.
Tool reach, collision-free access and fixture interference are still untested.

The adapter estimates patch clamp load using uniform pressure over the entire
mating face, from panel-to-panel contact normal force. This is an unvalidated
pressure approximation, not a measurement and not a policy observation. The
force solve supplies an explicit substep contact estimate; coupled contact,
fastening and tool dynamics still need timestep convergence testing.

Equal/opposite connector forces act at the same world point, preserving total
force and moment. Isotropic elastic force is along anchor separation; damping
uses relative velocity at the common application point and dissipates energy.
Anchor creation captures the current configuration without changing panel poses.
The energy ledger separates drive work, penetration/viscous losses, elastic
energy, damping, fracture, and stiffness-change work. Cure/embedding can change
stiffness: that energy is explicitly accounted for rather than claimed passive.
Whole-system energy balance with gravity, contacts and actuators is not yet tested.

`chemistry_time_scale=1` by default. Other values accelerate chemical aging,
including open time, but not physical clamp time or robot dynamics; both clocks
are logged. Such a run is a declared scheduling approximation, not a real-time
cure simulation.

## Controller evidence and unknowns

`camera_observation()` retains the existing RGB/proprioception boundary.
`tool_observation()` is a separate versioned packet containing tool pose, a nose
contact switch, ideal driver stroke/force and dispensed volume. No bond strength,
damage, object pose, exact contact pressure or success label is included. The
ideal internal stroke currently equals penetration in the constitutive model;
real compliance/slip must be identified before trusting that equivalence.
`decisions.py` deliberately requires a separate observed nail-head measurement.
The old camera worker does not yet consume this tool packet.

Camera-visible: panel alignment, exposed head protrusion, gross split/gap and
accessible glue coverage. Tool-observable: commanded/observed stroke, preload
switch, drive-force signature, dispensing history. Potentially hidden: buried
nail trajectory/bending, internal splits, adhesive wetting and strength, local
clamp pressure and weak bonds. A credible handling judgment needs a controlled
probe or proof load with support still available. Even a passed small proof load
cannot certify higher load capacity or long-term durability.

`decisions.py` is a conservative evidence-contract baseline, not learned vision.
Missing evidence causes inspection; absent material validation causes a hold;
visible damage or a failed proof test causes reject/physical-repair referral.
It never takes evaluator metrics. It does not implement a physical repair by
clearing failure flags. Normal episode reset is the only state reset.

## Smallest useful characterization pilot

First obtain the exact panel product/species, nail product and placement, glue
product, actual fastening tool, and fixture. These are still unspecified. Measure
specimens before fixing the design; if stock splits at this edge distance, revise
the construction, not the simulator's success threshold.

Use matching two-panel corner coupons preserving the intended thickness,
grain/end-grain orientation, joint geometry, nail edge distance and glue area.
Retain offcuts and label parent board/batch. Condition materials together, record
moisture and temperature/humidity, and randomize test order within stock blocks.

Equipment: calibrated load cell (or six-axis force sensor), displacement encoder
or extensometer, controlled loading stage/materials-testing machine, rigid fixture
with measured compliance, scale for glue mass, moisture meter, temperature/humidity
logger, synchronized close-up video and tool command/pressure/current logs.
Choose sensor range from a sacrificial pilot so loads are resolved without clipping.
Characterize signal bandwidth; collect at least ten samples across the shortest
relevant event, with anti-alias filtering and a common clock. Measure synchronization
error. A slow test's sample rate must not be reused blindly for an impact tool.
Correct machine/fixture compliance using separate calibration, not the held-out
joint outcomes. Save raw readings as well as processed displacement.

A minimum exploratory matrix is 72 independent destructive specimens:

| Condition | Ages | Loading modes | Replicates per cell | Specimens |
|---|---|---|---|---|
| Unfastened | Not applicable | Withdrawal and lateral | 6 | 12 |
| Nails only | Not applicable | Withdrawal and lateral | 6 | 12 |
| Glue only | Intended handling and full-cure ages | Withdrawal and lateral | 6 | 24 |
| Nails + glue | Same two ages | Withdrawal and lateral | 6 | 24 |

Choose actual ages from the selected product and calibration pilot, then freeze
before held-out testing. Use three calibration and three held-out specimens per
cell, from separately reserved stock boards/batches (36/36 total). Include
replicates across boards; do not split samples from a single force trace between
training and test. This is an exploratory pilot, not a reliability demonstration.
Intermediate cure ages require additional calibration coupons if a one-parameter
cure law fails; two ages do not identify arbitrary cure kinetics.

On every nail-containing specimen, record driving force, displacement, speed,
tool command and seating image. For a quasistatic driver, use at least two
controlled speeds across independent calibration specimens so viscosity and
resistance can be separated. If the real tool is impulsive, obtain the impulse
and motion traces first; the existing drive fitting script explicitly rejects
that regime. Record application mass/coverage, assembly/open time, clamping force
history and cure environment on every adhesive-containing specimen.

Before destruction, execute the same predeclared supported handling/proof-load
procedure and measure residual displacement, release stability and creep over a
fixed duration. Then load monotonically to failure in the assigned mode; retain
video and fracture surfaces. Label withdrawal, wood split, nail bending/head
pull-through, adhesive/interface failure, or mixed failure, preferably blinded
to the predicted outcome. Unfastened controls characterize contact/friction and
fixture effects; do not label their friction peak as adhesive strength.

For recovery ranking, reserve additional matched coupons after calibration.
Create one controlled fault (for example an intact proud nail or shortened cure),
assign credible alternatives (inspect/re-drive, wait, reject) before testing,
and compare subsequent handling/load outcomes. Start with three independent
matched replicates per action only as a pilot. A broken joint cannot serve as
multiple independent recovery trials. The initial 72 specimens alone do not
validate recovery ranking or low failure probabilities.

## Predeclared acceptance and model comparison

These are proposed research tolerances, not construction standards. Freeze them
before held-out outcomes. First establish measurement repeatability better than
one-third of the applicable error tolerance; otherwise improve instrumentation
or explicitly revise the protocol before registration.

- Force–displacement curve normalized RMSE <=15% per specimen, normalized by
  measured peak force with a 5 N denominator floor for weak/unfastened controls.
- Peak-load error <=20% with that same normalization.
- Failure displacement error <=max(0.5 mm, 20% of measured failure displacement).
- Separate seating-depth calibration target: <=0.5 mm absolute error. This is
  not automatically measured by the current curve-validation script.
- Failure-mode accuracy >=80%; supported-release classification accuracy >=90%.
- False-safe rate: one-sided 95% upper bound <=5% among recommended releases.
  With zero false-safe outcomes, at least 59 independent recommended-release
  tests are needed; the pilot cannot usually meet this gate. Correlated stock
  requires batch-aware confirmation beyond the simple binomial calculation.
- On prescribed constitutive histories, changing integration steps from 1 to
  0.5 to 0.25 ms changes reported failure displacement by <=1%; drive-work balance
  residual <=1e-10 J. Coupled MuJoCo trajectories need a separate convergence
  study on pose, impulse, load and release outcome; these offline checks do not
  establish it.

Compare an unfastened baseline, a constant-strength finite spring/breakable joint,
and the cure-dependent process model on exactly the same reserved specimens.
Give the constant model its own calibration fit. Justify added complexity only
if median relative curve-error improvement is >=10% with a positive paired
bootstrap 95% lower bound, and relevant decision errors do not worsen. The
implemented bootstrap is specimen-level; use stock-batch resampling in a larger
confirmation study. Report all four physical controls separately as well as
aggregate performance. A rigid weld is useful as a negative control, not a
credible physical baseline.

Freeze code, parameters, specimen IDs/splits, test inputs and predictions with a
SHA-256 manifest and an external timestamp/repository commit before opening the
held-out measurements. Store prediction curves over the planned displacement
ramp, failure mode/displacement, and release recommendation. Do not fit after
seeing a failure and then count that same specimen as validation. A second person
or independent acquisition/test script should check specimen IDs, raw traces and
labels. Automated reports cannot establish that data is authentic/prospective.

## Reproduction and data formats

Offline checks do not compile MuJoCo or run robot dynamics:

```bash
.venv-frame/bin/python -m unittest experiments.box_assembly.test_design experiments.box_assembly.test_fastening experiments.box_assembly.test_decisions experiments.box_assembly.test_validation
.venv-frame/bin/python -m experiments.box_assembly.numerical_validate --out outputs/box_assembly/fastening_numeric_NEW
```

An authorized non-RunPod Linux host is needed for the pending workcell check:

```bash
PHYLA_CLOUD_RUN=1 MUJOCO_GL=egl python -m experiments.box_assembly.validate_fastening_runtime --out outputs/box_assembly/fastening_runtime_NEW
```

It compiles the model, checks uncommanded separated/assembled diagnostics and
renders tool geometry. It does not demonstrate active nailing, glue dispensing,
fixture release under bonded loads, or autonomous manipulation. Those must be
validated after this first smoke check, with recorded commands/contact traces.
No external compute is launched by any of these scripts.

`measurement_templates/specimen.json` lists raw provenance fields.
`drive.csv` feeds `calibrate_drive.py INPUT.csv --out NEW.json`; the fitter rejects
held-out rows, impact tools, negative fits and rank-deficient/ill-conditioned
parameter identification. It fits drive resistance only, not connector or cure
parameters. Fit initial load slopes, peak loads and cure curves from separate
joint tests; preserve uncertainties and parameter correlations.

For `validate_physical.py --observed FILE --predicted FILE --out NEW.json`:
- Each observed record needs `specimen_id`, `split="heldout"`, `stock_batch`,
  `condition`, `mode` (`withdrawal`/`lateral`), `age_group` (`handling`/`cured` or
  `not_applicable`), monotonic `displacement_m` and `force_N` arrays,
  `failure_displacement_m`, `failure_mode`, and measured `release_stable` Boolean.
- Each prediction has the same ID, predicted curve, failure displacement/mode,
  and `recommend_release` Boolean, under `models.cure_dependent` or
  `models.constant_joint`. Include calibration specimen/batch IDs, frozen
  parameter hash and external registration reference at the top level.
- Missing measurements produce a blocked report, never success. Shared stock
  batches, specimen leakage, missing predictions and extrapolation are rejected.
- The script evaluates curve/failure/release criteria only. Seating metrology,
  recovery ranking, repeatability and coupled dynamics are separate required
  checks and must not be inferred from its result.

## Primary references

- [MuJoCo applied-force API](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-applyft): implementation of point forces/reactions, not material validation.
- [USDA Wood Handbook, Fastenings](https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr190/chapter_08.pdf): withdrawal/lateral resistance and construction dependence.
- [USDA Wood Handbook, Wood Adhesives](https://research.fs.usda.gov/treesearch/62250): bond formation, assembly/pressing and performance dependencies.

The chosen linear/exponential laws and numerical defaults are our testable
engineering approximations; these references do not validate those choices.

## Subsequent local runtime diagnostic (2026-10-17)

The user subsequently requested actual simulation/video without RunPod, authorizing
a local run. See `README.md` and `fastening_video.py`. This uses a fixed/guided
two-panel test fixture with the same constitutive laws. It does not replace the
physical measurements or validate unrestricted full-box dynamics. The original
cloud-only/no-local-execution statements above describe the earlier implementation
phase, not the subsequent authorized diagnostic.
