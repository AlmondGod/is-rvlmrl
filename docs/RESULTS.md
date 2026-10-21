# What the preserved experiments establish

The frozen v9 pilot has five completed result records and one incomplete case.
No completed model trial passes the full acceptance test. This is a useful
failure analysis, not a demonstration of reliable assembly.

## Protocol and acceptance

The authoritative [protocol](../experiments/assembly_benchmark/protocol.json)
fixes three paired layouts (23001–23003), 32 model calls, a 120-second request
deadline, 1,200 simulated seconds and a 3,600-second wall budget. It requires
eight intact fasteners, stable release, inversion and retention, no solver
warnings, and maximum penetration of 0.5 mm. Airborne relative member motion must
remain below 5 mm / 3 degrees. Full geometry and hold thresholds are in the file.

The runtime manifest is
`e3e7642677e25162b853f1a5d552f89810f5ab281aecf499c78d81508deba952`.
Do not pool prior versions or development layouts with this cohort. The original
protocol's planned n=3 for each model does not override the actual missing result.

## Completed outcomes

| Model/seed | Simulated s | Wall s | Peak penetration mm | Intact fasteners | Strict result |
|---|---:|---:|---:|---:|---|
| Codex/Astra 23001 | 784.60 | 1924.43 | 0.243 | 6 | Fail: two insertion misses |
| Codex/Astra 23002 | 780.58 | 1963.49 | 0.234 | 7 | Fail: one insertion miss |
| Codex/Astra 23003 | 504.40 | 850.34 | 0.542 | 3 at stop | Fail: numerical gate |
| Qwen 23001 | 120.14 | 250.90 | See raw result | 0 | Fail: first-request timeout |
| Qwen 23002 | 120.10 | 339.99 | See raw result | 0 | Fail: first-request timeout |

The remaining Qwen case has an empty output directory and no result, response or
observation. The driver had selected it; whether it reached physics execution
cannot be established from the retained evidence. It is unscored/incomplete.

Among completed records, exact two-sided 95% binomial intervals are 0–70.8% for
Astra (0/3) and 0–84.2% for Qwen (0/2). These are descriptive intervals, not a
claim of representative sampling, robust reliability, or a meaningful ranking.
An intention-to-run accounting has zero demonstrated successes in each planned
three-case cohort, with one unresolved Qwen case; that is not 0/3 observed Qwen
failures. Initial image bytes and public packets match for both completed pairs.

The Qwen timeouts measure failure of the deployed serving arrangement to meet
the deadline. They do not reveal what action it would eventually have produced.
Completed Astra request latency has a pooled median of about 10.33 s and p95
31.26 s. Compare neither this nor total episode duration as a controlled hardware
speed benchmark: local load varied, a sleep-prevention setting was added during
the cohort, and the providers, networks and token-budget mechanisms differ.

## What failed, and what remains uncertain

In Astra 23001, the saved post-hoc ray-gate analysis reports one frame shot with
a 7.893 mm gap along the ray (limit 6 mm) and one panel shot with 17.449 mm tool
standoff (limit 12 mm). For that panel shot the commanded tip was 17.673 mm above
the true simulated surface, while tracking error was 1.833 mm. This supports
investigating surface estimation/contact placement before blaming actuator
tracking alone. In 23002, the missed frame shot had a 7.673 mm ray gap. These
are diagnoses of an uncalibrated insertion surrogate, not proven physical nail
failure mechanisms.

In 23003, contact between `a0_arm_geom17` and `a0_collision_8` reached 0.542 mm
during `panel: lower`. The two names belong to the left-arm namespace. The
shared placement primitive was executing when the gate tripped; the result does
not by itself show that the model selected an invalid high-level action. The
failure was retained without changing physics, acceptance or the policy.

The visible completion of the first two episodes is not a sound-joint result.
Their qualifying inverted-hold timer is zero because the eight-intact-fastener
gate is unmet, even though the visible inversion sequence occurred. Likewise,
the field `fasteners.inserted` counts shot attempts, including misses; it must
not be described as the number of successful insertions.

## Development changes, separate from held-out results

1. A controller handoff dropped its settled target while waiting for a model
   response. Preserving that command removed the measured target jumps. A
   matched delayed development test reduced drift for some members, while one
   member's drift increased. It was not a universal improvement.
2. A delayed scripted test exposed biased single-view rail orientation near a
   joint. The existing two-view silhouette fit was extended to released rails,
   retaining the surveyed planar-support assumption and rejecting missing views.
3. A six-second-delay full scripted development run (`delayed_full_v8`) passed:
   eight intact fasteners, 0.208 mm peak penetration, 1.200 mm / 2.215 degrees
   maximum airborne relative motion, and 12.84 s qualifying inverted hold.
4. v9 changed Codex reconnect-progress parsing only. Physics and controller
   sources match v8; the development physical regression was reused explicitly.

The old v5 and v8 model failures and selected development failures are retained
under `results/historical_failures/`. They are not relabeled as v9 trials. The
v9 acceptance failures remain failures; no held-out retuning was performed.

## Scope and information access

Models choose fixed skills from a strongly structured task prompt. Those skills
already solve most motor tasks and expose explicit sequencing. The benchmark
does not establish image dependence, generated-controller quality, direct-action
quality, learned recovery, or RL benefits. Different pretraining, parameter
counts, image preprocessing, wrapper context, inference hardware and reasoning
budgets remain confounds. Astra identity is the configured Codex alias, not an
independently returned immutable provider snapshot.

The actor receives RGB, calibration, blueprint geometry, encoders and history.
No runtime workpiece truth or latent joint condition is serialized into its
packet. However, the shared controller has calibrated cameras, known CAD and
material segmentation, surveyed face-up supports, visually initialized no-slip
propagation and a stationary-frame prior after sheet occlusion. This is an
audited software boundary, not a general adversarial security guarantee.

Physics steps during model requests, and responses cannot apply before measured
inference latency has elapsed in simulation time. Synchronous planning and slow
local simulation still cause wall lag. This is not demonstrated real-time robot
control. Unintended robot/workpiece contact classification covers the flip at
20 ms intervals; substep penetration covers the full episode. Earlier unintended
contacts and brief contact events are not comprehensively ruled out.

## Physics and evidence limits

Nails use instantaneous ray-gated insertion and finite point springs, with
irreversible failure. There are no unconditional welds or object attachments.
There is no glue in the current torso benchmark. Nail, grip, wood and mounting
properties are provisional and lack physical calibration. The earlier box
package includes nail-driving/glue-cure prototypes and collection protocols;
their presence does not make the current torso task physically validated.

At packaging, `trace.jsonl.gz` was missing for Astra 23001 and 23002. Raw state
arrays, all saved camera observations and responses, videos, results, and
previously produced independent/shot audit reports remain. Those reports are
historical evidence: their full trace-based recalculation cannot currently be
repeated. No trace was synthesized. Astra 23003 and both completed Qwen traces
remain available. The release inventory records hashes and missing files.

Independent audits reconstruct geometry from saved qpos and compare it to
logged poses. Force readings and maximum penetration between recorded samples
remain runtime instrumentation; they are not independently reintegrated force
trajectories. A successful audit verifies this narrower consistency claim.

Actual dimensions, mass, grip/slip, actuator tracking, camera calibration and
joint load tests are still needed for transfer. Start with the protocols in
`experiments/box_assembly/FASTENING_PROTOCOL.md` and
`experiments/torso_assembly/nail_flip_protocol.txt`. No physical measurements,
real robot demonstrations or real-world success rates are claimed here.

