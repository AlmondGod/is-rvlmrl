# Assembly execution speed experiment

Status: the paired full continuous episode is exactly equivalent in all recorded states and all result fields except wall time. The separate known-successful regression also matches exactly and retains strict success; all three isolated paired timing trials match exactly. No new RL training was run.

The reference is commit `1d5ee70`, with the existing 0.5 ms physics step, 50 Hz controller, three 1280×960 RGB cameras at 10 Hz, finite breakable nails, and unchanged success gates. The development layout is training seed 41002. Both runs execute the same saved 22-call model program. They start from reset and run continuously; no episode prefixes, physics results or restored states are substituted.

Changes:

- Preserve the full-resolution RGB images and exact threshold expressions, but avoid copying uint8 RGB to float64 before thresholding. Other image dtypes retain the original conversion.
- Use OpenCV's mask scan to return the same row-major pixel coordinates as NumPy's scan, including intp coordinates and empty masks.
- An explicitly selected execution backend replaces general-purpose three-dimensional cross products and Python contact/saturation loops with equivalent specialized operations. It still computes internal fastener forces and checks contacts and actuator saturation at every integration step.

The backend is available to the full-assembly executor and GRPO trainer through `--assembly-backend vectorized`; the default remains `original`. The trainer saves that choice in run metadata and passes it to every spawned physics worker. The live backend identity and implementation source are included in provenance and cache keys. Test evaluation, record audits, program regeneration and independent physical replay restore the recorded backend. Switching back to the original backend after installing the vectorized implementation requires a fresh process.

The benchmark convenience flag is `analysis/benchmark_assembly_speed.py --optimized`. Full-run experiments use separate frozen checkouts and record the exact driver/backend source. The reference checkout predates the image-processing improvements; selecting the original physics backend in the new checkout does not revert those image improvements. Historical runs must still use their saved source/runtime; their caches and metadata are not rewritten.

The current installer verifies the original controller/force-method source hashes before replacing them. The full-episode candidate was frozen before that fail-closed installer guard was added; its exact source archive is retained separately. The controller and force implementation are the same. `integration-equivalence.json` verifies that all five numerical function syntax trees and all four perception files match the frozen candidate; module relocation and provenance/worker integration do not change their numerical operations.

Validation includes exhaustive 8-bit RGB threshold comparisons, exact mask coordinate order, cross products, force/failure/energy calculations, contact peak ties and nonfinite values, and a 100-control-step physical state comparison. The full-run comparator checks all result fields except wall time and every recorded state array for exact equality. Both task success/failure and the comparison outcome must be reported. Exact equivalence is established for the recorded arrays and results, not an exhaustive proof for every possible program or initial condition.

Full profiled runs overlap on the same host, so their wall-time ratio is an exploratory result under shared load. Inclusive profiler categories must not be summed: environment stepping includes the physics kernel, nail forces and inverse kinematics. A separate isolated benchmark runs the first 20 simulated seconds three times per backend, alternating order. Those prefixes end at the simulation deadline and are timing checks, not successful task episodes.

## Using the implementation

Add `--assembly-backend vectorized` to the documented full-assembly strict-reward training command in `experiments/cap_rl/ASSEMBLY_TASK.md`. For direct physical execution:

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m experiments.cap_rl.assembly \
  --program path/to/program.py --seed 41002 --out path/to/new-output \
  --support-profile contact_and_nails --controller-profile southwest_approach \
  --assembly-backend vectorized
```

Use a fresh output directory. Existing-source provenance checks remain enforced; this flag is not a way to overwrite or relabel an old run.

The integrated local suite passes 89 tests, including cache isolation between backends and propagation to an actual spawned worker. The frozen candidate also passes 68 existing simulator/benchmark tests with its backend installed on the GPU pod, plus its five initial numerical equivalence tests. No timestep, observation resolution/rate, model program, force limit or acceptance threshold changes are included.

## Full paired episode

Training layout 41002 completes all 22 program calls, including assembly and inversion, in 633.10 simulated seconds. The reference takes 2,134.26 seconds (35.57 minutes); the optimized candidate takes 1,275.02 seconds (21.25 minutes), an exploratory 1.674× wall-time ratio under shared host load. All 6,332 recorded rows (155 columns each) are numerically exact, the complete state archives have identical SHA-256 hashes, and every result field except wall time matches exactly. Each run executes 1,266,200 physics integration steps and renders 18,996 camera images.

Both fail the unchanged strict angle gate: 3.0867533885582095° exceeds 3°. This was already the outcome of the historical training record `9818cffb3beae859995f477c983b0c15c0db8604e1783200d2673f33149896ec`. The speed experiment must preserve that failure. Its orchestration completion criterion was corrected after the optimized run finished but before the reference result was available; the timestamped correction is preserved in the evidence archive. No task or reward criterion changed.

A separate integrated-backend run uses known-successful training layout 41001, compared against historical certified record `5c0d16edab574b3bf35ed41a16d4d3290e23706fa1df63159649547b05d5a27b`. Because the historical host/runtime differ, this is a strict-success and trajectory regression, not a controlled latency comparison.

The integrated production backend completes layout 41001 with strict success, 621.50 simulated seconds and the same 2.2499489920295463° peak relative angle. All 6,216 recorded rows (155 columns each), the state archive SHA-256, and all result fields except wall time match the preserved historical successful record exactly. This validates the actual `--assembly-backend vectorized` CLI path, not just the frozen experimental installer.

## Isolated timing and evidence

Three sequential trials per backend, alternating order, execute the first 20 simulated seconds with no concurrent benchmark jobs. Reference seconds: 45.397, 45.366, 45.062. Optimized seconds: 26.639, 26.801, 27.075. Medians are **45.366 → 26.801 seconds**, a **1.693×** ratio (40.9% less time). Every paired prefix has exactly equal recorded arrays and result fields except wall time. This isolated measurement covers early assembly only; it does not establish isolated whole-episode latency or end-to-end GRPO training speed.

The full reference profile attributes 13.85% of elapsed wall time to the MuJoCo integration kernel. Frame vision decreases from 640.23 to 375.78 seconds, nail-force calculations from 570.35 to 248.87 seconds, and the remaining environment-step work (excluding the kernel, nails and inverse kinematics) from 321.99 to 63.92 seconds. These measurements include shared-host load and instrumentation. Parent timing categories are not double-counted in `assembly-speed-summary.json`.

The final runtime-mutation guard also passes the eight focused backend tests after the 89-test integrated suite. GPU regression output records 68 existing tests passing with the frozen accelerated backend installed.

[Evidence release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-assembly-speed-2026-10-23) includes all continuous/prefix state recordings, logs, result files, source snapshots and a per-file checksum manifest. The downloaded archive verifies all 99 manifest entries and SHA-256 `645b6f0abc316f397b662363815852571169cc5c3f28e5f443172e84a5d6a897` (27,775,247 bytes). Compact results and receipts are checked into this directory.

No new model training or held-out policy-improvement claim is part of this speed experiment.
