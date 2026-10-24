# Assembly execution speed experiment

Status: numerical unit/regression checks passed; full continuous episode comparison is running. No full-episode speedup or equivalence claim yet.

The reference is commit `1d5ee70`, with the existing 0.5 ms physics step, 50 Hz controller, three 1280×960 RGB cameras at 10 Hz, finite breakable nails, and unchanged success gates. The development layout is training seed 41002. Both runs execute the same saved 22-call model program. They start from reset and run continuously; no episode prefixes, physics results or restored states are substituted.

Changes:

- Preserve the full-resolution RGB images and exact threshold expressions, but avoid copying uint8 RGB to float64 before thresholding. Other image dtypes retain the original conversion.
- Use OpenCV's mask scan to return the same row-major pixel coordinates as NumPy's scan, including intp coordinates and empty masks.
- An explicitly selected experimental execution backend replaces general-purpose three-dimensional cross products and Python contact/saturation loops with equivalent specialized operations. It still computes internal fastener forces and checks contacts and actuator saturation at every integration step.

The backend is available to the full-assembly executor and GRPO trainer through `--assembly-backend vectorized`; the default remains `original`. The trainer saves that choice in run metadata and passes it to every spawned physics worker. The live backend identity and implementation source are included in provenance and cache keys. Test evaluation, record audits, program regeneration and independent physical replay restore the recorded backend. Switching back to the original backend after installing the vectorized implementation requires a fresh process.

The benchmark convenience flag is `analysis/benchmark_assembly_speed.py --optimized`. Full-run experiments use separate frozen checkouts and record the exact driver/backend source. The reference checkout predates the image-processing improvements; selecting the original physics backend in the new checkout does not revert those image improvements. Historical runs must still use their saved source/runtime; their caches and metadata are not rewritten.

The current installer verifies the original controller/force-method source hashes before replacing them. The full-episode candidate was frozen before that fail-closed installer guard was added; its exact source archive is retained separately. The controller and force implementation are the same. `integration-equivalence.json` verifies that all five numerical function syntax trees and all four perception files match the frozen candidate; module relocation and provenance/worker integration do not change their numerical operations.

Validation includes exhaustive 8-bit RGB threshold comparisons, exact mask coordinate order, cross products, force/failure/energy calculations, contact peak ties and nonfinite values, and a 100-control-step physical state comparison. The full-run comparator checks all result fields except wall time and every recorded state array for exact equality. Both successful completion and the comparison outcome must be reported.

Full profiled runs overlap on the same host, so their wall-time ratio is an exploratory result under shared load. Inclusive profiler categories must not be summed: environment stepping includes the physics kernel, nail forces and inverse kinematics. A separate isolated benchmark is needed for a clean latency estimate.

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
