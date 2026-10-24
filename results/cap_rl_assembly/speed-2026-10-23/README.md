# Assembly execution speed experiment

Status: numerical unit/regression checks passed; full continuous episode comparison is running. No full-episode speedup or equivalence claim yet.

The reference is commit `1d5ee70`, with the existing 0.5 ms physics step, 50 Hz controller, three 1280×960 RGB cameras at 10 Hz, finite breakable nails, and unchanged success gates. The development layout is training seed 41002. Both runs execute the same saved 22-call model program. They start from reset and run continuously; no episode prefixes, physics results or restored states are substituted.

Changes:

- Preserve the full-resolution RGB images and exact threshold expressions, but avoid copying uint8 RGB to float64 before thresholding. Other image dtypes retain the original conversion.
- Use OpenCV's mask scan to return the same row-major pixel coordinates as NumPy's scan, including intp coordinates and empty masks.
- An explicitly selected experimental execution backend replaces general-purpose three-dimensional cross products and Python contact/saturation loops with equivalent specialized operations. It still computes internal fastener forces and checks contacts and actuator saturation at every integration step.

The backend is enabled by `analysis/benchmark_assembly_speed.py --optimized`. Its source is archived by the benchmark driver. It is an experiment-local process patch, not an automatic change to the historical trainer or its caches. Do not install it into a cached trainer without incorporating its identity into that trainer's provenance. The normal image-processing changes are tracked by the existing source manifest, so historical caches cannot silently mix with the new source.

The current installer verifies the original controller/force-method source hashes before replacing them. The full-episode candidate was frozen before that fail-closed installer guard was added; its exact source archive is retained separately. The controller and force implementation are the same.

Validation includes exhaustive 8-bit RGB threshold comparisons, exact mask coordinate order, cross products, force/failure/energy calculations, contact peak ties and nonfinite values, and a 100-control-step physical state comparison. The full-run comparator checks all result fields except wall time and every recorded state array for exact equality. Both successful completion and the comparison outcome must be reported.

Full profiled runs overlap on the same host, so their wall-time ratio is an exploratory result under shared load. Inclusive profiler categories must not be summed: environment stepping includes the physics kernel, nail forces and inverse kinematics. A separate isolated benchmark is needed for a clean latency estimate.
