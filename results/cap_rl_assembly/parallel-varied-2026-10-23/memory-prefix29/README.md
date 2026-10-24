# Experimental native-memory snapshot copy

The trusted in-memory codec copies each distinct MuJoCo model once and copies
full MjData with the installed native `mj_copyData`. Models remain independent
from the live environment, including mutable planner geometry. Object aliases,
forces, userdata and warm-start state are retained. Two small tests verify
complete initial native serialization bytes, subsequent dynamics, mutable-model
isolation and distinct-model identity.

A 20-second local probe matches all 200×155 values and non-timing outcomes.
It copies two models and three data objects, with about 229 KB of additional
Python serialization. Capture takes 0.108 seconds; restoration 0.063 seconds.
The 147-byte on-disk marker is **not** the snapshot memory usage. The process
peaks at 2.78 GB RSS and 3.43 GB macOS footprint. Both short executions correctly
fail the simulation deadline; they are not successful assembly episodes.

A complete same-program replay passes: all 6,494×155 recorded values and every
non-timing outcome match, retaining the original strict failure. Capture takes
0.159 seconds and restoration 0.359 seconds at 300 simulated seconds. The
continuous run also exactly matches the earlier full baseline. Whole-run timings
(880.24 seconds continuous, 465.40 seconds restored remainder) include shared-host
load and are not an isolated speed comparison. The process peaks at 3.55 GB
macOS memory footprint. A short different-suffix branch check also passed; a
full branching comparison is running.
This implementation is not installed in the active learner. Linux/runtime and
worker memory validation remain required before deployment.
