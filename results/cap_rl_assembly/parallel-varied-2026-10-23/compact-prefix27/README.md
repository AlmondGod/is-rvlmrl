# Lossless compact prefix snapshots

The initial whole-state pickle repeated large MuJoCo model payloads inside
several MjData instances. The first compact codec interns identical model bytes;
the final codec also shares identical 64 KiB blocks across the slightly different
models used for collision planning. It preserves the full native data bytes and
all Python state; it does not substitute a reduced integration-state vector.

On the same 20-second local prefix probe, the original snapshot was 894,484,493
bytes. Whole-model deduplication uses 360,861,740 bytes; block deduplication uses
175,416,535 bytes, a 5.10-fold storage reduction. The final snapshot took 1.16
seconds to write and 2.02 seconds to restore. These are one local probe's
measurements, not a full RL throughput result.

Both probes match all 200 by 155 recorded state values and every non-timing
result exactly. Their simulation-deadline failures are retained; a short prefix
is not task success. Two local tests additionally verify native serialization
bytes immediately after restoration, continued dynamics, object aliases and
separation of distinct models. Native MuJoCo serialization includes wall-clock
solver timers, so continued-dynamics comparisons exclude those elapsed timers.

A complete episode comparison with a snapshot at 300 simulated seconds passed: all
6,494 by 155 recorded values and every non-timing result match. Both runs retain
the original strict failure. The continuous run also exactly matches the earlier
full baseline recording. The full snapshot uses 179,173,283 bytes versus the
original 898,214,220 bytes; capture takes 1.40 seconds and restoration 2.12 seconds.
The continuous run takes 603.97 seconds, and the restored remainder 325.19 seconds.
The combined local process peaks at 3,839,852,544 bytes RSS and 4,524,298,032 bytes
of macOS memory footprint. This is not evidence that eight snapshot workers fit
the GPU pod. This codec is experimental and is not installed in the active
learner. Production use still needs memory-budget measurements, cache identity
and prefix-boundary checks, and GPU-host validation. Pickle inputs must be
trusted locally generated snapshots.
