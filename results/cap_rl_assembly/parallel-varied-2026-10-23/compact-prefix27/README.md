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

A complete episode comparison with a snapshot at 300 simulated seconds is
running locally. This codec is experimental and is not installed in the active
learner. Production use still needs memory-budget measurements, cache identity
and prefix-boundary checks, and GPU-host validation. Pickle inputs must be
trusted locally generated snapshots.
