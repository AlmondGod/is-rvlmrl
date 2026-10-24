# Independent reward-group scheduling

The existing single coordinator waits for one complete reward group before
submitting the next. The new queue can submit multiple groups immediately and
return whichever group finishes first. It preserves checked physical rewards,
in-flight deduplication, source checks and infrastructure-error propagation.
The caller must define policy versions, stale-rollout limits and optimizer
ordering; this module alone is not an asynchronous RL algorithm.

An eight-episode scheduling probe, arranged as four groups, took 54.99 seconds
with the old coordinator and 34.32 seconds with independent submission. Every
paired trajectory and non-timing result matched exactly. These were six-second
early-terminal programs, not completed assemblies or policy improvement.

Running the probe alongside training exhausted the pod's 31 GB host-memory
limit and killed the learner. The physics comparison passed, but the overall
resource deployment failed. The current queue therefore adds a conservative
memory admission check (2.25 GB per new worker plus a 3 GB reserve) and refuses
to coexist with an existing pool owned by the same task adapter. Missing memory
limits are reported as unguarded. Six unit tests cover scheduling, deduplication,
source changes, worker errors and memory admission. The guarded version has not
yet been benchmarked in a new learner.
