# Interrupted online run and exact replay recovery

Online run 13 completed one 16-program rollout round and four optimizer updates.
Ten programs succeeded; two of the four layout groups contained both successes
and failures. The round took 2,348.41 seconds, including 34.54 seconds of updates.
This is a sampled training result, not a paired policy improvement.

An additional scheduling benchmark exhausted the pod's 30,999,998,464-byte host
memory limit and the OS killed the learner. This was a resource-planning error.
The saved first checkpoint and certified physics results remain intact. Eight
orphaned physics workers are allowed to finish their in-flight evaluations;
the two speculative workers finish their current work and stop.

The recovery launcher waits for those workers to become idle, terminates only
the owned idle processes, and requires 24 GB of working-set headroom. It runs
one learner and eight physics workers, with no overlapping benchmark, second
model, or speculative worker pool.

Recovery starts from the same original warm adapter and seed, regenerates the
first batch and replays its four AdamW updates using verified physical rewards.
Before continuing, it requires identical prompts, programs, rewards, evidence
hashes and a bit-identical first checkpoint. It does not pretend an adapter-only
checkpoint contains optimizer or random-number state. The interrupted attempt
and reconstruction cost remain separately recorded. Recovery is pending until
its verification receipt says it passed.

No post-training validation result or held-out improvement has been observed.
The 24 final test layouts remain untouched.

An independent check is available as
`python -m analysis.audit_assembly_reconstruction --run RUN --out AUDIT.json`.
It binds the reviewed recovery source, preserved interrupted records, unchanged
protocol, exact first 16 programs and results, four optimizer update metrics,
and checkpoint bytes. It supplements the physical-record audit. Four tamper
and valid-prefix tests pass; the actual recovery receipt remains pending.
