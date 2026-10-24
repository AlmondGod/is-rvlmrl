# Full-state prefix replay feasibility

The local prototype serializes the complete simulator, controller, vision,
evaluator, random generators and loop state, then resumes an identical program
prefix. It does not reset object poses or predict a reward. It is not installed
in training or used by the certified reward cache.

The first 20-second comparison matched all 200 by 155 recorded values. A second
comparison captured state at 300 simulated seconds and completed the full
649.3-second assembly sequence. All 6,494 by 155 values and every non-timing
result matched exactly. Both local macOS executions failed the strict inverted
hold; that failure was preserved. These local outcomes are not substituted for
Linux training evidence.

The cold full execution took 597.88 seconds including snapshot creation; the
resumed tail took 322.97 seconds including a 2.51-second restore. Creating the
snapshot took 0.81 seconds and about 898 MB of scratch space. This is a measured
reuse cost for one common prefix, not an end-to-end training speedup. The large
scratch snapshot was removed after verification; full traces, source and
receipts remain in the published 35-file evidence archive. The earlier
interrupted disk-full probe is preserved as a failed attempt.
