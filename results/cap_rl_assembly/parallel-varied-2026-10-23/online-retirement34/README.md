# Prospective four-round no-improvement contingency

Before the fourth validation is observed, declare a cutoff for online25: if
both scheduled learned checkpoints remain at or below the original 4/8, retire
this small-layout-batch configuration at four complete rounds/16 updates.
Otherwise the original eight-round protocol continues unchanged. The next
experiment will freeze a separate full-layout-cohort protocol and any conditional
fastening-order curriculum before execution. All final test layouts stay untouched.

The supervisor validates layout coverage and aggregates, stops only the intended
learner at the completed validation boundary, verifies the full fourth checkpoint,
and refuses retirement if the next physical batch has already begun. It retains
the last adapter, all evidence and a separately hashed retirement receipt. A
partial next model-generation call may have begun; process wall time includes it.
It never claims an exact optimizer continuation or a perfect-validation stop.

The historical record auditor remains unchanged. A separate pre-test auditor
retains its source/reward/selection checks and permits only this independently
verified retirement exception to the original eight-round/ceiling stopping rule.
Three pure decision tests cover non-improvement, improvement, missing evaluations,
wrong aggregates, repeated layouts and an unexpected baseline.
