# Bounded speculative execution of training programs

The balanced online experiment retains its frozen sampler, optimizer, source
and outcomes. A separate optional scheduling helper can use two otherwise spare
CPU slots on the pod's 10.2-core quota while its eight rollout workers run.

The helper queues the frozen warm policy's verified greedy program for each of
the 28 future training layouts, in the already-declared training schedule order.
It excludes the first four layouts already in progress and every validation or
test layout. The program report binds the model revision and warm-start weights;
it compares ordinary and experimental chunked greedy decoding before reuse.
No additional successful program is hand-authored or selected from outcomes.

The helper uses the exact same physical runtime and checked evidence cache.
The existing per-identity interprocess lock prevents duplicate execution if a
learner requests a program already being prefetched. A precomputed reward enters
GRPO only when the on-policy sampler independently requests that identical
program/layout/source identity. There is no predicted reward, modified sample,
changed advantage, extra optimizer update or validation-data training.

At most two speculative episodes run concurrently. Stop queuing when training
exits or writes selection, and finish already-started episodes. Preserve every
candidate, outcome, physical certificate and cache-hit status. Record this
work separately, including unused prefetches; elapsed times overlap the same
continuously allocated GPU and must not be added as extra GPU-hours. This is a
scheduling hypothesis, not a demonstrated whole-loop speedup until measured.

The completed decoder probe matched all 32 training programs but changed one
of eight validation programs. The global decoder-equivalence check therefore
failed, and chunked decoding is not installed into the active learner. Prefetch
uses only the 32 individually verified training candidates; its metadata retains
the failed global check alongside the successful training-candidate check. No
validation candidate is queued or used as training data.
