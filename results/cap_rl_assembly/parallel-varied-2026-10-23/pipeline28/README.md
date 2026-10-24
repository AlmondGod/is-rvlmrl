# Nonblocking rollout component

The experimental dispatcher generates a constrained program batch on the GPU,
submits its real physical evaluations immediately, computes frozen old/reference
log probabilities, and returns CPU tensors without waiting for physics. Several
groups can be submitted before collection. It retains policy-version labels,
real worker errors and duplicate/cache attribution; no synthetic reward is used
to bypass a blocking call.

Four CPU fixture tests pass, covering dispatch-before-log-probability ordering,
frozen score storage, collection of later completed groups, worker failures and
parameter-version guards. This is not yet integrated into a learner or tested
with a real Qwen rollout. It makes no whole-loop throughput or policy-improvement
claim. A future learner must explicitly define policy staleness, update order,
validation selection and memory admission. The active frozen run is unchanged.
