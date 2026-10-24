# Conditional panel-nailing sampler component

This is a proposed curriculum component, not a trained model or deployed loop.
A separate frozen model must supply each legal full plan. The sampler preserves
that plan through sheet placement/release and learns only four panel-nailing
indices: 24 orders, with three nontrivial decisions. All three arms still execute
the entire physical task and unchanged strict assembly/flip checks. This is a
restricted conditional ordering policy, not unrestricted full-program learning.

Three CPU tests verify distinct frozen prefixes are retained, suffix sampling
remains legal and diverse, conditional log-probabilities agree with the sampler,
forced-prefix tokens receive no gradient, and altered prefixes are rejected.
No real Qwen or physical training run uses it yet. Any actual curriculum must
record the frozen prefix-policy checkpoint, generated plans, budget and selection
rule before sampling; manually written successful programs are not substitutes.
