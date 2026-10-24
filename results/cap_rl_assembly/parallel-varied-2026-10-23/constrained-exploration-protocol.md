# Constrained exploration amendment, before held-out evaluation

The initial validation policy passes 4/8 new placements. In the 16-program GPU
smoke sample, seven programs contain every required call; six are the identical
ascending-order plan and the seventh violates prerequisite ordering. This is
training-side sampling evidence, not a test-set observation. The unrestricted
pilot is retained, including any completed optimization round and all evidence.

The next candidate experiment explicitly narrows the generated-program action
space. It samples three permutations: member placement/release pairs, corner
nails, and panel nails. Each permutation has four distinct indices; the total
space is 24³ = 13,824 complete programs. Required prerequisite calls and syntax
are forced during decoding. This supplies a program structure, not a successful
ordering or supervised target. No generated program is repaired after sampling.
The prompt and full physical evaluator are unchanged. All failures remain zero;
all successful plans still require the original strict physical gates.

Only the 12 index-choice positions contribute to the policy loss. Each choice
normalizes over the remaining indices in its own permutation, using the same
temperature for generation, rollout, current-policy and reference probabilities.
Forced tokens have probability one and no policy gradient; the last index of
each permutation is also deterministic. The initial-adapter KL reference remains
frozen. The decoder and exact source digest are recorded separately from the
unchanged physical provenance. Cached physics is reusable only for the exact
program/layout/provenance or a previously certified terminal-failure prefix.

Initial candidate budget: 16 rounds, two layouts per round, four programs per
layout, four clipped optimizer epochs, temperature 3, learning rate 0.00005,
KL coefficient 0.005, eight physics workers, one OpenCV thread each, validation
every two rounds, and checkpoint every round. This is at most 128 candidate
programs and 64 optimizer steps. The initial constrained greedy policy is
measured explicitly; compare its code/results with the unrestricted starting
policy to distinguish a decoding change from learned improvement. Validation
selects the earliest maximum strict success, with the initial policy eligible.
The run may stop at a scheduled perfect-validation boundary: no later checkpoint can strictly exceed 8/8 binary success, and the earliest maximum wins. The original budget, actual rounds/optimizer counts and stopping evidence are retained in a source-hashed receipt. Further changes may use training/validation evidence only and must be recorded.

The 24 held-out seeds and the three predeclared replay seeds remain unchanged
and unused until final checkpoint selection. Final comparisons identify the
constrained initial/selected policies and the unrestricted raw model explicitly.
No held-out result may be used to choose a new order or train another candidate.
