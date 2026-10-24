# Integrated fused backend: complete success and failure replay

Both full episodes exactly match their independently recorded baseline: all
6,482×155 states in successful training layout 51002, and all 6,309×155 states
in failed validation layout 52001, plus every non-timing result. Success and
failure remain unchanged. The integrated backend had already passed 16 Linux
regression tests covering force arithmetic, failure history and joint overrides.

Full physical-loop times are 731.87 and 779.86 seconds, compared with the earlier
instrumented native-connector baselines of 1,000.40 and 1,059.86 seconds. One
bounded extra simulator overlapped training; these ratios are not isolated
throughput or whole-RL speed measurements. The supervisor allowed at most one
extra simulator, admitted below 21 GB anonymous/kernel/shared memory, and would
stop its child at 25 GB in the 31 GB cgroup. Peak measured working memory was
21.62 GB; both children exited successfully without a memory stop.

The archive retains the two fresh runs, their baseline recordings, complete
source/asset manifest, exact native binaries, regression logs and admission
receipt. Its 278 payloads have been verified locally. The active learner's
previous physical backend remains frozen; subsequent experiments can now use
this separately validated implementation with a new provenance/cache identity.
