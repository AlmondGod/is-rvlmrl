# Parallel rollouts and varied-placement RL

Status: parallel execution and GPU optimization preflights verified. The prior policy passes 4/8 fresh validation placements. Constrained-order GRPO is now training; the 24 held-out placements remain untouched. No new held-out policy-performance claim yet.

Interim measured progress at 2026-10-24 03:32 UTC is preserved in
`run03-progress/`: two rollout rounds, 16 sampled programs and eight optimizer
updates. Sampled training successes were 1/8 and 6/8. Greedy validation remains
4/8 because the generated ordering is unchanged; this is not yet a learned
improvement. The training loop used 2,395.86 seconds (0.666 allocated GPU-hours),
including CPU environment waits. Generation totaled 21.60 seconds and the eight
optimizer updates totaled 25.32 seconds. Environment reward wait after overlap
totaled 2,341.98 seconds; this includes rendering, vision and scripted control,
not only the MuJoCo kernel. The first two checkpoints and all their certified
evidence have hash-verified local backups. Training continues.

Eight fresh training layouts (51001–51008) each execute the first 20 simulated seconds of the same complete program. Every run starts from reset, uses the vectorized backend, original physics/controller/camera parameters, and one OpenCV thread per worker. There is no rollout-cache reuse. Process startup is included.

| Workers | Time for eight layouts | Throughput vs serial |
|---|---:|---:|
| 1 | 229.723 s | 1× |
| 4 | 53.659 s | 4.281× |
| 8 | 28.830 s | 7.968× |

All 16 parallel-versus-serial paired comparisons have exactly equal recorded arrays and result fields other than wall time. These are timed prefixes that terminate at the simulation deadline, not full-episode success or end-to-end RL measurements. The benchmark source is commit `420d576`.

The hybrid training path uses GPU-batched Qwen inference and optimization, GPU camera rendering, and eight independent CPU MuJoCo workers. GPU old/reference log-probability calculation overlaps physical rollouts without changing the policy until every reward arrives. Four clipped GRPO optimizer epochs reuse each expensive physical batch; frozen old probabilities and a frozen initial-adapter KL reference prevent this from being described as four independent rollout batches. Per-round timing, optimizer-step counts, cache hits and physical evidence identities are recorded.

The separately preserved `warp-probe.json` is an exploratory native-GPU physics-kernel probe on the initial three-arm scene with fixture half-width 0.40 m and default shoulder half-span 0.34 m. It does not use the task's complete geometry configuration, RGB feedback or active nail forces, and is not a validated training backend or an end-to-end performance claim.

The fresh data splits, initial plank perturbations, strict reward, planned training configuration, target and final verification are declared in `experiments/cap_rl/VARIED_ASSEMBLY.md`. The 24 held-out placements have not been evaluated.

The corrected `warp-probe-protocol.json` uses the actual task fixture half-width 0.38 m and shoulder half-span 0.55 m. Its 32-world initial-physics throughput is 51,591 world-steps/s. It still omits the complete controller, active custom fasteners and acceptance checks, so it remains a feasibility probe.

The 16-program Qwen GPU smoke check completed generation in 77.68 s including first-use kernel warmup, with peak allocated CUDA memory 19.42 GB. Backpropagation produced a finite gradient; the frozen reference received no gradients, initial policy/reference probabilities matched, and saved default-adapter weights matched the original checkpoint exactly. No optimizer update was performed in this smoke test. The training environment subsequently passed all 31 preflight tests.

The optimizer-only necessary-call filter is specified in `static-rejection-protocol.md`. Its historical regression rejects 15 measured failures among 49 unique program/layout rows and rejects none of six successful rows. It does not replace physical evaluation for eligible programs.

The unrestricted pilot completed one rollout round: 16 sampled programs, four
optimizer epochs, four distinct physical executions, six successful samples,
and three active groups in 1,318.26 s. Optimization itself took 27.62 s. It was
retired because its successful samples repeated the same ordering; its weights
are preserved but do not initialize the next run. Detailed generation/physics
phase timing was omitted by a wrapper bug in this pilot; that reporting bug is
fixed for the next run. See `unrestricted-pilot-retired.json`.

The new action-space restriction and selection/stopping rules are declared in
`constrained-exploration-protocol.md`. It supplies the necessary program
structure while learning member/corner-nail/panel-nail permutations. Its initial
greedy code exactly matches the unrestricted prior, so the starting 4/8 result
is unchanged. Eight sampled plans contain seven distinct legal orderings.

GPU preflight with real Qwen weights, frozen initial reference and backward
passes: without activation checkpointing, microbatch 2 peaks at 20.18 GB and
microbatch 4 fails with GPU OOM. Non-reentrant checkpointing in training mode,
with dropout explicitly disabled, reduces microbatch-2 peak allocation to
9.44 GB and supports all eight programs together at 11.36 GB. Choice-only
vocabulary projection avoids computing optimization logits for forced syntax.
Eight-program generation takes 10.77 s in this preflight. First-use backward
measurements include kernel compilation and are not steady-state throughput.
These checks deliberately stub physical rewards and perform no optimizer step;
they verify implementation and capacity, not task improvement. Both preflight
JSON files retain the raw measurements and sampled programs.

A complete successful training-layout episode (51002, 648.1 simulated seconds)
was profiled separately. Its 6,482×155 recorded states and all result fields
except wall time exactly match its original certified execution. Of the 1,117.7
seconds of instrumented wall time, frame vision uses 30.7%, native MuJoCo stepping
25.3%, custom nail forces 23.0%, camera rendering 3.5% and panel vision 3.2%.
These are disjoint regions; the remaining 14.3% includes control, checks and
overhead. The raw profile also contains inclusive parent timers, which must not
be added to their children. This is one episode on a shared host, not a
population average or isolated speed comparison.

An additional **opt-in** `vectorized_pixels` backend compiles the integer RGB
predicates and restricts pixel-coordinate scans to each mask's nonzero bounds.
It retains full-resolution images, the original masks and row-major pixel
coordinates. Its compiler/source/binary identity is recorded in physical
provenance, and it requires a working C compiler. The current RL run and its
verification checkout remain frozen on the earlier `vectorized` backend.
The new backend's complete Linux test suite passes 50 tests, including exhaustive
checks of all 16,777,216 RGB values for each of four predicate combinations.
Full successful and failed episode equivalence runs are in progress; no complete
episode speedup is claimed yet. The exploratory `pixel_probe03.json` uses
machine-specific compiler flags and measures only synthetic pixel predicates;
it is not the production backend's full-episode performance result.
