# Parallel rollouts and varied-placement RL

Status: parallel execution is verified; policy improvement is still under evaluation. The 24 held-out placements remain untouched. The constrained GRPO pilot was explicitly retired at its declared round-six unchanged-policy boundary, and the constant-prompt preference phase completed without improvement. The camera-conditioned preference phase also completed without improvement. A balanced camera-conditioned on-policy GRPO continuation is prepared. No new held-out improvement claim yet.

The completed pilot in `run03-progress/` contains six rollout rounds, 48 sampled
programs and 24 optimizer updates. Greedy validation at rounds 2, 4 and 6 remains
4/8 with the identical original program. Sampled training successes total
27/48 across changing/repeated layouts; that is not a paired improvement estimate.
The loop and final validation/checkpoint interval used 7400.27 seconds
(2.056 allocated GPU-hours), including environment waits.
Generation totaled 65.05 seconds and optimization
75.45 seconds. All six checkpoints
and their certified physical evidence have hash-verified local backups.

The sample-efficiency contingency is documented in `preference-replay-protocol.md`.
It constructs 23 strict success-versus-failure preferences within the same
training layout, from the pilot's own certified programs. The next phase is
**GRPO followed by offline DPO**, not additional on-policy GRPO rollouts. It
uses the completed GRPO weights as its warm start, the original prior as its
frozen reference/baseline, and the unchanged grammar, physics, reward and splits.
Forty-eight preference optimizer updates were completed, with full eight-layout
validation every eight updates and earliest-maximum selection.

`preference-failed04/` preserves an infrastructure-failed attempt: eight updates
in 22.52 seconds, followed by a missing cwd-relative robot-asset error before
fresh simulation could start. Those eight validation launches received no
reward and their incomplete files were quarantined intact. `preference-progress05/`
records the restart after an asset-path smoke check. The dataset and sampled
pair sequence are identical, but GPU gradients/weights were not bit-reproducible;
the recovery receipt discloses the difference. Validation at updates
8/16/24/32/40/48 is 4/8, 4/8, 2/8, 2/8, 2/8, 2/8. The earliest maximum
remains the original baseline (step 0), so final testing was refused. The loop
took 3,716.40 seconds (1.032 allocated GPU-hours), including validation.
`preference05-final/` contains the final selection, updates and passing record audit. The permanent source-
relative asset-path fix is separately tested to retain identical scene XML.
Active experiments retain their frozen simulator source and resolve the original
assets through the recorded overlay link.

The preference GPU preflight verifies finite gradients, an untouched frozen
reference, exact initial/reference forward probabilities, and 9.44 GB peak
allocated memory for one pair. It performs no optimizer step or task evaluation.
All 34 preference/data/accounting tests pass on the GPU host. Archived data now
retains parent-pilot evidence and counts replayed examples separately from new
physical executions; repeated identical payloads can be stored as verified tar
hard links instead of duplicated model bytes.

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
Full successful and failed replays now match every recorded state and all result
fields except elapsed time: 6,482×155 states on training layout 51002, and
6,309×155 states on validation layout 52001. The latter retains its original
strict failure; acceleration does not turn failures into successes. The measured
full episode times are 1,143.3→1,029.3 seconds and 1,162.4→1,087.7 seconds,
respectively (1.11× and 1.07× throughput). These shared-host observations are
not isolated speed estimates. Production compiler flags avoid machine-specific
instructions and fast-math; its synthetic color-mask/coordinate microbenchmarks
measure 3.05×/2.80× median speedups. The exploratory `pixel_probe03.json` uses
machine-specific compiler flags and measures only synthetic pixel predicates;
it is not the production backend's full-episode performance result.

The native-pixel evidence archive is attached to the [speed release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-assembly-speed-2026-10-23): `cap-rl-native-pixels-2026-10-24.tar.gz`, 44,786,925 bytes, SHA-256 `d31d81564bd0098ae41c7f0fd112e8e5c479bbe10efa566217d709da2582c678`. All 493 payloads were verified locally against the manifest before upload; see `native-pixels-archive-verification.json`.

The [in-progress checkpoint release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-varied-assembly-2026-10-24) retains all seven GRPO snapshots, the failed preference attempt’s step 8, and all seven preference05 snapshots through step 48. Those 15 checkpoint archives passed local payload verification and their GitHub SHA-256 digests match. A separate final-record archive retains the completed selection, last validation and final adapter. These are incremental adapter/evidence snapshots, not optimizer/RNG resume states; this release makes no held-out improvement claim.

An observation-conditioning feasibility probe uses the unchanged initial RGB frames, camera calibration and public robot packet on two training layouts (51001, 51002). The shared vision estimator localizes all four members; exact replay from saved PNGs and calibration reproduces the estimates. A proposed prompt includes only rounded visual XY/yaw and observed near/far ordering. Scene IDs, truth poses and outcome metadata are excluded. This probe motivated the separately declared camera-conditioned phase; it is not itself a policy improvement claim. The full 18-file input/source archive `initial-observation09.tar.gz` is attached to the in-progress checkpoint release.

The optional `vectorized_native` backend combines native pixels with a checked C implementation of the existing connector arithmetic. It records compiler/source/binary identities, pins the reference implementation and rejects runtimes that fail a 512-case arithmetic compatibility check. The integrated GPU-host suite passes 18 tests, including 20,000 reference cases and irreversible failure/work-history checks. This is CPU force arithmetic, not native GPU physics.

Its isolated prototype preserves the complete successful 51002 and failed 52001 trajectories: the state NPZ files have identical SHA-256 hashes and all result fields except wall time match. Full times are 1,143.3→966.7 seconds and 1,162.4→1,016.2 seconds (1.18×/1.14×), with both pixel and connector acceleration enabled on a shared host. The wrapper-only microbenchmark is 1.30× and includes data-copy overhead. An earlier arithmetic candidate failed exact comparison and was not deployed. The integrated full replays also passed exact state/result comparison. Their instrumented times are 1,000.4 and 1,059.9 seconds; profiler overhead and overlapping host load prevent interpreting these as isolated speed estimates. Active training remains frozen on the earlier vectorized backend.

The complete native-connector evidence archive contains 587 verified payloads,
including successful and failed replays, source, binaries and exploratory
arithmetic failures. It is attached to the speed release as
`cap-rl-native-connectors-2026-10-24.tar.gz` (81,698,492 bytes, SHA-256
`642da1e5b154558589485868dea63344585b34839de50eac388ae1113c1f0d07`).
The runtime supplement retains the exact prototype binary and original
microbenchmark wrapper.

The completed camera-conditioned replay phase follows `visual-conditioning-protocol.md`.
Both baseline and trained models receive initial RGB-derived member locations
and near/far ordering; the model never receives layout seeds or true object
poses. Forty development observations were exactly reconstructed from saved
images. Real Qwen forward/backward tests on two scene contexts passed, with
9.77 GB peak GPU allocation, no reference gradients and no optimizer step.
All 44 preflight tests passed on the pod. The actual run uses the same 23
self-generated, certified within-layout preferences, now paired with the
corresponding initial observation. It is offline DPO, not new training rollouts.
Its learned 8/8 validation gate and untouched 24-layout final test remain in force.

The camera-conditioned replay phase finished all 48 updates in 250.24 seconds,
including cached validation, and still selected the initial 4/8 policy. It did
not run held-out evaluation. `visual-online-protocol.md` declares the next
on-policy continuation: all 32 training layouts, four samples per layout,
16-program GPU batches, eight physics workers and four GRPO updates per batch.
The real-GPU preflight generated 16 programs (seven unique) in 23.01 seconds
and passed an eight-program-microbatch gradient check at 12.31 GB peak memory.
Synthetic preflight advantages are explicitly excluded from task evidence.
