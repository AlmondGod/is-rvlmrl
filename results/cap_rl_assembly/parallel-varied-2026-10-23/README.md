# Parallel rollouts and varied-placement RL

Status: parallel execution is verified; policy improvement is still under evaluation. The 24 held-out placements remain untouched. The constrained GRPO pilot was explicitly retired at its declared round-six unchanged-policy boundary, and a training-only offline preference phase is active. No new held-out improvement claim yet.

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
Forty-eight preference optimizer updates are planned, with full eight-layout
validation every eight updates and earliest-maximum selection.

`preference-failed04/` preserves an infrastructure-failed attempt: eight updates
in 22.52 seconds, followed by a missing cwd-relative robot-asset error before
fresh simulation could start. Those eight validation launches received no
reward and their incomplete files were quarantined intact. `preference-progress05/`
records the restart after an asset-path smoke check. The dataset and sampled
pair sequence are identical, but GPU gradients/weights were not bit-reproducible;
the recovery receipt discloses the difference. Validation at updates 8 and 16 is 4/8;
update 16 changes the program and the per-layout outcomes, but gives no aggregate
gain. Checkpoint 24 is undergoing full physical validation. The permanent source-
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

The [in-progress checkpoint release](https://github.com/AlmondGod/is-rvlmrl/releases/tag/cap-rl-varied-assembly-2026-10-24) retains all seven GRPO snapshots, the failed preference attempt’s step 8, and preference05 snapshots through step 24. All 12 archives (2,552,507,073 bytes) passed local payload verification and their GitHub SHA-256 digests match. These are incremental adapter/evidence snapshots, not optimizer/RNG resume states; this release makes no held-out improvement claim.

An observation-conditioning feasibility probe uses the unchanged initial RGB frames, camera calibration and public robot packet on two training layouts (51001, 51002). The shared vision estimator localizes all four members; exact replay from saved PNGs and calibration reproduces the estimates. A proposed prompt includes only rounded visual XY/yaw and observed near/far ordering. Scene IDs, truth poses and outcome metadata are excluded. This is not integrated into the active constant-prompt experiment and is not a policy improvement claim. The full 18-file input/source archive `initial-observation09.tar.gz` is attached to the in-progress checkpoint release.
