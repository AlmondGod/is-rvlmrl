# Parallel rollouts and varied-placement RL

Status: parallel execution verified; GPU model smoke check and new RL run pending. No new policy-performance claim yet.

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
