# Qwen3.5-4B full three-arm assembly pilot

Status: the physical training integration is implemented and the pilot is in
progress. **Sampled training successes exist; no held-out improvement is
claimed yet.** The historical live visual-language benchmark remains unchanged.

The task starts with four loose frame members and a loose plywood sheet. Three
torso-mounted UR10e arms place the members, make four frame connections, feed and
place the sheet, make four sheet connections, then lift and flip the assembly.
The model writes a complete program using supplied RGB/encoder feedback skills;
it does not receive images or hidden workpiece state and does not learn motors.

[Task, reward, splits, training budget and reproduction](../../../experiments/cap_rl/ASSEMBLY_TASK.md).
Use [requirements-cap-assembly.txt](../../../requirements-cap-assembly.txt) for
this pilot; it retains the full simulator's NumPy 2.5.4 pin alongside the Qwen
dependencies, rather than the older single-rail runtime's NumPy 2.5.2 pin.

## Completed development checks

The scripted reference completed development seed 3 in **630.70 simulated
seconds / 1,003.02 wall seconds** on macOS. It passed the existing independent
full-task gates with **8/8 intact fasteners**, **12.74 seconds inverted hold**,
**0.430 mm peak penetration** (limit 0.5 mm), **0.831 mm peak airborne relative
displacement**, and **1.351 degrees peak relative angle**. This is a scripted
feasibility result, not model inference or an RL result. No physics, nail strength
or acceptance threshold was relaxed. Its exact 227 source/asset files were
reconstructed against the recorded hashes for preservation.

![Scripted full-task feasibility; not model-generated](development/scripted-feasibility.png)

The same scripted reference **fails training layout 41001** on the GPU host:
it finishes the motion but retains only **7/8 fasteners**. The first corner
connection is missed. A post-hoc ray test at recorded states near that trigger
finds a **6.280 mm material gap along the nail ray**, exceeding the unchanged
6 mm insertion limit. This diagnostic does not feed state or answers to the
model. The failed full run is preserved in `development/train41001`; it took
626.90 simulated seconds and 1,970.40 wall seconds. The single successful
development layout therefore does not establish controller reliability across
randomized layouts, and valid program sequencing alone may not solve every case.

Qwen3.5-4B's initial greedy program picks all four members before releasing any.
The second call tries to pick up another member with an occupied gripper. The
physical executor rejects it with `Arm already carries a member`. The GPU smoke
check produced finite nonzero LoRA gradients and used **14.06 GiB peak allocated
memory** on the new 24 GB RTX PRO 4000 Blackwell.

The new language/reward tests and inherited benchmark, geometry, perception,
fastener and flip tests pass: **54 tests** on the GPU host. A separate parallel
regression compares physical grasp trajectories and failure results against
sequential execution at absolute tolerance **1e-7**, including AST-equivalent
batch deduplication.

The first sequential preparation run was stopped with **zero optimizer updates**
to improve simulation throughput. Its partial baseline records are retained.
The replacement uses four independent physics workers and 32 GRPO updates,
two layouts × four samples per update (256 programs). Eight training layouts,
four validation layouts and eight untouched test layouts are predeclared.
Test evaluation must wait until validation-based selection is final.

Model weights: `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, raw initialization, non-thinking mode,
rank-16 all-linear LoRA. This does not continue the single-rail adapter.

The cached model files on the mounted volume were unreadable; the same pinned
official weights were downloaded to local pod storage. The failed setup log is
retained separately and no optimizer update occurred during that failure.

## First learning update (interim evidence)

The first optimizer update completed with a finite nonzero gradient (norm
0.15794); mean sampled reward was 0.15469 and strict success was 0/8.
The correctly interleaved full program on layout 41001 earned 0.61875 but retained
7/8 fasteners. An independent execution of its identical call sequence matches
all 6,270 recorded states exactly (maximum difference zero). The comparison
records the wrapper/helper source differences; physics, controllers, assets and
runtime match. This is not presented as an identical-full-provenance audit.

On layout 41002 the same sampled sequence placed all four rails and inserted all
four frame fasteners, then failed the unchanged numerical gate during plywood
positioning: peak penetration 0.755 mm versus a 0.5 mm limit. Reward is zero.
Its program and compact result are preserved in `initial-training`. The terminal
contact diagnostic is a post-hoc measurement; the peak occurred within an
integration substep and cannot be attributed to a particular contact from the
saved terminal state alone.

To reduce waiting, an auxiliary process physically evaluates this already
model-sampled program on additional **training layouts only**. It publishes
complete hash-certified records only when their exact cache key is absent;
existing or running records are never replaced. The learner receives a cached
reward only if it independently samples the same parsed program for that layout.
No program is injected into an optimizer batch or used as a demonstration target.
The exact auxiliary driver is retained in `initial-training`; its extra physical
executions are to be counted separately from the 256 on-policy training samples.
The first helper setup was interrupted and retained after a result-aggregation
bug was caught; the corrected helper uses a separate evidence directory.

A tighter-placement controller candidate was tested separately on training
layout 41001, reducing its XY stop tolerance from 3 mm to 1.5 mm. Both paired
local scripted runs finish with only 7/8 fasteners; the first corner still misses.
The candidate is rejected and is **not part of the active frozen RL run**.
The full comparison is in `development/placement-precision-trial`. Local and GPU
trajectories differ, so this comparison uses the same local runtime for both
variants and does not attribute cross-platform differences to the controller.
No held-out test has been run.

After training-only prefetch freed workers, a separate auxiliary phase began
simulating the same already sampled program on the four validation layouts.
A scheduled checkpoint can consume these results only if its independently
generated greedy program matches exactly. Validation rewards never enter
optimizer batches; no test layouts are prefetched. The driver prints no success
or reward values, and these candidate validation results are not consulted for
controller development before the scheduled checkpoint evaluation. Its exact
driver is preserved alongside the training-only driver. Unused candidate
executions remain part of the compute record.

`analysis/cap_rl_execution_inventory.py` inventories explicit evidence roots and
verifies available file certificates. It collapses copied cache records using
the exact result bytes (which include measured wall duration), while retaining
independent deterministic replays as separate executions. Summed parallel
execution durations are explicitly not reported as allocated GPU hours.

## Support-measurement diagnosis (training layout only)

A read-only instrumented replay on training layout 41001 matches all 5,350 saved
states of the uninstrumented reference prefix **exactly**. During simulated
seconds 525–530, the panel weight is 67.06193 N; mean contact support is
54.95869 N and the original finite nail connections supply another **12.10359 N**
upward. Their sum balances weight to 0.00035 N, consistent with the near-zero
vertical acceleration. The contact-only stability predicate omits that load
path. This diagnosis does not excuse the independently missing frame fastener;
this particular trajectory still has only 7/8 connections.

An opt-in `support_measurement.py` observer captures the existing internal nail
force before the simulator clears its scratch buffer, and a separate evaluator
includes its signed vertical component in the same 5% support-balance check.
The original benchmark/default evaluator and active RL run remain unchanged.
Eight focused local tests pass, including downward preload, missing support,
external support, bad geometry, and preservation of the full-task gates. The
separate GPU replay of training layout 41004 has now completed: all 6,419 saved
physical states match exactly (maximum difference zero). The original score is
0.70/fail; the corrected score is 1.00/pass with 13.86 seconds pre-flip stability
and a 12.80-second inverted hold. Contact support is 51.75864 N and internal nail
support is 15.30331 N, balancing the 67.06193 N panel weight. This classification
change is a measurement correction, not an RL gain. No test layout has been used.

The first diagnostic sampled after buffer clearing and therefore recorded zero
rather than the actual internal force; it is retained with that limitation. The
corrected measurement occurs immediately after the original force calculation.

## Nail-approach development (separate from RL)

On training layout 41001, the first-corner-only approach adjustment passes the
**original** full acceptance checks: 8/8 intact nails, 12.82 s inverted hold,
88.36 s pre-flip stability, 0.230 mm peak penetration, 1.194 mm airborne relative
displacement and 2.364 degrees relative angle. It took 626.90 simulated seconds
and 837.72 wall seconds locally. This is a scripted controller result, not model
inference or an RL gain, and it does not use the support-measurement correction.

The development history is preserved in `development/nail-approach-trials`:
changing an unused legacy helper had no effect; the active all-corner 0.90
horizontal direction formed the first three connections but timed out at corner
3; the smaller all-corner 0.85 direction exceeded the unchanged 0.5 mm penetration
limit. The candidate uses 0.90 only at corner 0, retaining the original nominal
0.80 direction at other corners. Exact source patches and raw-evidence hashes
are retained. Paired local runs on training layouts 41006 and 41008 also completed. Layout
41006 passes with both controllers; 41008 changes from 7/8 fasteners/fail to
8/8/pass. Across these three training-only pairs, original scripted success is
1/3 and candidate scripted success is 3/3. The candidate is not part of the
original frozen RL pilot. These development cases do not estimate held-out
performance or a learned model improvement.

## Restart decision and integration status

The original contact-only pilot was retired after five completed updates
(2.574 training-loop hours; 2.639 process hours including model setup/baseline).
Mean sampled reward progressed from 0.15469 to 0.37266. Updates 4 and 5 each
contain one successful sample out of eight on training layout 41002: the model
placed members in order 0,3,1,2 and completed all eight fasteners and a 12.74-second
inverted hold under the original checks. Update 5 reused the same measured
program/layout evidence. This is a sampled training success, not a held-out
accuracy or a controlled before/after improvement.

Retirement was requested after batch 4, but the cache made batch 5 finish before
the watcher was armed; all five complete batches are retained. The pilot's
learned weights were not checkpointed before the scheduled update-8 validation.
The next experiment restarts from raw Qwen3.5-4B, with no weight continuity.

The corrected experiment uses explicit fixed evaluator/controller profiles for
both baseline and trained evaluation, retains the original acceptance result
as a shadow measurement, and saves each optimizer update. The new integration
is under verification, not yet a completed training result. Initial local
verification passed 74 runnable tests; the GRPO test module requires PyTorch and
will run on the GPU host (the local physics-only environment lacks PyTorch).

### Corrected experiment launched

The fixed profiles passed 78 GPU-host tests. A full local integration replay
matches all 6,270 recorded states of the isolated controller candidate exactly
and preserves every original acceptance field. The three training-layout GPU
verification episodes (41001, 41004, 41008) all pass both corrected and original
shadow acceptance. The independent-process cache contention test passes and
matches sequential physical execution within 1e-7. Evidence and exact launchers
are under `corrected-preflight`.

Fresh raw-model training (`assembly03`, frozen implementation commit `b833a18`)
started on 2026-10-23 at 07:57 UTC with the declared 32-update configuration.
Baseline greedy validation is 0/4, reward zero. Corrected-profile evidence is
separate from the retired pilot. An auxiliary job evaluates the already sampled
canonical program on the remaining five training layouts; rewards can be reused
only when the learner independently samples exactly that program/layout under
the same frozen profile. No test layouts are included.

Verification tools now cover saved-record integrity and score/selection
accounting, independent regeneration from model weights, and fresh physical
replays of every distinct held-out episode. These tools are prepared; final
held-out improvement has **not yet been measured**.

### Optimizer reward fallback prepared (not yet active)

The corrected dense-reward run completes its first three updates with sampled
strict success rates 2/8, 3/8 and 3/8. Updates 4–6 then sample only incomplete
two-member programs (reward 0.15, strict success zero, no within-group reward
variance). In update 3, the two-member programs outrank other failures scoring
0.075 in their group, so group-relative optimization can reinforce them despite
full successes in the other layout's group. This is training evidence, not a
held-out finding or a claim that later validation must fail.

`analysis/train_assembly_strict.py` prepares a success-only optimizer alternative:
its learning reward is 1 only for measured strict physical success, otherwise 0.
It keeps the original dense reward in `dense_reward` for auditing. The frozen
physical environment, its certificate identities and every acceptance threshold
remain unchanged. The optimizer adapter's exact source and hash are archived
in any run that uses it. This is an explicit training-reward transformation,
not a rewrite of cached physical results. Nine focused accounting/adapter tests
pass, including preserved physics results, source integrity and high-progress
failures receiving zero learning reward. The candidate is not yet running;
held-out layouts remain unused while the dense-reward experiment continues.

## Dense-reward retirement and strict-success restart

The corrected-environment dense-reward pilot completed eight updates, then
met a recorded futility rule: updates 5–8 had no successes and no within-group
reward variance; greedy validation at update 8 scored 0/4. The optimizer had
reinforced incomplete two-member programs as the best failures within a group.
Its 64 samples, validation, retirement decision and retained checkpoint hashes
are preserved in `retired-dense-pilot`. Training-loop time was 1.265 GPU-hours;
process time including setup and final validation was 1.433 GPU-hours.

The next candidate restarts raw Qwen3.5-4B with a 1/0 strict-success optimization
reward, learning rate 3e-5, 16 updates and validation every two updates. The
physical dense reward remains a diagnostic. Exact external optimizer-wrapper
source is archived and hashed in run metadata. Frozen simulator/controller
source b833a18, prompts, splits and all success thresholds are unchanged.
No demonstrations or repaired programs enter training. Matching certified
physical records can be reused because their physical provenance is unchanged.

`strict-success-run` preserves the launcher, backup driver and gated finisher.
The finisher evaluates the untouched eight test layouts only after the full
budget and validation improvement, then audits records, regenerates every test
completion, and freshly replays every distinct test trajectory. Results remain
pending; eight test layouts constitute a small simulation pilot.

The strict-success restart's greedy validation improves from 0/4 to **4/4 at
update 2**, repeated at update 4. The retained original contact-only evaluator
classifies those same trained validation trajectories as 3/4. Both measurements
are preserved; no held-out score is claimed. The selected adapter matches the
atomic update-2 checkpoint hash and has been backed up locally. A validation-ceiling stopping rule was then declared before any test use: the
maximum tuple (1, 1) and earliest-best tie rule prove later updates cannot
replace checkpoint 2. The run will stop after the in-progress update 6 and its
scheduled validation. The original 16-update budget remains in metadata, with
a separate hashed stopping receipt. A pre-test audit must pass before the
untouched held-out evaluation begins. Exact source for this finalizer and the
updated verification driver is preserved in `strict-success-run`.
