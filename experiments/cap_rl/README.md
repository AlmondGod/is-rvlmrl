# Small code-as-policy RL experiment

Follow-ups: [ordered two-object task](ORDERED_TASK.md) and
[MuJoCo inertial docking](MUJOCO_TASK.md). The MuJoCo run improves held-out
success from 1/128 to 107/128; see its [results and physical episode replay](../../results/cap_rl_mujoco/2026-10-22/README.md).

This experiment adapts the core **CaP-RL** procedure from
[CaP-X](https://github.com/capgym/cap-x): sample executable programs on-policy,
obtain environment rewards, and update the language model with GRPO. It is an
independent single-GPU implementation using Transformers and PEFT, **not a
reproduction of the paper's Franka results or its VeRL training stack**.
Upstream reference inspected: `53e9966d7a8e2fa7494676772bccc35280f5c0ed`,
particularly `scripts/train_franka_grpo.sh` and `docs/rl-training.md`.

## Task and scope

A discrete 5 × 5 board contains a gripper, one wooden member, and a target. The
language model sees their integer coordinates and generates up to eight Python
calls: `move(dx, dy)`, `pick()`, and `place()`. Movements are relative. A successful
program must pick up the member, transport it to the target, and release it.

This is a deliberately small **symbolic code-generation task**, not MuJoCo,
vision, continuous control, collision-aware planning, or the frozen full-frame
assembly benchmark. Its coordinate observations and kinematic attachment are
explicit simplifications. Performance here does not demonstrate competence or
transfer to the existing assembly workcell. Existing benchmark source, thresholds,
and historical measurements remain independent.

Generated source is parsed with Python's AST and interpreted by an allowlisted
grammar; it never runs through `eval` or `exec`. Imports, assignments, loops,
attributes, computed arguments, and arbitrary Python are rejected. A single
optional Markdown code fence is accepted identically for both models.

## Protocol

- The fixed seed `20261022` permutes all ordered triples of distinct board cells.
  The first 256 layouts are training, the next 64 validation, and the next 128
  test. Layouts are disjoint; this is in-distribution generalization to new
  coordinate combinations, not an unseen task family.
- Strict success is binary. The training reward is 0.15 for approach progress,
  0.20 for pickup, 0.15 for transport progress, and 0.50 for strict success.
  Invalid or truncated programs receive zero. No reward comes from explanations,
  textual claims, or merely using valid syntax.
- Approach progress is the best reduction in Manhattan distance before pickup;
  transport progress is the final reduction in member-to-target distance.
  Both are measured by the environment, bounded to [0, 1], and separate from
  strict success. A partial-reward gain alone does not satisfy the experiment.
- Training uses fresh sampled programs, temperature 1, no top-k/p filtering,
  within-prompt normalized advantages, an immutable base-policy KL penalty,
  completion-only loss, and rank-16 LoRA. One optimizer update is taken per
  rollout batch; the PPO clip is present but is inactive at the initial ratio 1.
  No supervised warmup, teacher, or oracle-generated training answers are used.
- Checkpoints are selected by validation success, then validation reward. The
  base model and selected adapter are compared once on test with the same prompt,
  greedy decoding, token budget, executor, and layouts. The report includes every
  generated program, paired wins/losses, and a paired bootstrap interval.
- The bootstrap describes variability across these layouts for a single trained
  model. It does not measure variability across independent training seeds.

`task.oracle` is a test-only feasibility check; it is not imported by the trainer
and its answers are not placed in prompts or training data.

## Run

From the repository root, with Python 3.12 and a CUDA-enabled PyTorch installation:

```bash
python -m venv --system-site-packages .venv-cap
source .venv-cap/bin/activate
pip install -r experiments/cap_rl/requirements.txt
python -m unittest experiments.cap_rl.test_task experiments.cap_rl.test_grpo
python -m experiments.cap_rl.train train --out outputs/cap_rl/new_run
python -m experiments.cap_rl.train test --out outputs/cap_rl/new_run
```

Each training invocation requires a new output directory. Test evaluation refuses
to overwrite existing results. Use the exact model revision and arguments in a
published run's `metadata.json` for reproduction. The initial hardware environment
is PyTorch 2.8.0+cu128 on one NVIDIA RTX PRO 4000 Blackwell (24 GB).

Training writes metadata and source hashes, full rollout records, per-step metrics,
validation outputs, selected and final LoRA adapters, and checkpoint selection.
Testing writes `test_comparison.json`. Large model weights belong in a release
artifact or external storage, not the Git source tree.

Replay every test program and verify the aggregate result without a GPU:

```bash
python -m experiments.cap_rl.audit outputs/cap_rl/new_run
```

Generate a program for a chosen layout using a saved run and compare with its
base model (this is a demo, not an additional benchmark):

```bash
python -m experiments.cap_rl.demo --run outputs/cap_rl/new_run \
  --gripper -2 0 --member 1 1 --target 0 -1
```

The demo downloads the pinned base model and loads `best_adapter/` from the run.
The model remains specialized to this tiny API and coordinate range.

## Development record

`run01` used a syntax-only 0.05 reward and was stopped after its first validation
check (step 20): 0/64 successes. The model learned short, valid movements rather
than completing pickup and placement. That reward was removed before the next
run. No final test layouts were evaluated during this pilot.

`run02` restarted the 0.5B base with the revised reward but found no nonzero rewards
in its first 19 updates and was stopped. Both early pilots inherited Qwen's
`repetition_penalty=1.05`, which does not match the raw policy probabilities in the
loss. They are failed development attempts, not valid GRPO result claims.

`run03` uses Qwen2.5-Coder-1.5B-Instruct and an explicit neutral generation
configuration (repetition penalty 1.0, no sampling filters). Its first validation
checkpoint at update 20 improved strict success from 0/64 to 12/64. This is a
development observation, not the final test result.

The completed 120-update run reached **60/64 validation successes**. The selected
checkpoint then improved untouched-test success from **4/128 to 121/128**.
See the [full result, exact reproduction command, and checkpoint download](../../results/cap_rl/2026-10-22/README.md).


## Harder follow-up: two objects

[Ordered two-object delivery](ORDERED_TASK.md) doubles the typical program from
four calls to eight and requires carrying the gripper's position across two
deliveries. It continues RL from the selected one-object adapter on new, disjoint
full layouts. The baseline for this follow-up is the one-object adapter on the
new task, using exactly the same prompts as the final trained checkpoint.

The follow-up has its own task module, reward, split seed, raw development runs,
and replay viewer. See `analysis/cap_rl_replay.py` for the viewer and
`analysis/cap_rl_audit_records.py` to replay saved training and validation
programs against each run's frozen task source.

The completed follow-up improves held-out success from **3/128 to 103/128**,
with 100 paired wins and no losses. [Results and checkpoint](../../results/cap_rl_ordered/2026-10-22/README.md).

## Articulated robot follow-up

The [xArm7 frame-rail lift task](ROBOT_TASK.md) adds a real articulated arm and
contact-driven grasp. Raw-base held-out physical success improves from
**55/128 (43.0%) to 122/128 (95.3%)** after 80 GRPO updates.
[Results, video, exact code, checkpoint and audits](../../results/cap_rl_robot/2026-10-22/README.md).
This isolated pickup task uses privileged coordinates and supplied motor skills;
it is separate from the full visual assembly benchmark.


## Qwen3.5-4B rail placement

The [frame-rail placement follow-up](../../results/cap_rl_frame/2026-10-22/README.md)
uses Qwen3.5-4B in non-thinking mode and adds transfer, lowering, release and
withdrawal. After 80 updates and 1.12 A40 training GPU-hours, held-out success
improves from 21/128 to 128/128, including a separate pickup-first check.
The original reward admits an ordering loophole; the report preserves that
training evidence and documents the stricter evaluation. For future training,
use `--task frame_place_ordered`; `frame_place` remains the historical mode.
[Task, original and corrected reward](FRAME_TASK.md).


## Full three-arm assembly and faster execution

[Varied-placement experiment and protocol](../../results/cap_rl_assembly/parallel-varied-2026-10-23/README.md)
records the full frame task, strict reward, failed attempts and checkpoint
selection. The current online run uses its frozen `vectorized` backend.

New runs can opt into `--assembly-backend vectorized_fused` in
`experiments.cap_rl.train` or `experiments.cap_rl.assembly`. This Linux backend
fuses the internal MuJoCo step loop and uses native pixel and nail-force
operations. Physics still runs on CPUs; batched policy generation and learning
run on the GPU. It does not change the timestep, controller, nail law or reward.
It requires a C compiler and the installed MuJoCo headers/shared library.
Source guards, numerical compatibility checks and binary hashes bind the
backend to each physical certificate; use a fresh process to switch backends.

The integrated implementation passed 16 Linux regression tests, including
4,000 exact internal steps for each of two force profiles, joint overrides,
nail failure events and infrastructure error handling. Its prototype also
matched all 6,482 saved states of a full successful episode and took 780.5
seconds versus 1,143.3 seconds for the vectorized reference on the shared host
([prototype evidence](../../results/cap_rl_assembly/parallel-varied-2026-10-23/native-substep18/README.md)).
Full successful and failed episode replays of the integrated backend remain
pending. These measurements do not establish whole-training speedup or a
held-out policy improvement.

Keep one simulation pool per learner and budget host RAM as well as GPU memory.
The separate `analysis.assembly_async_rewards` queue supports independent
reward groups and checks available cgroup RAM before allocating workers.
A learner must still define its policy versions and update ordering; the queue
alone is not an asynchronous RL algorithm.
