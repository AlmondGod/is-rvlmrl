# Small code-as-policy RL experiment

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
