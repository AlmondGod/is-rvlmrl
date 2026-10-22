# Harder task: ordered delivery of two objects

This follow-up keeps the 1.5B model and symbolic 5×5 board, but asks for two
colored objects to be delivered to distinct matching targets in a specified
order. The order varies across episodes. A typical correct program takes eight
calls instead of four. The second approach must start from the first target,
not the episode's original gripper position. Up to 12 calls are allowed.

`move(dx, dy)`, `pick()`, and `place()` have the original relative-motion semantics.
Picking the wrong object first, re-picking a delivered object, dropping onto
another object, leaving the board, or using an unsupported expression is invalid.
Objects can be carried over each other: there are no path collisions or physics.
All five initial cells (gripper, objects, targets) are distinct.

Strict success requires BOTH objects correctly released in the requested order,
with an empty gripper. Partial completion does not count as success.
For each object, reward is:

- 0.075 × best approach progress before its first pickup;
- 0.10 × successful pickup indicator;
- 0.075 × final transport progress toward its target;
- 0.20 × correct release indicator.

Completing both adds 0.10, for a total maximum of 1.0. Delivering just the first
object earns 0.45. Approach and transport use Manhattan distance. The second
approach's reference origin is the first target. A runtime error stops execution and always counts as failure, but preserves credit
for the correctly executed prefix. It never receives the 0.10 success bonus. A syntax error has no executed prefix
and receives zero. Truncated programs receive zero. This denser feedback is
needed because longer programs rarely finish without an error during exploration. This reward is separate from the
unchanged one-object experiment.

Seed 20261024 generates 704 unique layouts: 512 train, 64 validation, 128 test.
Both requested orders occur in every split. These are unseen configurations
within the same task, not an out-of-distribution or general coding benchmark.

The follow-up initializes from the selected one-object run03 adapter and uses
additional online GRPO, without demonstrations or oracle answers. The baseline
is that same initial adapter evaluated on this NEW task, not the untrained base
model. The KL reference remains the immutable pretrained base model. The initial
adapter is copied into the run with hashes; final testing compares that exact
copy with the validation-selected checkpoint, using identical greedy decoding.

Example invocation (the main run uses 160 updates):

```bash
python -m experiments.cap_rl.train train --task ordered_two \
  --out /opt/cap-rl/ordered04 --init-adapter /opt/cap-rl/run03/best_adapter \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --steps 160 --max-tokens 192 --eval-every 20 --lr 0.0001
python -m experiments.cap_rl.train test --out /opt/cap-rl/ordered04
python -m experiments.cap_rl.audit /opt/cap-rl/ordered04
```

The test set is evaluated only after validation selection. Source snapshots,
all sampled programs, evaluation traces, and both adapters are saved for audit.

## Development pilot

The initial `ordered01` pilot used a zero reward for any invalid program. No
sampled program earned positive reward in its completed updates, so it was
stopped before any test evaluation. Its raw outputs and original source are
retained. `ordered02` restarted from the original one-object adapter with valid-prefix
credit. It reached only 1/64 validation success at update 40 and showed a strong
bias to pick the first listed color (red), irrespective of requested order.
It was stopped before test evaluation.

`ordered03` restarted from the same original adapter and listed observations as
Delivery 1 and Delivery 2 in the requested order. It improved first-delivery
accuracy but learned to stop after four calls: 61/64 validation traces had four
calls at update 20, with zero complete successes. The 10% runtime-error discount
made stopping early preferable to attempting a second delivery and failing.

`ordered04` uses the same clear prompt but removes that discount, preserving
credit for successful actions even if a later action fails. The unchanged strict
success rule still requires a fully valid program completing both deliveries.
No movement offsets or solution programs are supplied. Its baseline and trained
checkpoint use identical prompts, rewards, and constraints.

## Try an episode

```bash
python -m experiments.cap_rl.demo_ordered --run /opt/cap-rl/ordered04 \
  --gripper 0 0 --red -2 0 --blue 2 1 --red-target 0 -2 --blue-target -1 2 \
  --first blue
```

Build a self-contained replay fragment from all recorded test episodes:

```bash
python -m analysis.cap_rl_replay /opt/cap-rl/ordered04 --out ordered-replay.html
```
