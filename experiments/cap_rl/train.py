"""Single-GPU CaP-RL-style GRPO: online programs -> execution -> policy update.

This is an independent, small adaptation, not the upstream VeRL implementation.
No demonstrations, teacher, answer injection, rejection-SFT, or test-set updates.
Sampling has no top-k/p truncation; log-probabilities use the same temperature
as generation. Dropout is disabled for rollout and optimization.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
import transformers
import peft
from peft import LoraConfig, PeftModel, get_peft_model
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .tasks import get_task

task = get_task()


MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def append(path, rows):
    with Path(path).open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_model(name, revision=None, adapter=None, trainable=False, reference_adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    # Model repositories can supply sampling processors. Raw policy log-probabilities
    # are only correct when these are disabled (Qwen ships repetition_penalty=1.05).
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, repetition_penalty=1., top_k=0,
        top_p=1., temperature=1., do_sample=False)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0., target_modules="all-linear",
            task_type="CAUSAL_LM"))
    if reference_adapter is not None:
        model.load_adapter(reference_adapter, adapter_name='reference', is_trainable=False)
        model.set_adapter('default')
        model._cap_reference_adapter = 'reference'
    return model.cuda().eval(), tokenizer


@contextmanager
def reference_policy(model):
    if getattr(model, '_cap_reference_adapter', None):
        previous = model.active_adapter
        model.set_adapter(model._cap_reference_adapter)
        try:
            yield
        finally:
            model.set_adapter(previous)
    else:
        with model.disable_adapter():
            yield


def save_policy(model, path):
    kwargs = {'selected_adapters': ['default']} if getattr(model, '_cap_reference_adapter', None) else {}
    model.save_pretrained(path, **kwargs)


def encode(tokenizer, layouts):
    texts = [tokenizer.apply_chat_template([
        {"role": "system", "content": task.SYSTEM}, {"role": "user", "content": task.prompt(layout)}
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for layout in layouts]
    return tokenizer(texts, return_tensors="pt", padding=True).to("cuda")


def completion_mask(tokens, eos):
    end = tokens.eq(eos)
    return (end.cumsum(-1) - end.long()).eq(0)


@torch.no_grad()
def rollout(model, tokenizer, layouts, max_tokens, sample, prepare=None, temperature=1.):
    started = time.monotonic()
    inputs = encode(tokenizer, layouts)
    kwargs = {"temperature": temperature, "top_p": 1., "top_k": 0} if sample else {}
    sequences = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=sample,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id, **kwargs)
    prompt_length = inputs.input_ids.shape[1]
    completions = sequences[:, prompt_length:]
    mask = completion_mask(completions, tokenizer.eos_token_id)
    attention = torch.cat([inputs.attention_mask, mask.long()], dim=1)
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    complete = completions.eq(tokenizer.eos_token_id).any(-1).tolist()
    indices = [i for i, done in enumerate(complete) if done]
    scored_layouts = [layouts[i] for i in indices]
    scored_texts = [texts[i] for i in indices]
    torch.cuda.synchronize()
    generated_at = time.monotonic()
    if prepare is not None and hasattr(task, 'submit_batch'):
        pending = task.submit_batch(scored_layouts, scored_texts)
        prepare(sequences, attention, mask)
        prepared_at = time.monotonic()
        results = pending.result()
    else:
        results = (task.execute_batch(scored_layouts, scored_texts) if hasattr(task, 'execute_batch') else
                   [task.execute(layout, text) for layout, text in zip(scored_layouts, scored_texts)])
        if prepare is not None:
            prepare(sequences, attention, mask)
        prepared_at = time.monotonic()
    rollout.last_timing = dict(generation_s=generated_at-started,
                              generation_and_rewards_s=time.monotonic()-started,
                              reward_wait_after_preparation_s=time.monotonic()-prepared_at,
                              overlapped_logprobs=prepare is not None and hasattr(task, 'submit_batch'))
    # Truncation already implies zero reward; do not spend physics on a response
    # that cannot be scored. Every complete program still uses its real reward.
    scored = iter(results)
    results = [dict(next(scored), truncated=False) if done else
               dict(reward=0., success=False, valid=False, error='truncated',
                    truncated=True, physics_started=False) for done in complete]
    return sequences, attention, mask, texts, results


def token_logprobs(model, sequences, attention, completion_length, temperature=1.):
    # Supply position IDs: generation uses positions excluding left padding too.
    position_ids = attention.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention == 0, 1)
    logits = model(input_ids=sequences, attention_mask=attention,
                   position_ids=position_ids, use_cache=False,
                   logits_to_keep=completion_length + 1).logits[:, :-1].float() / temperature
    labels = sequences[:, -completion_length:]
    return -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)


def advantages(rewards, group_size):
    grouped = rewards.reshape(-1, group_size)
    std = grouped.std(-1, unbiased=False, keepdim=True)
    return ((grouped - grouped.mean(-1, keepdim=True)) / (std + 1e-4)).flatten()


def policy_loss(logp, old, reference, mask, advantage, beta=.02, clip=.2):
    ratio = (logp - old).exp()
    surrogate = torch.minimum(ratio * advantage[:, None],
                              ratio.clamp(1-clip, 1+clip) * advantage[:, None])
    delta = reference - logp
    kl = delta.exp() - delta - 1
    per_sequence = ((-surrogate + beta * kl) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_sequence.mean(), (kl * mask).sum() / mask.sum().clamp_min(1)


def update_policy(model, optimizer, sequences, attention, mask, old, reference, advantage,
                  *, microbatch, epochs, beta, temperature):
    """Several clipped GRPO updates reuse one expensive physical rollout batch."""
    size, length = mask.shape
    metrics = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_kl = 0.
        for i in range(0, size, microbatch):
            sl = slice(i, i+microbatch)
            logp = token_logprobs(model, sequences[sl], attention[sl], length, temperature)
            loss, kl = policy_loss(logp, old[sl], reference[sl], mask[sl], advantage[sl], beta)
            fraction = sequences[sl].shape[0] / size
            (loss * fraction).backward()
            total_loss += float(loss.detach()) * fraction
            total_kl += float(kl.detach()) * fraction
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not all(map(lambda x: x == x and abs(x) != float('inf'), [total_loss, total_kl, grad_norm])):
            raise RuntimeError('nonfinite update')
        optimizer.step()
        metrics.append(dict(epoch=epoch+1, loss=total_loss, kl=total_kl, grad_norm=grad_norm))
    return metrics


def rows_for(layouts, texts, results):
    return [{"layout_id": layout.key, "prompt": task.prompt(layout), "completion": text, **result}
            for layout, text, result in zip(layouts, texts, results)]


@torch.no_grad()
def evaluate(model, tokenizer, layouts, max_tokens, batch_size=16):
    rows = []
    for start in range(0, len(layouts), batch_size):
        batch = layouts[start:start+batch_size]
        _, _, _, texts, results = rollout(model, tokenizer, batch, max_tokens, False)
        rows.extend(rows_for(batch, texts, results))
    return {"n": len(rows), "successes": sum(r["success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows)/len(rows),
            "mean_reward": sum(r["reward"] for r in rows)/len(rows),
            "valid_rate": sum(r["valid"] for r in rows)/len(rows)}, rows


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    global task
    task = get_task(args.task)
    if args.task == 'assembly_full':
        task.PHYSICS_WORKERS = args.physics_workers
        task.PHYSICS_THREADS = args.physics_threads
        task.DATASET_PROFILE = args.assembly_split
        task.configure(args.support_profile, args.controller_profile, args.assembly_backend)
    set_seed(args.seed)
    torch.set_num_threads(4)
    if args.init_adapter:
        shutil.copytree(args.init_adapter, out / "initial_adapter")
    # New Transformers text-only configs need not retain the outer commit hash.
    # Resolve once and use that immutable revision for both loading and provenance.
    resolved_revision = model_info(args.model, revision=args.revision).sha
    if args.reference_policy == 'initial' and not args.init_adapter:
        raise ValueError('Initial-adapter reference requires --init-adapter')
    model, tokenizer = load_model(args.model, resolved_revision, args.init_adapter, trainable=True,
        reference_adapter=args.init_adapter if args.reference_policy == 'initial' else None)
    splits = task.datasets()
    source_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")}
    metadata = {**vars(args), "model_revision": resolved_revision,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "transformers": transformers.__version__, "peft": peft.__version__,
                "thinking_enabled": False,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "source_sha256": source_files,
                "initial_adapter_sha256": ({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (out / "initial_adapter").iterdir() if p.is_file()} if args.init_adapter else None),
                "splits": {s: [layout.key for layout in rows] for s, rows in splits.items()},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if args.task in ("mujoco_docking", "robot_lift", "frame_place", "frame_place_ordered"):
        import mujoco
        import numpy
        metadata["physics_runtime"] = {"mujoco": mujoco.__version__, "numpy": numpy.__version__}
    if args.task in ("robot_lift", "frame_place", "frame_place_ordered"):
        from .robot_lift import asset_hashes
        metadata["robot_assets_sha256"] = asset_hashes()
    dump(out / "metadata.json", metadata)
    if args.task == 'assembly_full':
        from .assembly import ROOT
        manifest = task.provenance()
        task.EXPECTED_PROVENANCE = manifest
        dump(out / 'assembly_provenance.json', manifest)
        for name in manifest['sha256']:
            target = out / 'assembly_source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    (out / "source").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        (out / "source" / source.name).write_bytes(source.read_bytes())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    rng = random.Random(args.seed)
    baseline, rows = evaluate(model, tokenizer, splits["validation"], args.max_tokens)
    dump(out / "validation_baseline.json", {"metrics": baseline, "rows": rows})
    print(json.dumps({"stage": "baseline_validation", **baseline}), flush=True)
    best = (baseline["success_rate"], baseline["mean_reward"])
    best_step = 0
    save_policy(model, out / "best_adapter")
    tokenizer.save_pretrained(out / "best_adapter")
    started = time.monotonic()
    for step in range(1, args.steps+1):
        step_started = time.monotonic()
        layouts = [layout for layout in rng.sample(splits["train"], args.prompts_per_step)
                   for _ in range(args.group_size)]
        old, reference = [], []
        preparation = {}
        @torch.no_grad()
        def prepare(sequences, attention, mask):
            t = time.monotonic()
            size, length = mask.shape
            for i in range(0, size, args.microbatch):
                sl = slice(i, i+args.microbatch)
                old.append(token_logprobs(model, sequences[sl], attention[sl], length, args.sampling_temperature))
                with reference_policy(model):
                    reference.append(token_logprobs(model, sequences[sl], attention[sl], length, args.sampling_temperature))
            torch.cuda.synchronize()
            preparation['logprob_s'] = time.monotonic()-t
        sequences, attention, mask, texts, results = rollout(
            model, tokenizer, layouts, args.max_tokens, True, temperature=args.sampling_temperature,
            **({'prepare': prepare} if args.overlap_rollouts else {}))
        rollout_timing = dict(getattr(rollout, 'last_timing', {}))
        reward = torch.tensor([r["reward"] for r in results], device="cuda")
        advantage = advantages(reward, args.group_size)
        size, length = mask.shape
        if not args.overlap_rollouts:
            prepare(sequences, attention, mask)
        old, reference = torch.cat(old), torch.cat(reference)
        optimization_started = time.monotonic()
        epoch_metrics = update_policy(model, optimizer, sequences, attention, mask, old, reference, advantage,
            microbatch=args.microbatch, epochs=args.update_epochs, beta=args.beta, temperature=args.sampling_temperature)
        torch.cuda.synchronize()
        metric = {"step": step, "reward": float(reward.mean()),
                  "success_rate": sum(r["success"] for r in results)/size,
                  "active_groups": int((reward.reshape(-1, args.group_size).std(-1)>0).sum()),
                  "loss": epoch_metrics[-1]['loss'], "kl": epoch_metrics[-1]['kl'],
                  "grad_norm": epoch_metrics[-1]['grad_norm'],
                  "epoch_metrics": epoch_metrics, "optimizer_steps_total": step*args.update_epochs,
                  "elapsed_seconds": time.monotonic()-started,
                  "step_seconds": time.monotonic()-step_started,
                  "optimization_seconds": time.monotonic()-optimization_started,
                  **preparation, **rollout_timing,
                  "sampled_programs": size,
                  "unique_evidence": len({r['evidence_id'] for r in results if 'evidence_id' in r}),
                  "cache_hits": sum(bool(r.get('cache_hit')) for r in results),
                  "batch_duplicates": sum(bool(r.get('batch_duplicate')) for r in results)}
        append(out / "training.jsonl", [metric])
        append(out / "rollouts.jsonl", [{"step": step, **r} for r in rows_for(layouts,texts,results)])
        print(json.dumps(metric), flush=True)
        if args.checkpoint_every and (step % args.checkpoint_every == 0 or step == args.steps):
            from .checkpoints import save_step_adapter
            save_step_adapter(model, out, metric)
        if step % args.eval_every == 0 or step == args.steps:
            summary, rows = evaluate(model, tokenizer, splits["validation"], args.max_tokens)
            dump(out / f"validation_{step:04d}.json", {"metrics": summary, "rows": rows})
            print(json.dumps({"stage": "validation", "step": step, **summary}), flush=True)
            score = (summary["success_rate"], summary["mean_reward"])
            if score > best:
                best, best_step = score, step
                save_policy(model, out / "best_adapter")
    save_policy(model, out / "last_adapter")
    dump(out / "selection.json", {"best_step": best_step, "validation_success_rate": best[0],
                                  "validation_mean_reward": best[1], "selection_split": "validation",
                                  "training_seconds": time.monotonic()-started})


def test(args):
    out = Path(args.out)
    metadata = json.loads((out / "metadata.json").read_text())
    destination = out / "test_comparison.json"
    if destination.exists():
        raise FileExistsError("Test results already exist; do not tune on this test set.")
    global task
    task = get_task(metadata.get("task", "pick_place"))
    if metadata.get('task') == 'assembly_full':
        task.PHYSICS_WORKERS = metadata.get('physics_workers', 1)
        task.PHYSICS_THREADS = metadata.get('physics_threads', 1)
        task.DATASET_PROFILE = metadata.get('assembly_split', 'pilot_v1')
        task.configure(metadata.get('support_profile', 'contact_only'),
                       metadata.get('controller_profile', 'original'),
                       metadata.get('assembly_backend', 'original'))
    set_seed(metadata["seed"])
    torch.set_num_threads(4)
    layouts = task.datasets()["test"]
    task_path = Path(task.__file__)
    task_hash = hashlib.sha256(task_path.read_bytes()).hexdigest()
    if task_hash != metadata["source_sha256"][task_path.name]:
        raise RuntimeError("Task changed since training; restore the recorded task source.")
    if [layout.key for layout in layouts] != metadata["splits"]["test"]:
        raise RuntimeError("Test split changed since training.")
    if metadata.get('task') == 'assembly_full':
        if task.provenance() != json.loads((out / 'assembly_provenance.json').read_text()):
            raise RuntimeError('Full assembly source, assets or runtime changed since training')
        task.EXPECTED_PROVENANCE = task.provenance()
    if metadata.get("task") in ("robot_lift", "frame_place", "frame_place_ordered"):
        from .robot_lift import asset_hashes
        if asset_hashes() != metadata["robot_assets_sha256"]:
            raise RuntimeError("Robot assets changed since training")
        dependency = task_path.with_name("robot_lift.py")
        if hashlib.sha256(dependency.read_bytes()).hexdigest() != metadata["source_sha256"][dependency.name]:
            raise RuntimeError("Robot controller changed since training")
        if metadata.get("task") == "frame_place_ordered":
            dependency = task_path.with_name("task_frame.py")
            if hashlib.sha256(dependency.read_bytes()).hexdigest() != metadata["source_sha256"][dependency.name]:
                raise RuntimeError("Underlying frame task changed since training")
    initial_adapter = out / "initial_adapter"
    if metadata.get("initial_adapter_sha256"):
        for name, digest in metadata["initial_adapter_sha256"].items():
            if hashlib.sha256((initial_adapter/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Initial adapter changed since training")
    model, tokenizer = load_model(metadata["model"], metadata["model_revision"],
                                  str(initial_adapter) if metadata.get("init_adapter") else None)
    baseline, before = evaluate(model, tokenizer, layouts, metadata["max_tokens"])
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_model(metadata["model"], metadata["model_revision"], str(out/"best_adapter"))
    trained, after = evaluate(model, tokenizer, layouts, metadata["max_tokens"])
    wins = sum(not a["success"] and b["success"] for a,b in zip(before, after))
    losses = sum(a["success"] and not b["success"] for a,b in zip(before, after))
    deltas = [int(b["success"])-int(a["success"]) for a,b in zip(before, after)]
    rng = random.Random(20261023)
    boot = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(10000))
    result = {"baseline": baseline, "trained": trained, "paired_wins": wins,
              "paired_losses": losses, "delta_success_rate": trained["success_rate"]-baseline["success_rate"],
              "paired_bootstrap_95_ci": [boot[250], boot[9749]],
              "before": before, "after": after}
    dump(destination, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("before", "after")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--task", choices=["pick_place", "ordered_two", "mujoco_docking", "robot_lift", "frame_place", "frame_place_ordered", "assembly_full"], default="pick_place")
    parser.add_argument("--init-adapter", help="Continue RL from a previous adapter; saved as test baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--physics-workers", type=int, default=1, help="Independent full-assembly simulation workers")
    parser.add_argument('--physics-threads', type=int, default=1, help='OpenCV threads per simulation worker')
    parser.add_argument('--overlap-rollouts', action='store_true',
                        help='Compute GPU old/reference log-probabilities while independent physics runs')
    parser.add_argument('--assembly-split', choices=('pilot_v1', 'varied_v2'), default='pilot_v1')
    parser.add_argument('--sampling-temperature', type=float, default=1.)
    parser.add_argument('--update-epochs', type=int, default=1,
                        help='Clipped GRPO optimizer steps per physical rollout batch')
    parser.add_argument('--reference-policy', choices=('base', 'initial'), default='base',
                        help='KL reference: raw base or frozen initial adapter for continued RL')
    parser.add_argument('--support-profile', choices=('contact_only', 'contact_and_nails'), default='contact_only')
    parser.add_argument('--controller-profile', choices=('original', 'southwest_approach'), default='original')
    parser.add_argument('--assembly-backend', choices=('original', 'vectorized', 'vectorized_pixels', 'vectorized_native', 'vectorized_fused'), default='original',
                        help='Versioned assembly execution backend; recorded in metadata and physics cache identity')
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Save a complete adapter snapshot every N updates; 0 disables. Does not save optimizer/RNG state.')
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=.02)
    args = parser.parse_args()
    if not 0 < args.sampling_temperature < float('inf'):
        parser.error('A finite positive sampling temperature is required')
    if args.checkpoint_every < 0 or args.group_size < 2 or min(args.steps, args.prompts_per_step, args.microbatch, args.physics_workers,
                                  args.max_tokens, args.eval_every, args.physics_threads, args.update_epochs) < 1:
        parser.error("positive sizes and group_size >= 2 required")
    try:
        train(args) if args.mode == "train" else test(args)
    finally:
        if hasattr(task, 'shutdown_workers'):
            task.shutdown_workers()


if __name__ == "__main__":
    main()
