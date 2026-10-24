"""Global, synchronous GRPO updates for independently dispatched rollouts.

A cohort samples all layout groups from one unchanged policy, queues physics
without per-generation-batch barriers, then accumulates each optimizer update
over the entire cohort. This avoids making four consecutive updates on only
four layouts. No optimizer update runs before all actual rewards are available.
This component is not installed in the frozen online25 experiment.
"""
import math
import time

from analysis.assembly_pipelined_rollouts import parameter_versions


def validate_cohort(batches, results, group_size):
    if not batches or len(batches) != len(results) or group_size < 2:
        raise ValueError('Expected nonempty aligned batches and nontrivial groups')
    versions = {batch.policy_version for batch in batches}
    if len(versions) != 1:raise ValueError('All cohort samples must share one policy version')
    seen = set()
    for batch, rows in zip(batches, results):
        size = len(batch.layouts)
        if size != len(rows) or size % group_size:
            raise ValueError('Every sampled program needs its own measured reward')
        if batch.old is None or batch.reference is None:
            raise ValueError('Frozen old/reference probabilities are required')
        if (batch.sequences.shape != batch.attention.shape or batch.mask.ndim != 2 or
            batch.old.shape != batch.mask.shape or batch.reference.shape != batch.mask.shape or
            batch.mask.shape[1] > batch.sequences.shape[1]):
            raise ValueError('Rollout probability/mask shapes differ')
        for start in range(0, size, group_size):
            keys = [layout.key for layout in batch.layouts[start:start+group_size]]
            if len(set(keys)) != 1 or keys[0] in seen:
                raise ValueError('Each layout must occur in exactly one complete group')
            seen.add(keys[0])
        if not all(row.get('reward') in (0., 1.) and
                   bool(row.get('success')) == bool(row['reward']) for row in rows):
            raise ValueError('Cohort updates require strict binary physical outcomes')
        for tensor in (batch.sequences, batch.attention, batch.mask, batch.old, batch.reference):
            if tensor.shape[0] != size or tensor.requires_grad:
                raise ValueError('Rollout tensors must be aligned and detached')
    return seen


def update(trainer, model, optimizer, batches, results, *, group_size=4,
           microbatch=8, epochs=4, beta=.005, temperature=3.):
    """One full-cohort gradient accumulation per epoch, with frozen PPO ratios."""
    import torch
    from analysis import cap_rl_assembly_grammar as grammar
    validate_cohort(batches, results, group_size)
    if microbatch < 1 or epochs < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive optimization sizes and temperature required')
    device = next(model.parameters()).device
    size = sum(len(batch.layouts) for batch in batches)
    advantages = [trainer.advantages(torch.tensor([r['reward'] for r in rows], device=device),group_size)
                  for rows in results]
    metrics = []
    for epoch in range(epochs):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_kl = 0.
        for batch, advantage in zip(batches, advantages):
            for offset in range(0, len(batch.layouts), microbatch):
                section = slice(offset,offset+microbatch)
                sequences = batch.sequences[section].to(device)
                attention = batch.attention[section].to(device)
                mask = batch.mask[section].to(device)
                old = batch.old[section].to(device)
                reference = batch.reference[section].to(device)
                logp = grammar.constrained_logprobs(model,sequences,attention,mask.shape[1],temperature)
                loss,kl = trainer.policy_loss(logp,old,reference,mask,advantage[section],beta)
                fraction = len(sequences)/size
                (loss*fraction).backward()
                total_loss += float(loss.detach())*fraction
                total_kl += float(kl.detach())*fraction
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not all(math.isfinite(x) for x in (total_loss,total_kl,norm)):
            raise RuntimeError('Nonfinite cohort update')
        optimizer.step()
        if device.type == 'cuda':torch.cuda.synchronize()
        metrics.append(dict(epoch=epoch+1,loss=total_loss,kl=total_kl,grad_norm=norm,
                            seconds=time.monotonic()-started,sampled_programs=size))
    return metrics


def dispatch_all(trainer, model, tokenizer, layout_groups, queue, *, policy_version,
                 max_tokens, temperature=3., microbatch=8):
    """All groups retain one policy version; no reward wait between dispatches."""
    from analysis.assembly_pipelined_rollouts import dispatch
    before = parameter_versions(model)
    batches = []
    for layouts in layout_groups:
        if parameter_versions(model) != before:
            raise RuntimeError('Policy changed inside cohort generation')
        batches.append(dispatch(trainer,model,tokenizer,layouts,queue,
            policy_version=policy_version,max_tokens=max_tokens,
            temperature=temperature,microbatch=microbatch))
    if parameter_versions(model) != before:
        raise RuntimeError('Policy changed inside cohort generation')
    return batches
