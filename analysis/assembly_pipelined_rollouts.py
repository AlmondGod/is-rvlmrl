"""Nonblocking constrained generation with real, independently queued rewards.

This is a rollout component, not a new learner. A caller must record policy
versions and define update/selection order. It never fabricates a reward to
make a blocking learner return early. Current frozen experiments do not use it.
"""
from dataclasses import dataclass
import time
import math


def parameter_versions(model):
    return tuple((name,id(p),p._version) for name,p in model.named_parameters() if p.requires_grad)


@dataclass
class PendingPrograms:
    pending: object
    policy_version: str
    layouts: list
    sequences: object
    attention: object
    mask: object
    texts: list
    old: object
    reference: object
    timing: dict
    started: float

    def ready(self):
        return self.pending.ready()

    def collect(self):
        # Worker failures propagate; callers must not replace them with zero.
        start=time.monotonic()
        results=self.pending.result()
        if len(results)!=len(self.layouts):raise ValueError('Incomplete physical reward batch')
        self.timing.update(collect_wait_s=time.monotonic()-start,
                           dispatch_to_collection_s=time.monotonic()-self.started)
        return [dict(r,truncated=False) for r in results]


def dispatch(trainer,model,tokenizer,layouts,queue,*,policy_version,max_tokens,
             sample=True,temperature=1.,microbatch=8,prepare_logprobs=True):
    """Generate once, submit immediately, and retain frozen old/reference scores.

    All returned tensors reside on CPU. Several batches can be queued before
    any collection. Callers transfer individual optimization microbatches back
    to the GPU, so outstanding simulations do not retain model activations.
    """
    import torch
    from analysis import cap_rl_assembly_grammar as grammar
    if not policy_version or not layouts or microbatch<1 or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Policy version, nonempty layouts and positive sizes required')
    started=time.monotonic();versions=parameter_versions(model)
    with torch.no_grad():
        inputs=trainer.encode(tokenizer,layouts)
        kwargs=grammar.generation_kwargs(tokenizer,inputs.input_ids.shape[1],max_tokens)
        sampling=dict(temperature=temperature,top_p=1.,top_k=0) if sample else {}
        training=model.training;model.eval()
        try:
            sequences=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=sample,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                **sampling,**kwargs)
        finally:model.train(training)
        completions=sequences[:,inputs.input_ids.shape[1]:]
        for row in completions.tolist():grammar._GRAMMAR.validate(row)
        full_mask=trainer.completion_mask(completions,tokenizer.eos_token_id)
        if not completions.eq(tokenizer.eos_token_id).any(-1).all():
            raise ValueError('Constrained program was truncated')
        attention=torch.cat([inputs.attention_mask,full_mask.long()],dim=1)
        mask=grammar._GRAMMAR.decision_mask(full_mask)
        texts=tokenizer.batch_decode(completions,skip_special_tokens=True)
        if sequences.is_cuda:torch.cuda.synchronize()
        generated=time.monotonic()
        pending=queue.submit(layouts,texts)
        old=[];reference=[]
        if prepare_logprobs:
            for offset in range(0,len(layouts),microbatch):
                section=slice(offset,offset+microbatch)
                old.append(grammar.constrained_logprobs(model,sequences[section],attention[section],mask.shape[1],temperature).cpu())
                with trainer.reference_policy(model):
                    reference.append(grammar.constrained_logprobs(model,sequences[section],attention[section],mask.shape[1],temperature).cpu())
        if sequences.is_cuda:torch.cuda.synchronize()
        if versions!=parameter_versions(model):raise RuntimeError('Policy weights changed during dispatch')
        result=PendingPrograms(pending,str(policy_version),list(layouts),sequences.cpu(),attention.cpu(),mask.cpu(),texts,
            torch.cat(old) if old else None,torch.cat(reference) if reference else None,
            dict(generation_s=generated-started,submit_and_logprob_s=time.monotonic()-generated,
                 sampled_programs=len(layouts),old_reference_frozen=prepare_logprobs),started)
    return result
