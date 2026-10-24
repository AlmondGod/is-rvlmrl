"""GPU smoke check for generation, finite gradients and batch padding consistency.

Uses training layouts only and performs no optimizer update. Saves measurements,
not hard thresholds for BF16 generation-vs-full-forward numerical differences.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from . import train as tr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    tr.task=tr.get_task('frame_place')
    start=time.monotonic()
    model,tok=tr.load_model(a.model,a.revision,trainable=True)
    layouts=tr.task.datasets()['train'][:2]
    inputs=tr.encode(tok,layouts)
    with torch.no_grad():
        generated=model.generate(**inputs,max_new_tokens=160,do_sample=False,return_dict_in_generate=True,output_scores=True)
        sequences=generated.sequences
        length=sequences.shape[1]-inputs.input_ids.shape[1]
        tokens=sequences[:,-length:]
        mask=tr.completion_mask(tokens,tok.eos_token_id)
        attention=torch.cat([inputs.attention_mask,mask.long()],1)
        generated_logp=torch.stack([torch.log_softmax(s.float(),-1).gather(1,tokens[:,i,None]).squeeze(1) for i,s in enumerate(generated.scores)],1)
        full_logp=tr.token_logprobs(model,sequences,attention,length)
        difference=(generated_logp-full_logp).abs()[mask]
        single_logp=tr.token_logprobs(model,sequences[:1],attention[:1],length)
        batch_difference=(single_logp-full_logp[:1]).abs()[mask[:1]]
    logp=tr.token_logprobs(model,sequences[:1],attention[:1],length)
    loss=-(logp*mask[:1]).sum()/mask[:1].sum()
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite=bool(gradients and all(torch.isfinite(g).all() for g in gradients))
    if not finite or not any(torch.count_nonzero(g) for g in gradients):
        raise RuntimeError('No finite nonzero LoRA gradients')
    result={'model':a.model,'revision':a.revision,'finite_nonzero_gradients':finite,
            'generated_programs':tok.batch_decode(tokens,skip_special_tokens=True),
            'generation_vs_teacher_logprob_abs_mean':float(difference.mean()),
            'generation_vs_teacher_logprob_abs_max':float(difference.max()),
            'batch_vs_single_logprob_abs_mean':float(batch_difference.mean()),
            'batch_vs_single_logprob_abs_max':float(batch_difference.max()),
            'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
            'seconds':time.monotonic()-start}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
