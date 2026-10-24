"""Experimental cached decoding that batches grammar-forced token stretches.

The mathematical choice distribution is unchanged: normalize only over legal
indices. Stochastic random-number consumption differs from HF token-by-token
multinomial sampling. Kernel/chunk floating-point differences must be measured;
this module is not installed into an existing frozen experiment implicitly.
"""


def generate(model,input_ids,attention_mask,grammar,*,sample=False,temperature=1.,generator=None,trace=False):
    import torch
    if input_ids.ndim!=2 or attention_mask.shape!=input_ids.shape:
        raise ValueError('Expected matching batched input IDs and attention masks')
    if temperature<=0:raise ValueError('Positive sampling temperature required')
    size,prompt_length=input_ids.shape
    completion=input_ids.new_empty((size,len(grammar.tokens)))
    cache=None;consumed=0;forward_calls=0;choice_logprobs=[]
    was_training=model.training;model.eval()
    try:
        with torch.inference_mode():
            for position in range(len(grammar.tokens)):
                prefix=completion[:,:position].tolist()
                legal=[grammar.allowed(position,row) for row in prefix]
                counts={len(row) for row in legal}
                if len(counts)!=1 or 0 in counts:raise ValueError('Incompatible batched grammar branches')
                if counts=={1}:
                    completion[:,position]=torch.tensor([row[0] for row in legal],device=input_ids.device)
                    continue
                # Every unconsumed token (including the previous choice) is
                # processed exactly once. No cache/state is skipped or reset.
                extended_mask=torch.cat([attention_mask,attention_mask.new_ones((size,position))],dim=1)
                positions=extended_mask.long().cumsum(-1)-1
                positions.masked_fill_(extended_mask==0,1)
                if cache is None:
                    tokens=torch.cat([input_ids,completion[:,:position]],dim=1)
                else:
                    tokens=completion[:,consumed:position]
                    positions=positions[:,-tokens.shape[1]:]
                if tokens.shape[1]==0:raise ValueError('Empty incremental model input')
                result=model(input_ids=tokens,attention_mask=extended_mask,position_ids=positions,
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values
                if cache is None:raise ValueError('Model did not return the required cache')
                consumed=position;forward_calls+=1
                scores=result.logits[:,-1,grammar.digits].float()/temperature
                allowed=torch.tensor([[d in row for d in grammar.digits] for row in legal],device=scores.device,dtype=torch.bool)
                scores=scores.masked_fill(~allowed,float('-inf'))
                if sample:choice=torch.multinomial(scores.softmax(-1),1,generator=generator).squeeze(1)
                else:choice=scores.argmax(-1)
                if trace:choice_logprobs.append(scores.log_softmax(-1).gather(1,choice[:,None]).squeeze(1))
                completion[:,position]=torch.tensor(grammar.digits,device=input_ids.device)[choice]
            for row in completion.tolist():grammar.validate(row)
            return torch.cat([input_ids,completion],dim=1),dict(
                forward_calls=forward_calls,forced_tokens=len(grammar.tokens)-forward_calls,
                completion_tokens=len(grammar.tokens),batch=size,
                **({'choice_logprobs':torch.stack(choice_logprobs,dim=1).cpu().tolist()} if trace else {}),
                scope='Grammar-forced stretches batched through the cache; choice-only sampling has its own RNG stream.')
    finally:model.train(was_training)
