import hashlib,json,time
from pathlib import Path
import torch
from experiments.cap_rl import train as trainer
from analysis import cap_rl_assembly_grammar as grammar
from analysis.train_assembly_preferences import pair_loss
root=Path('/opt/cap-speed/varied-run03');m=json.loads((root/'metadata.json').read_text());data=json.loads(Path('/opt/cap-speed/varied-preferences-step5.json').read_text());pair=data['pairs'][0]
grammar.install(trainer);trainer.task=trainer.get_task('assembly_full');trainer.task.DATASET_PROFILE='varied_v2'
model,tok=trainer.load_model(m['model'],m['model_revision'],str(root/'initial_adapter'),trainable=True,reference_adapter=str(root/'initial_adapter'))
grammar._GRAMMAR=grammar.AssemblyGrammar(tok)
completion=[tok.encode(pair[k],add_special_tokens=False)+[tok.eos_token_id] for k in ('chosen','rejected')]
for c in completion:grammar._GRAMMAR.validate(c)
inputs=trainer.encode(tok,[trainer.task.datasets()['train'][0]]*2);labels=torch.tensor(completion,device='cuda')
seq=torch.cat([inputs.input_ids,labels],1);attention=torch.cat([inputs.attention_mask,torch.ones_like(labels)],1)
with torch.no_grad(),trainer.reference_policy(model):ref=grammar.constrained_logprobs(model,seq,attention,labels.shape[1],1.).sum(-1)
start=time.monotonic();policy=grammar.constrained_logprobs(model,seq,attention,labels.shape[1],1.).sum(-1)
loss,margin=pair_loss(policy[None],ref[None],.1);loss.backward();torch.cuda.synchronize()
refgrads=[n for n,p in model.named_parameters() if '.reference.' in n and p.grad is not None]
grads=[p.grad for p in model.parameters() if p.grad is not None]
assert not refgrads and grads and all(torch.isfinite(g).all() for g in grads)
assert torch.equal(policy.detach(),ref), (policy,ref)
r=dict(passes=True,optimizer_updates=0,physics_episodes=0,loss=float(loss.detach()),initial_margin=float(margin.detach()),
 peak_allocated_bytes=torch.cuda.max_memory_allocated(),forward_backward_seconds=time.monotonic()-start,finite_gradients=True,reference_gradients=refgrads,
 programs=[pair['chosen'],pair['rejected']],weights_sha256=hashlib.sha256((root/'initial_adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Real Qwen pair forward/backward with certified training programs; no optimizer step or physical evaluation.')
Path('/opt/cap-speed/preference_preflight04.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
