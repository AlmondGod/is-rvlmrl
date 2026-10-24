import hashlib,json,time
from pathlib import Path
import torch
from transformers import set_seed
from experiments.cap_rl import train
from analysis import cap_rl_assembly_grammar as grammar,assembly_visual_prompt as visual
from analysis.train_assembly_visual_grpo import schedule
root=Path('/opt/cap-speed/varied-visual12');metadata=visual.read(root/'metadata.json')
set_seed(97);torch.set_num_threads(4);grammar.install(train)
train.task=train.get_task('assembly_full');train.task.DATASET_PROFILE='varied_v2'
train.task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
keys=schedule(metadata['splits']['train'])[0];bykey={x.key:x for x in train.task.datasets()['train']};layouts=[bykey[k] for k in keys for _ in range(4)]
model,tok=train.load_model(metadata['model'],metadata['model_revision'],'/opt/cap-speed/varied-run03/checkpoints/step-0006',trainable=True,reference_adapter=str(root/'initial_adapter'))
train.task.execute_batch=lambda layouts,texts:[dict(reward=0.,success=False,valid=True) for _ in texts]
with visual.installed(root,metadata):
 started=time.monotonic();seq,attention,mask,texts,_=train.rollout(model,tok,layouts,metadata['max_tokens'],True,temperature=3.)
 torch.cuda.synchronize();generation=time.monotonic()-started
length=mask.shape[1];started=time.monotonic();grads=[]
for i in (0,8):
 sl=slice(i,i+8)
 with torch.no_grad():old=grammar.constrained_logprobs(model,seq[sl],attention[sl],length,3.)
 with torch.no_grad(),train.reference_policy(model):reference=grammar.constrained_logprobs(model,seq[sl],attention[sl],length,3.)
 current=grammar.constrained_logprobs(model,seq[sl],attention[sl],length,3.)
 torch.testing.assert_close(current.detach(),old)
 synthetic_advantage=torch.tensor([1.,-1.,1.,-1.]*2,device='cuda')
 loss,kl=train.policy_loss(current,old,reference,mask[sl],synthetic_advantage,.005);(loss/2).backward()
 assert torch.isfinite(loss) and torch.isfinite(kl)
assert all(p.grad is None for n,p in model.named_parameters() if '.reference.' in n)
grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(grad))
torch.cuda.synchronize()
r=dict(passes=True,optimizer_updates=0,physics_episodes=0,source_sha256=visual.sha(__file__),batch=16,microbatch=8,layouts=[l.key for l in layouts],generation_seconds=generation,forward_reference_backward_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_norm=grad,reference_gradients=False,choice_tokens=mask.sum(-1).tolist(),unique_programs=len(set(texts)),programs=texts,scope='Capacity and gradient check with synthetic advantages, no optimizer step or measured physical outcome.')
assert r['unique_programs']>=4
visual.dump('/opt/cap-speed/visual-online-preflight13.json',r);print(json.dumps({k:v for k,v in r.items() if k!='programs'}),flush=True)
