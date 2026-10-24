import hashlib,json,shutil,time
from pathlib import Path
import torch
from experiments.cap_rl import train as trainer
from analysis import cap_rl_assembly_grammar as grammar, assembly_visual_prompt as visual
from analysis.assembly_preference_data import build
from analysis.train_assembly_preferences import pair_loss,program_inventory
parent=Path('/opt/cap-speed/varied-run03');out=Path('/opt/cap-speed/visual-preflight12');out.mkdir(exist_ok=False)
m=json.loads((parent/'metadata.json').read_text());m['prompt_profile']=visual.PROFILE
(out/'preference_source').mkdir()
for name in visual.SOURCES:shutil.copyfile(Path(visual.__file__).with_name(name),out/'preference_source'/name)
m['visual_source_sha256']={n:visual.sha(out/'preference_source'/n) for n in visual.SOURCES}
shutil.copyfile(parent/'assembly_provenance.json',out/'assembly_provenance.json')
grammar.install(trainer);trainer.task=trainer.get_task('assembly_full');trainer.task.DATASET_PROFILE=m['assembly_split']
trainer.task.configure(m['support_profile'],m['controller_profile'],m['assembly_backend'])
visual.prepare(out,m,'development');m['observations_manifest_sha256']=visual.sha(out/'observations_manifest.json')
visual.dump(out/'metadata.json',m);obs=visual.audit_observations(out,m)
data=visual.augment_preferences(build(parent),out,m);visual.dump(out/'preference_data.json',data)
chosen=[data['pairs'][0]]
chosen.append(next(p for p in data['pairs'] if p['layout_id']!=chosen[0]['layout_id']))
layouts={l.key:l for l in trainer.task.datasets()['train']}
torch.set_num_threads(4)
model,tok=trainer.load_model(m['model'],m['model_revision'],str(parent/'initial_adapter'),trainable=True,reference_adapter=str(parent/'initial_adapter'))
grammar._GRAMMAR=grammar.AssemblyGrammar(tok)
with visual.installed(out,m):
 inputs=trainer.encode(tok,[layouts[p['layout_id']] for p in chosen for role in ('chosen','rejected')])
completion=[tok.encode(p[role],add_special_tokens=False)+[tok.eos_token_id] for p in chosen for role in ('chosen','rejected')]
for c in completion:grammar._GRAMMAR.validate(c)
labels=torch.tensor(completion,device='cuda');seq=torch.cat([inputs.input_ids,labels],1);attention=torch.cat([inputs.attention_mask,torch.ones_like(labels)],1)
start=time.monotonic();losses=[]
for i in (0,2):
 with torch.no_grad(),trainer.reference_policy(model):ref=grammar.constrained_logprobs(model,seq[i:i+2],attention[i:i+2],labels.shape[1],1.).sum(-1)
 policy=grammar.constrained_logprobs(model,seq[i:i+2],attention[i:i+2],labels.shape[1],1.).sum(-1)
 assert torch.equal(policy.detach(),ref),(policy,ref)
 loss,margin=pair_loss(policy[None],ref[None],.1);loss.backward();losses.append(float(loss.detach()))
torch.cuda.synchronize();refgrads=[n for n,p in model.named_parameters() if '.reference.' in n and p.grad is not None];grads=[p.grad for p in model.parameters() if p.grad is not None]
assert not refgrads and grads and all(torch.isfinite(g).all() for g in grads)
r=dict(passes=True,optimizer_updates=0,physics_episodes=0,initial_sensor_captures=40,observation_audit=obs,losses=losses,peak_allocated_bytes=torch.cuda.max_memory_allocated(),forward_backward_seconds=time.monotonic()-start,finite_gradients=True,reference_gradients=refgrads,prompt_token_lengths=inputs.attention_mask.sum(-1).tolist(),layouts=[p['layout_id'] for p in chosen],context_programs=len(program_inventory(data,'unused',True)),weights_sha256=visual.sha(parent/'initial_adapter/adapter_model.safetensors'),source_sha256=visual.sha(__file__),scope='Real Qwen with scene-matched RGB-derived prompts; two preference pairs, no optimizer step and no task outcome.')
visual.dump(out/'result.json',r);print(json.dumps(r),flush=True)
