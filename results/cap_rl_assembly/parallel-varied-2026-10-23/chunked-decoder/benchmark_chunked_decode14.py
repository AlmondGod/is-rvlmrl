import json,time,statistics,gc
from pathlib import Path
import torch
from experiments.cap_rl import train
from analysis import cap_rl_assembly_grammar as grammar,assembly_visual_prompt as visual
from assembly_chunked_decode14 import generate
# Keep this independent inference probe below 40% of device memory while the
# active simulator/trainer retains its own model. No optimizer or physics here.
torch.cuda.set_per_process_memory_fraction(.40);torch.set_num_threads(2)
root=Path('/opt/cap-speed/varied-visual12');m=visual.read(root/'metadata.json')
train.task=train.get_task('assembly_full');train.task.DATASET_PROFILE='varied_v2'
model,tok=train.load_model(m['model'],m['model_revision'],'/opt/cap-speed/varied-run03/checkpoints/step-0006')
g=grammar.AssemblyGrammar(tok)
layouts=train.task.datasets()['train']+train.task.datasets()['validation'];rows=[];timings=[]
def baseline(inputs):
 with torch.inference_mode():
  return model.generate(**inputs,max_new_tokens=m['max_tokens'],do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
      **grammar.generation_kwargs(tok,inputs.input_ids.shape[1],m['max_tokens']))
with visual.installed(root,m):
 inputs=train.encode(tok,layouts[:2]);a=baseline(inputs);b,info=generate(model,inputs.input_ids,inputs.attention_mask,g)
 assert torch.equal(a,b),'Initial greedy chunked output mismatch'
 for trial in range(3):
  for name in (('hf','chunked') if trial%2==0 else ('chunked','hf')):
   torch.cuda.synchronize();start=time.monotonic()
   result=baseline(inputs) if name=='hf' else generate(model,inputs.input_ids,inputs.attention_mask,g)[0]
   torch.cuda.synchronize();timings.append(dict(trial=trial,decoder=name,seconds=time.monotonic()-start))
   assert torch.equal(result,a)
 for i in range(0,len(layouts),2):
  inputs=train.encode(tok,layouts[i:i+2]);a=baseline(inputs);b,info=generate(model,inputs.input_ids,inputs.attention_mask,g)
  for j,l in enumerate(layouts[i:i+2]):rows.append(dict(layout_id=l.key,exact=bool(torch.equal(a[j],b[j])),completion=tok.decode(b[j,inputs.input_ids.shape[1]:],skip_special_tokens=True)))
  print('GREEDY_PAIR',i,all(r['exact'] for r in rows[-2:]),flush=True)
 inputs=train.encode(tok,layouts[:2]);torch.manual_seed(14)
 sample,trace=generate(model,inputs.input_ids,inputs.attention_mask,g,sample=True,temperature=3.,trace=True)
 attention=torch.cat([inputs.attention_mask,inputs.attention_mask.new_ones((2,len(g.tokens)))],1);grammar._GRAMMAR=g
 with torch.inference_mode():logp=grammar.constrained_logprobs(model,sample,attention,len(g.tokens),3.)
 decisions=[p for p in g.choices if p!=g.groups[g.choice_group[p]][-1]]
 delta=(logp[:,decisions].cpu()-torch.tensor(trace['choice_logprobs'])).abs()
r=dict(passes=all(x['exact'] for x in rows),scope='Development-only inference probe; no optimizer, task outcomes or held-out inputs. RNG streams differ; floating-point chunk effects measured separately.',rows=rows,timings=timings,
 median_seconds={name:statistics.median(x['seconds'] for x in timings if x['decoder']==name) for name in ('hf','chunked')},
 max_abs_choice_logprob_difference=float(delta.max()),mean_abs_choice_logprob_difference=float(delta.mean()),
 chunked_info=info,peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256=visual.sha('/opt/cap-speed/assembly_chunked_decode14.py'),benchmark_source_sha256=visual.sha(__file__),model_revision=m['model_revision'],weights_sha256=visual.sha('/opt/cap-speed/varied-run03/checkpoints/step-0006/adapter_model.safetensors'))
r['speedup']=r['median_seconds']['hf']/r['median_seconds']['chunked']
visual.dump('/opt/cap-speed/chunked-decode14.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
