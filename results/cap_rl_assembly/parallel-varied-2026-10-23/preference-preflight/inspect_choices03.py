import json,hashlib,time
from pathlib import Path
import torch
from experiments.cap_rl import train as trainer
from analysis.cap_rl_assembly_grammar import generation_kwargs,AssemblyGrammar
root=Path('/opt/cap-speed/varied-run03'); metadata=json.loads((root/'metadata.json').read_text())
trainer.task=trainer.get_task('assembly_full');trainer.task.DATASET_PROFILE='varied_v2'
trainer.task.configure(metadata['support_profile'],metadata['controller_profile'],metadata['assembly_backend'])
paths=[root/'initial_adapter']+[p for p in sorted((root/'checkpoints').glob('step-*')) if (p/'checkpoint.json').exists()]
model,tokenizer=trainer.load_model(metadata['model'],metadata['model_revision'],str(paths[0]))
inputs=trainer.encode(tokenizer,[trainer.task.datasets()['train'][0]])
rows=[]
for i,p in enumerate(paths):
 if i:
  model.load_adapter(str(p),adapter_name='probe',is_trainable=False); model.set_adapter('probe')
 with torch.no_grad():
  seq=model.generate(**inputs,max_new_tokens=512,do_sample=False,**generation_kwargs(tokenizer,inputs.input_ids.shape[1],512))
  completion=seq[0,inputs.input_ids.shape[1]:].tolist(); grammar=AssemblyGrammar(tokenizer);grammar.validate(completion)
 row=dict(checkpoint=str(p),weights_sha256=hashlib.sha256((p/'adapter_model.safetensors').read_bytes()).hexdigest(),completion=tokenizer.decode(completion,skip_special_tokens=True))
 rows.append(row);print(json.dumps(row),flush=True)
 if i:model.set_adapter('default');model.delete_adapter('probe')
Path('/opt/cap-speed/varied-choice-diagnostic03.json').write_text(json.dumps(dict(scope='Read-only greedy decoding; no physical execution, reward, selection or optimizer update.',records=rows),indent=2)+'\n')
