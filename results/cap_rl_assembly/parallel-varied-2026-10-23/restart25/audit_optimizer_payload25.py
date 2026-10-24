import hashlib,json
from pathlib import Path
import torch
root=Path('/opt/cap-speed/varied-online25/checkpoints/step-0002')
receipt=json.loads((root/'checkpoint.json').read_text())
for name,digest in receipt['files_sha256'].items():
 with (root/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
state=torch.load(root/'training_state.pt',map_location='cpu',weights_only=False)
assert state['step']==2 and state['adapter_sha256']==receipt['files_sha256']['adapter_model.safetensors']
optim=state['optimizer'];count=0;steps=set()
for values in optim['state'].values():
 steps.add(int(values['step']))
 for name,tensor in values.items():
  if isinstance(tensor,torch.Tensor):assert torch.isfinite(tensor).all(),name;count+=tensor.numel()
assert steps=={8},steps
assert state['cuda_rng'] and state['torch_rng'].numel()>0
report=dict(passes=True,checkpoint_step=2,optimizer_steps=8,parameters_with_optimizer_state=len(optim['state']),
 trainable_parameter_tensors=len(state['trainable_parameters']),optimizer_tensor_elements=count,
 checkpoint_receipt_sha256=hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest(),
 training_state_sha256=receipt['files_sha256']['training_state.pt'],
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='Read-only CPU inspection of the real checkpoint: payload hashes, finite optimizer tensors, AdamW step counts and saved RNG payloads. No model instantiated, no update, no bitwise GPU reproducibility claim.')
(root.parent.parent/'optimizer_payload_audit_step0002.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
