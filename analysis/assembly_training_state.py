"""Atomic adapter plus AdamW/RNG checkpoints for the assembly learner."""
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(model, optimizer, output, metric):
    import numpy as np
    import torch
    root = Path(output)/'checkpoints'
    root.mkdir(exist_ok=True)
    name = f"step-{metric['step']:04d}"
    pending, complete = root/('.'+name+'.pending'), root/name
    if pending.exists() or complete.exists():
        raise FileExistsError('Checkpoint already exists or is incomplete: '+name)
    pending.mkdir()
    kwargs = {'selected_adapters':['default']} if getattr(model, '_cap_reference_adapter', None) else {}
    model.save_pretrained(pending, **kwargs)
    state = dict(format='assembly_optimizer_rng_v1', step=metric['step'],
        optimizer=optimizer.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
        adapter_sha256=sha(pending/'adapter_model.safetensors'))
    torch.save(state, pending/'training_state.pt')
    record = dict(format='cap_rl_adapter_checkpoint_v1', step=metric['step'], metric=metric,
        scope='Adapter plus optimizer and RNG state, atomically published together. '
              'GPU arithmetic may remain nondeterministic even with saved RNG state.',
        training_state_format=state['format'], training_state_source_sha256=sha(__file__),
        files_sha256={p.name:sha(p) for p in pending.iterdir() if p.is_file()})
    (pending/'checkpoint.json').write_text(json.dumps(record,indent=2)+'\n')
    pending.rename(complete)
    return complete


def restore_optimizer_rng(model, optimizer, checkpoint):
    """Load trusted local state only, after loading this checkpoint's adapter."""
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint/'checkpoint.json').read_text())
    if receipt.get('training_state_source_sha256') != sha(__file__):
        raise ValueError('Training-state implementation changed')
    for name,digest in receipt['files_sha256'].items():
        if sha(checkpoint/name) != digest:
            raise ValueError('Checkpoint payload changed: '+name)
    state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=False)
    if state['format'] != 'assembly_optimizer_rng_v1' or state['step'] != receipt['step']:
        raise ValueError('Wrong training-state format or step')
    if state['adapter_sha256'] != sha(checkpoint/'adapter_model.safetensors'):
        raise ValueError('Optimizer belongs to another adapter')
    expected = load_file(str(checkpoint/'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model,adapter_name='default')
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k].detach().cpu(),expected[k]) for k in expected):
        raise ValueError('Load the checkpoint adapter before its optimizer')
    if state['trainable_parameters'] != [n for n,p in model.named_parameters() if p.requires_grad]:
        raise ValueError('Optimizer parameter order changed')
    if len(state['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('CUDA device count changed')
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']
