"""Publish complete adapter snapshots without exposing a partial checkpoint."""
import hashlib
import json
from pathlib import Path


def save_step_adapter(model, output, metric):
    root = Path(output)/'checkpoints'
    root.mkdir(exist_ok=True)
    name = f"step-{metric['step']:04d}"
    pending, complete = root/('.'+name+'.pending'), root/name
    if pending.exists() or complete.exists():
        raise FileExistsError('Checkpoint already exists or is incomplete: '+name)
    pending.mkdir()
    kwargs = {'selected_adapters': ['default']} if getattr(model, '_cap_reference_adapter', None) else {}
    model.save_pretrained(pending, **kwargs)
    record = dict(format='cap_rl_adapter_checkpoint_v1', step=metric['step'], metric=metric,
        scope='Adapter weights only, not optimizer/RNG resume state. A warm start reinitializes the optimizer. Validation selects the final policy separately.',
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in pending.iterdir() if p.is_file()})
    (pending/'checkpoint.json').write_text(json.dumps(record, indent=2)+'\n')
    pending.rename(complete)
    return complete
