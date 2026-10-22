"""Reload both checkpoints and generate programs for a custom physical layout."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import mujoco
import torch
from transformers import set_seed

from . import train
from . import task_robot as task
from .robot_lift import asset_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--start', type=int, nargs=2, default=(3, -3))
    parser.add_argument('--center', type=int, nargs=2, default=(-3, 4))
    parser.add_argument('--lift', type=int, choices=[6, 8, 10, 12], default=10)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    layout = task.Layout(tuple(args.start), tuple(args.center), args.lift)
    if any(abs(x)>3 for x in layout.start) or any(abs(x)>4 for x in layout.center):
        parser.error('Use start coordinates in [-3,3] and center coordinates in [-4,4] cm.')
    metadata = json.loads((args.directory/'metadata.json').read_text())
    if metadata.get('task') != 'robot_lift':
        parser.error('This checkpoint is not a robot lift run.')
    for name in ('task_robot.py', 'robot_lift.py'):
        if hashlib.sha256(Path(task.__file__).with_name(name).read_bytes()).hexdigest() != metadata['source_sha256'][name]:
            raise ValueError('Task or robot controller differs from training')
    if asset_hashes() != metadata['robot_assets_sha256']:
        raise ValueError('Robot assets differ from training')
    if mujoco.__version__ != metadata['physics_runtime']['mujoco']:
        raise ValueError('MuJoCo version differs from training')
    train.task = task
    torch.set_num_threads(4)
    set_seed(metadata['seed'])
    initial = args.directory/'initial_adapter' if metadata.get('init_adapter') else None
    if initial:
        for name, digest in metadata['initial_adapter_sha256'].items():
            if hashlib.sha256((initial/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Initial adapter differs from training')
    report = {'layout_id': layout.key, 'prompt': task.prompt(layout)}
    for label, adapter in [('before', initial), ('after', args.directory/'best_adapter')]:
        model, tokenizer = train.load_model(metadata['model'], metadata['model_revision'], str(adapter) if adapter else None)
        _, rows = train.evaluate(model, tokenizer, [layout], metadata['max_tokens'], batch_size=1)
        report[label] = rows[0]
        del model
        gc.collect()
        torch.cuda.empty_cache()
    text = json.dumps(report, indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
