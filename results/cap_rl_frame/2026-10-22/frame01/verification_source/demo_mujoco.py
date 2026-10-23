"""Reload both checkpoints and generate programs for a custom physical layout."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import set_seed

from . import train
from . import task_mujoco as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--start', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--first', type=int, nargs=2, default=(2, -1))
    parser.add_argument('--second', type=int, nargs=2, default=(-1, 2))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    layout = task.Layout(tuple(args.start), tuple(args.first), tuple(args.second))
    if len({layout.start, layout.first, layout.second}) != 3 or any(abs(x)>2 for p in (layout.start, layout.first, layout.second) for x in p):
        parser.error('Use three distinct coordinate pairs with integer values in [-2,2] (10 cm units).')
    metadata = json.loads((args.directory/'metadata.json').read_text())
    if metadata.get('task') != 'mujoco_docking':
        parser.error('This checkpoint is not a MuJoCo docking run.')
    if hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest() != metadata['source_sha256']['task_mujoco.py']:
        raise ValueError('Task source differs from training')
    if task.mujoco.__version__ != metadata['physics_runtime']['mujoco']:
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
