"""Compare two-object programs from a saved initial and trained adapter."""
import argparse
import gc
import json
from pathlib import Path

import torch

from . import train
from .task_ordered import Layout
from .tasks import get_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    for name in ('gripper', 'red', 'blue', 'red-target', 'blue-target'):
        parser.add_argument('--'+name, nargs=2, required=True, type=int)
    parser.add_argument('--first', choices=['red', 'blue'], required=True)
    args = parser.parse_args()
    points = [tuple(getattr(args, name)) for name in
              ('gripper', 'red', 'blue', 'red_target', 'blue_target')]
    if len(set(points)) != 5 or any(abs(x)>2 for p in points for x in p):
        parser.error('Choose five distinct cells with integer coordinates in [-2, 2].')
    metadata = json.loads((args.run/'metadata.json').read_text())
    if metadata.get('task') != 'ordered_two':
        parser.error('The run must use ordered_two.')
    train.task = get_task('ordered_two')
    torch.set_num_threads(4)
    layout = Layout(*points, first=args.first)
    result = {'layout_id': layout.key, 'prompt': train.task.prompt(layout)}
    for side, adapter in [('before', 'initial_adapter'), ('after', 'best_adapter')]:
        model, tokenizer = train.load_model(metadata['model'], metadata['model_revision'],
                                            str(args.run/adapter))
        _, _, _, texts, outcomes = train.rollout(model, tokenizer, [layout],
                                                  metadata['max_tokens'], False)
        result[side] = {'program': texts[0], **outcomes[0]}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
