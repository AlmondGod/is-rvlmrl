"""Summarize audited harder-task results, including both requested delivery orders."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from experiments.cap_rl.audit import audit
from experiments.cap_rl.task_ordered import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checked = audit(args.directory)
    report = json.loads((args.directory/'test_comparison.json').read_text())
    metadata = json.loads((args.directory/'metadata.json').read_text())
    selection = json.loads((args.directory/'selection.json').read_text())
    layouts = datasets()['test']
    summary = {'audit': checked, 'selection': selection, 'by_order': {}, 'errors': {}}
    for side in ['before', 'after']:
        rows = report[side]
        summary['by_order'][side] = {
            color: {'n': sum(x.first == color for x in layouts),
                    'successes': sum(x.first == color and row['success'] for x, row in zip(layouts, rows))}
            for color in ['red', 'blue']}
        summary['errors'][side] = dict(Counter(
            row['error'] or ('success' if row['success'] else 'incomplete_delivery') for row in rows))
    summary['validation'] = [
        {'step': 0 if path.stem == 'validation_baseline' else int(path.stem.split('_')[1]),
         **json.loads(path.read_text())['metrics']}
        for path in args.directory.glob('validation_*.json')]
    summary['validation'].sort(key=lambda x: x['step'])
    rollout_path = args.directory/'rollouts.jsonl'
    opener = open
    if not rollout_path.exists():
        rollout_path = args.directory/'rollouts.jsonl.gz'
        opener = gzip.open
    with opener(rollout_path, 'rt') as handle:
        rollouts = [json.loads(line) for line in handle]
    summary['training_programs'] = len(rollouts)
    summary['training_layouts_seen'] = len({x['layout_id'] for x in rollouts})
    summary['training_successful_programs'] = sum(x['success'] for x in rollouts)
    summary['initial_adapter_sha256'] = metadata['initial_adapter_sha256']['adapter_model.safetensors']
    args.out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
