"""Summarize physical-task evidence without running the language model."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['task'] != 'mujoco_docking':
        raise ValueError('Expected a MuJoCo docking experiment')
    path = directory/'rollouts.jsonl'
    if path.exists():
        lines = path.read_text().splitlines()
    else:
        with gzip.open(str(path)+'.gz', 'rt') as handle:
            lines = handle.readlines()
    samples = [json.loads(line) for line in lines]
    report = json.loads((directory/'test_comparison.json').read_text())
    result = {'training_steps': len({r['step'] for r in samples}),
              'training_programs': len(samples),
              'distinct_training_layouts': len({r['layout_id'] for r in samples}),
              'successful_training_programs': sum(r['success'] for r in samples),
              'selection': json.loads((directory/'selection.json').read_text()),
              'test': {k: v for k, v in report.items() if k not in ('before', 'after')}}
    for label in ('before', 'after'):
        rows = report[label]
        result[label] = {'errors': dict(Counter(r['error'] or ('success' if r['success'] else 'incomplete') for r in rows)),
                         'docked_stations': dict(Counter(r['components']['docked'] for r in rows)),
                         'mean_final_distance_m': sum(r['final_distance_m'] for r in rows)/len(rows),
                         'mean_final_speed_m_s': sum(r['final_speed_m_s'] for r in rows)/len(rows)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
