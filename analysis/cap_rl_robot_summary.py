"""Summarize saved robot results without rerunning or changing the test split."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    report = json.loads((directory/'test_comparison.json').read_text())
    metadata = json.loads((directory/'metadata.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    path = directory/'rollouts.jsonl'
    opener = open
    if not path.exists():
        path = directory/'rollouts.jsonl.gz'; opener = gzip.open
    with opener(path, 'rt') as f:
        train = [json.loads(line) for line in f]
    validation = [{'step': 0, **json.loads((directory/'validation_baseline.json').read_text())['metrics']}]
    validation += [{'step': int(p.stem.split('_')[1]), **json.loads(p.read_text())['metrics']}
                   for p in sorted(directory.glob('validation_[0-9]*.json'))]
    phases = {}
    for phase, key in [('before', 'before'), ('after', 'after')]:
        rows = report[key]
        phases[phase] = {'successes': sum(r['success'] for r in rows), 'n': len(rows),
            'outcomes': dict(Counter('success' if r['success'] else r['error'] or 'physical_criteria' for r in rows)),
            'max_success_penetration_m': max((r['final']['peak_penetration'] for r in rows if r['success']), default=None),
            'max_success_final_tilt_rad': max((r['final']['tilt'] for r in rows if r['success']), default=None)}
    return {'task': metadata['task'], 'training_programs': len(train),
            'distinct_training_layouts': len({r['layout_id'] for r in train}),
            'successful_training_programs': sum(r['success'] for r in train),
            'selection': selection, 'validation': validation, 'test': phases,
            'paired_wins': report['paired_wins'], 'paired_losses': report['paired_losses'],
            'delta_success_rate': report['delta_success_rate'],
            'paired_bootstrap_95_ci': report['paired_bootstrap_95_ci']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    text = json.dumps(summarize(args.directory), indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
