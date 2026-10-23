"""Summarize saved robot results without rerunning or changing the test split."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from experiments.cap_rl.tasks import get_task


def summarize(directory):
    directory = Path(directory)
    report = json.loads((directory/'test_comparison.json').read_text())
    metadata = json.loads((directory/'metadata.json').read_text())
    task = get_task(metadata['task'])
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
    layouts = {l.key: l for l in task.datasets()['test']}
    def canonical(row):
        try:
            return task.parse_program(row['completion']) == task.parse_program(task.oracle(layouts[row['layout_id']]))
        except (ValueError, SyntaxError):
            return False
    for phase, key in [('before', 'before'), ('after', 'after')]:
        rows = report[key]
        phases[phase] = {'successes': sum(r['success'] for r in rows), 'n': len(rows),
            ('canonical_eight_call_program_matches' if metadata['task'] == 'frame_place' else 'canonical_four_call_program_matches'): sum(canonical(r) for r in rows),
            'outcomes': dict(Counter('success' if r['success'] else r['error'] or 'physical_criteria' for r in rows)),
            'max_success_penetration_m': max((r['final']['peak_penetration'] for r in rows if r['success']), default=None),
            'max_success_final_tilt_rad': max((r['final']['tilt'] for r in rows if r['success']), default=None)}
        if metadata['task'] == 'frame_place':
            successes = [r for r in rows if r['success']]
            counts = Counter()
            repeated_lifts = 0
            for row in rows:
                try:
                    calls = task.parse_program(row['completion'])
                    counts[len(calls)] += 1
                    repeated_lifts += sum(name == 'lift' for name, _ in calls) > 1
                except (ValueError, SyntaxError):
                    pass
            phases[phase].update(
                syntactically_valid_program_call_counts=dict(sorted(counts.items())),
                programs_with_multiple_lift_calls=repeated_lifts,
                max_success_target_error_m=max((math.dist(r['final']['wood'][:2],
                    [float(o)+v*.01 for o, v in zip(task.ORIGIN, layouts[r['layout_id']].target)])
                    for r in successes), default=None),
                max_success_final_abs_yaw_rad=max((abs(r['final']['yaw']) for r in successes), default=None),
                max_success_final_contact_n=max((r['final']['grip_force'] for r in successes), default=None),
                failed_test_indices_one_based=[i+1 for i, r in enumerate(rows) if not r['success']])
    selected = [r for r in train if r['step'] <= selection['best_step']]
    return {'task': metadata['task'], 'training_programs': len(train),
            'selected_checkpoint_training_programs': len(selected),
            'selected_checkpoint_distinct_training_layouts': len({r['layout_id'] for r in selected}),
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
