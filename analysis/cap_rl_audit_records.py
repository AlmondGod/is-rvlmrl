"""Replay all saved validation and training records using each run's frozen task.

Source snapshots are trusted experiment source, never model-generated programs.
This verifies execution and accounting, not that model weights emitted the text.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

from experiments.cap_rl.audit import equivalent


def audit_records(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    source = directory/'source'
    for name, digest in metadata['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source mismatch: {name}')
    namespace = '_cap_frozen_'+hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(namespace, source/'__init__.py',
                                                 submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    task_name = {'ordered_two': 'task_ordered', 'mujoco_docking': 'task_mujoco',
                 'pick_place': 'task'}[metadata.get('task', 'pick_place')]
    task = importlib.import_module('.'+task_name, namespace)
    physics = metadata.get('task') == 'mujoco_docking'
    if physics and task.mujoco.__version__ != metadata['physics_runtime']['mujoco']:
        raise ValueError('MuJoCo runtime differs from training')
    splits = task.datasets()
    for name, layouts in splits.items():
        if [x.key for x in layouts] != metadata['splits'][name]:
            raise ValueError(f'Split mismatch: {name}')
    byid = {x.key: x for layouts in splits.values() for x in layouts}

    def check(row):
        layout = byid[row['layout_id']]
        if row['prompt'] != task.prompt(layout):
            raise ValueError('Prompt mismatch')
        replay = task.execute(layout, row['completion'])
        if row['truncated']:
            replay.update(reward=0., success=False, valid=False, error='truncated')
        replay = json.loads(json.dumps(replay))
        keys = ['success', 'valid', 'error', 'trace']
        if physics:
            keys += ['final_state', 'final_distance_m', 'final_speed_m_s', 'components']
        for key in keys:
            if not equivalent(replay[key], row[key], 1e-8 if physics else 0.):
                raise ValueError(f'Replay mismatch: {row["layout_id"]}/{key}')
        if abs(replay['reward']-row['reward']) > 1e-9:
            raise ValueError('Reward mismatch')

    validation_count = 0
    for path in directory.glob('validation_*.json'):
        report = json.loads(path.read_text())
        rows = report['rows']
        if [x['layout_id'] for x in rows] != metadata['splits']['validation']:
            raise ValueError('Validation rows mismatch')
        for row in rows:
            check(row)
        expected = {'n': len(rows), 'successes': sum(x['success'] for x in rows),
                    'success_rate': sum(x['success'] for x in rows)/len(rows),
                    'valid_rate': sum(x['valid'] for x in rows)/len(rows),
                    'mean_reward': sum(x['reward'] for x in rows)/len(rows)}
        if any(abs(value-report['metrics'][key]) > 1e-12 for key, value in expected.items()):
            raise ValueError('Validation aggregate mismatch')
        validation_count += len(rows)
    path = directory/'rollouts.jsonl'
    opener = open
    if not path.exists():
        path = directory/'rollouts.jsonl.gz'
        opener = gzip.open
    train_ids = set(metadata['splits']['train'])
    with opener(path, 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        if row['layout_id'] not in train_ids:
            raise ValueError('Non-training layout in rollouts')
        check(row)
    by_step = defaultdict(list)
    for row in rows:
        by_step[row['step']].append(row)
    metrics = [json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    if sorted(by_step) != [row['step'] for row in metrics]:
        raise ValueError('Training step coverage mismatch')
    expected_count = metadata['group_size'] * metadata['prompts_per_step']
    for metric in metrics:
        batch = by_step[metric['step']]
        if len(batch) != expected_count:
            raise ValueError('Incomplete rollout group')
        if abs(sum(x['reward'] for x in batch)/len(batch)-metric['reward']) > 1e-6:
            raise ValueError('Training reward aggregate mismatch')
        if sum(x['success'] for x in batch)/len(batch) != metric['success_rate']:
            raise ValueError('Training success aggregate mismatch')
    return {'audit': 'passed', 'training_steps_verified': len(metrics),
            'training_programs_replayed': len(rows),
            'validation_programs_replayed': validation_count,
            'source_hashes_verified': len(metadata['source_sha256'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit_records(args.directory), indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
