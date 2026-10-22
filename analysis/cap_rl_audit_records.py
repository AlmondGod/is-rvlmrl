"""Replay all saved validation and training records using each run's frozen task.

Source snapshots are trusted experiment source, never model-generated programs.
This verifies execution and accounting, not that model weights emitted the text.
"""
import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


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
    task_name = 'task_ordered' if metadata.get('task') == 'ordered_two' else 'task'
    task = importlib.import_module('.'+task_name, namespace)
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
        for key in ['success', 'valid', 'error', 'trace']:
            if replay[key] != row[key]:
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
    return {'audit': 'passed', 'training_programs_replayed': len(rows),
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
