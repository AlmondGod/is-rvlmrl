"""Replay all saved validation and training records using each run's frozen task.

Source snapshots are trusted experiment source, never model-generated programs.
This verifies execution and accounting, not that model weights emitted the text.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

from experiments.cap_rl.audit import equivalent


def load_frozen_task(source, task_name):
    source = Path(source)
    namespace = '_cap_frozen_'+hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(namespace, source/'__init__.py',
                                                 submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    return importlib.import_module('.'+task_name, namespace), namespace


def init_replay_worker(source, task_name):
    global worker_task, worker_layouts
    worker_task, _ = load_frozen_task(source, task_name)
    worker_layouts = {x.key: x for rows in worker_task.datasets().values() for x in rows}


def replay_worker(key):
    layout_id, completion, truncated = key
    replay = worker_task.execute(worker_layouts[layout_id], completion)
    if truncated:
        replay.update(reward=0., success=False, valid=False, error='truncated')
    return key, json.loads(json.dumps(replay))


def audit_records(directory, workers=1):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    source = directory/'source'
    for name, digest in metadata['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source mismatch: {name}')
    task_name = {'ordered_two': 'task_ordered', 'mujoco_docking': 'task_mujoco',
                 'frame_place_ordered': 'frame_order', 'frame_place': 'task_frame', 'robot_lift': 'task_robot',
                 'pick_place': 'task'}[metadata.get('task', 'pick_place')]
    task, namespace = load_frozen_task(source, task_name)
    physics = metadata.get('task') in ('mujoco_docking', 'robot_lift', 'frame_place', 'frame_place_ordered')
    if physics:
        import mujoco
        if mujoco.__version__ != metadata['physics_runtime']['mujoco']:
            raise ValueError('MuJoCo runtime differs from training')
    if metadata.get('task') in ('robot_lift', 'frame_place', 'frame_place_ordered'):
        robot = importlib.import_module('.robot_lift', namespace)
        if robot.asset_hashes() != metadata['robot_assets_sha256']:
            raise ValueError('Robot assets differ from training')
    splits = task.datasets()
    for name, layouts in splits.items():
        if [x.key for x in layouts] != metadata['splits'][name]:
            raise ValueError(f'Split mismatch: {name}')
    byid = {x.key: x for layouts in splits.values() for x in layouts}

    # Optional CPU-only prefetch. Every row is still compared below, including
    # duplicates; this changes execution scheduling, not the audit tolerance.
    precomputed = {}
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers > 1:
        pending = []
        for path in directory.glob('validation_*.json'):
            pending.extend(json.loads(path.read_text())['rows'])
        path = directory/'rollouts.jsonl'
        opener = open
        if not path.exists():
            path = directory/'rollouts.jsonl.gz'; opener = gzip.open
        with opener(path, 'rt') as handle:
            pending.extend(json.loads(line) for line in handle)
        keys = list(dict.fromkeys((r['layout_id'], r['completion'], r['truncated']) for r in pending))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_replay_worker, initargs=(str(source.resolve()), task_name)) as pool:
            precomputed = dict(pool.map(replay_worker, keys, chunksize=1))

    replay_cache = {}
    unique_executions = cached_checks = 0

    def check(row):
        nonlocal unique_executions, cached_checks
        layout = byid[row['layout_id']]
        if row['prompt'] != task.prompt(layout):
            raise ValueError('Prompt mismatch')
        # The robot simulator is deterministic and resets every episode. Reuse
        # a replay only for identical layout, exact program, and truncation flag;
        # every saved record still has all fields checked against that replay.
        cache_key = (layout.key, row['completion'], row['truncated'])
        if metadata.get('task') in ('robot_lift', 'frame_place', 'frame_place_ordered') and cache_key in replay_cache:
            replay = replay_cache[cache_key]
            cached_checks += 1
        else:
            if cache_key in precomputed:
                replay = precomputed[cache_key]
            else:
                replay = task.execute(layout, row['completion'])
                if row['truncated']:
                    replay.update(reward=0., success=False, valid=False, error='truncated')
                replay = json.loads(json.dumps(replay))
            unique_executions += 1
            if metadata.get('task') in ('robot_lift', 'frame_place', 'frame_place_ordered'):
                replay_cache[cache_key] = replay
        keys = ['success', 'valid', 'error', 'trace']
        if metadata.get("task") in ("frame_place", "frame_place_ordered"):
            keys += ["lifted", "arrived"]
        if metadata.get("task") == "frame_place_ordered":
            keys += ["order_check"]
        if metadata.get('task') in ('robot_lift', 'frame_place', 'frame_place_ordered'):
            keys += ['final', 'components', 'time']
        elif physics:
            keys += ['final_state', 'final_distance_m', 'final_speed_m_s', 'components']
        for key in keys:
            if (key in replay) != (key in row) or not equivalent(replay.get(key), row.get(key), 1e-7 if metadata.get("task") in ("robot_lift", "frame_place", "frame_place_ordered") else 1e-8 if physics else 0.):
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
    if metadata.get('task') in ('robot_lift', 'frame_place', 'frame_place_ordered'):
        return {'audit': 'passed', 'training_steps_verified': len(metrics),
                'training_records_verified': len(rows), 'validation_records_verified': validation_count,
                'unique_programs_executed': unique_executions, 'cached_record_checks': cached_checks,
                'source_hashes_verified': len(metadata['source_sha256']),
                'robot_asset_hashes_verified': len(metadata['robot_assets_sha256']),
                'trace_absolute_tolerance': 1e-7, 'reward_absolute_tolerance': 1e-9}
    return {'audit': 'passed', 'training_steps_verified': len(metrics),
            'training_programs_replayed': len(rows),
            'validation_programs_replayed': validation_count,
            'source_hashes_verified': len(metadata['source_sha256'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    text = json.dumps(audit_records(args.directory, args.workers), indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
