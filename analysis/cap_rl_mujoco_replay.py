"""Build an inline replay of all held-out MuJoCo trajectories and exact code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from experiments.cap_rl.audit import audit, equivalent
from experiments.cap_rl import task_mujoco as task


def build(directory, output, validation_step=None):
    directory, output = Path(directory), Path(output)
    split = 'test' if validation_step is None else 'validation'
    if split == 'test':
        audit(directory)
        report = json.loads((directory/'test_comparison.json').read_text())
    else:
        metadata = json.loads((directory/'metadata.json').read_text())
        if hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest() != metadata['source_sha256']['task_mujoco.py']:
            raise ValueError('Task source mismatch')
        report = {'before': json.loads((directory/'validation_baseline.json').read_text())['rows'],
                  'after': json.loads((directory/f'validation_{validation_step:04d}.json').read_text())['rows']}
        for rows in report.values():
            if [r['layout_id'] for r in rows] != metadata['splits'][split]:
                raise ValueError('Validation case coverage mismatch')
    cases = []
    for i, layout in enumerate(task.datasets()[split]):
        case = {'id': layout.key, 'start': task.meters(layout.start),
                'first': task.meters(layout.first), 'second': task.meters(layout.second)}
        for label in ('before', 'after'):
            row = report[label][i]
            replay = task.execute(layout, row['completion'], record_frames=True)
            if row['truncated']:
                replay.update(reward=0., success=False, valid=False, error='truncated')
            for key in ('reward', 'success', 'valid', 'error', 'trace', 'final_state'):
                if not equivalent(replay[key], row[key], 1e-8):
                    raise ValueError(f'Frame replay mismatch: {key}')
            if not equivalent(replay['final_state'], row['final_state'], 1e-8):
                raise ValueError('Frame replay mismatch')
            frames = replay['frames']
            chosen = [frames[0]]
            for n, f in enumerate(frames[1:], 1):
                if n % 4 == 0 or f['call_index'] != frames[n-1]['call_index'] or n == len(frames)-1:
                    chosen.append(f)
            compact = [[round(f['time'], 6), *[round(x, 6) for x in f['position']],
                        *[round(x, 6) for x in f['velocity']], f['docked'], f['call_index']] for f in chosen]
            source = row['completion'].strip()
            offset = 1 if source.startswith('```') else 0
            try:
                task.parse_program(source)
                parsed = '\n'.join(source.splitlines()[1:-1]) if offset else source
                lines = [node.lineno-1+offset for node in ast.parse(parsed).body]
            except (SyntaxError, ValueError, TypeError, RecursionError):
                lines = []
            case[label] = {'code': row['completion'], 'lines': lines, 'frames': compact,
                           'success': row['success'], 'error': row['error'], 'reward': row['reward']}
        cases.append(case)
    template = Path(__file__).with_name('cap_rl_mujoco_replay_template.html').read_text()
    if split == 'validation':
        template = template.replace('Held-out episode', f'Validation episode (update {validation_step})')
    payload = json.dumps(cases, separators=(',', ':')).replace('<', '\\u003c')
    fragment = template.replace('__MUJOCO_DATA__', payload)
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError('Inline replay exceeds size limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)
    print(json.dumps({'cases': len(cases), 'bytes': len(fragment.encode()), 'out': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--validation-step', type=int)
    args = parser.parse_args()
    build(args.directory, args.out, args.validation_step)
