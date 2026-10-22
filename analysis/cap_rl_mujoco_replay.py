"""Build an inline replay of all held-out MuJoCo trajectories and exact code."""
import argparse
import ast
import json
from pathlib import Path

from experiments.cap_rl.audit import audit, equivalent
from experiments.cap_rl import task_mujoco as task


def build(directory, output):
    directory, output = Path(directory), Path(output)
    audit(directory)
    report = json.loads((directory/'test_comparison.json').read_text())
    cases = []
    for i, layout in enumerate(task.datasets()['test']):
        case = {'id': layout.key, 'start': task.meters(layout.start),
                'first': task.meters(layout.first), 'second': task.meters(layout.second)}
        for label in ('before', 'after'):
            row = report[label][i]
            replay = task.execute(layout, row['completion'], record_frames=True)
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
    args = parser.parse_args()
    build(args.directory, args.out)
