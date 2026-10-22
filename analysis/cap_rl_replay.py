"""Build an audited two-object episode replay fragment from saved model outputs.

Run as: python -m analysis.cap_rl_replay RUN_DIRECTORY --out replay.html
The fragment is self-contained and makes no network requests.
"""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path

from experiments.cap_rl.audit import audit
from experiments.cap_rl.task_ordered import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.directory)
    metadata = json.loads((args.directory/'metadata.json').read_text())
    if metadata.get('task') != 'ordered_two':
        parser.error('This replay supports the ordered_two task')
    report = json.loads((args.directory/'test_comparison.json').read_text())
    selection = json.loads((args.directory/'selection.json').read_text())
    episodes = []
    for layout, before, after in zip(datasets()['test'], report['before'], report['after']):
        episode = {'id': layout.key, **asdict(layout)}
        for side, row in [('before', before), ('after', after)]:
            lines = row['completion'].strip().splitlines()
            offset = 1 if lines and lines[0] in ('```', '```python') else 0
            source = '\n'.join(lines[1:-1] if offset else lines)
            try:
                call_lines = [node.lineno+offset for node in ast.parse(source).body]
            except (SyntaxError, ValueError, RecursionError):
                call_lines = []
            episode[side] = {k: row[k] for k in ('success', 'reward', 'error', 'trace')}
            episode[side].update(lines=lines, callLines=call_lines)
        episodes.append(episode)
    template = Path(__file__).with_name('cap_rl_replay_template.html').read_text()
    payload = json.dumps(episodes, separators=(',', ':')).replace('<', '\\u003c')
    result = template.replace('__EPISODE_DATA__', payload).replace('__BEST_STEP__', str(selection['best_step']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(json.dumps({'episodes': len(episodes), 'bytes': len(result.encode()), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
