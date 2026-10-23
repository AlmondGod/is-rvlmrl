"""Inventory physical executions without double-counting copied cache records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def inventory(roots):
    directories = set()
    for root in roots:
        directories.update(p.parent.resolve() for p in Path(root).rglob('program.py')
                           if (p.parent/'provenance.json').is_file())
    completed, incomplete, nonphysical = {}, [], []
    for directory in sorted(directories):
        result_path = directory/'result.json'
        if not result_path.exists():
            incomplete.append(dict(directory=str(directory), scene_saved=(directory/'scene.xml').exists()))
            continue
        payload = result_path.read_bytes()
        result = json.loads(payload)
        if not result.get('physics_started'):
            nonphysical.append(str(directory))
            continue
        certificate = directory/'certificate.json'
        if certificate.exists():
            for name, expected in json.loads(certificate.read_text())['files'].items():
                actual = hashlib.sha256((directory/name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Certificate mismatch: {directory/name}')
        # The result includes measured wall_s. Published cache copies retain
        # identical bytes; independent replay executions have separate results.
        digest = hashlib.sha256(payload).hexdigest()
        manifest = json.loads((directory/'provenance.json').read_text())
        entry = completed.setdefault(digest, dict(result_sha256=digest, seed=result['seed'],
            success=result['success'], error=result.get('error'), wall_s=result.get('wall_s'),
            simulated_s=result.get('simulated_s'), platform=manifest['platform'], copies=[]))
        entry['copies'].append(str(directory))
    executions = list(completed.values())
    return dict(scope='Physical execution records across explicit roots; includes scripted diagnostics if supplied. This is not the number of on-policy samples.',
        deduplication='Exact result-file SHA256, including measured wall_s; collapses published cache copies, not separate deterministic replays.',
        completed_physical_executions=len(executions),
        copied_records=sum(len(e['copies'])-1 for e in executions),
        incomplete_records=incomplete, nonphysical_records=nonphysical,
        platforms=dict(Counter(e['platform'] for e in executions)),
        summed_execution_wall_seconds=sum(e['wall_s'] or 0 for e in executions),
        timing_warning='Execution durations overlap under parallelism; their sum is NOT allocated GPU hours or billed time.',
        executions=executions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('executions','incomplete_records','nonphysical_records')}))


if __name__ == '__main__':
    main()
