"""Fresh physical replay of a full-assembly cached episode, without cache reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .assembly import provenance, run


def compare(expected, actual, path='result'):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(path + ': keys differ')
        for key in expected:
            if key != 'wall_s':
                compare(expected[key], actual[key], path + '.' + key)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(path + ': length differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f'{path}[{i}]')
    elif type(expected) is float:
        if not np.isfinite(expected) or not np.isfinite(actual) or not np.isclose(expected, actual, rtol=0, atol=1e-7):
            raise AssertionError(f'{path}: {expected!r} != {actual!r}')
    elif type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f'{path}: {expected!r} != {actual!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    evidence = Path(args.evidence)
    recorded_provenance = json.loads((evidence / 'provenance.json').read_text())
    from experiments.torso_assembly.acceleration import configure
    configure(recorded_provenance.get('execution_backend', 'original'))
    profiles = recorded_provenance.get('profiles', {})
    if recorded_provenance != provenance(**profiles):
        raise RuntimeError('Replay must use the recorded source, assets and runtime')
    certificate = json.loads((evidence / 'certificate.json').read_text())
    for name, sha in certificate['files'].items():
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Evidence changed: ' + name)
    expected = json.loads((evidence / 'result.json').read_text())
    actual = run(expected['seed'], (evidence / 'program.py').read_text(), args.out,
                 video=args.video, progress=True, **profiles)
    compare(expected, actual)
    old = np.load(evidence / 'states.npz')
    new = np.load(Path(args.out) / 'states.npz')
    np.testing.assert_allclose(old['rows'], new['rows'], rtol=0, atol=1e-7)
    result = dict(passes=True, source=str(evidence.resolve()),
                  result_sha256=certificate['files']['result.json'],
                  state_samples=len(new['rows']), absolute_tolerance=1e-7)
    (Path(args.out) / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
