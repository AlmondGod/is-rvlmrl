"""Compare complete continuous runs; never treat a truncated prefix as success."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(before, after):
    before, after = Path(before), Path(after)
    a, b = [json.loads((p/'result.json').read_text()) for p in (before, after)]
    da, db = [dict(d) for d in (a, b)]
    da.pop('wall_s'); db.pop('wall_s')
    differing_keys = [k for k in sorted(da.keys() | db.keys()) if da.get(k) != db.get(k)]
    with np.load(before/'states.npz') as x, np.load(after/'states.npz') as y:
        assert set(x.files) == set(y.files), 'State recording schema changed'
        arrays = {}
        for key in x.files:
            same_shape = x[key].shape == y[key].shape
            arrays[key] = dict(shape_before=list(x[key].shape), shape_after=list(y[key].shape),
                               exact=same_shape and bool(np.array_equal(x[key], y[key])),
                               max_abs_difference=float(np.max(np.abs(x[key]-y[key]))) if same_shape else None)
    exact = not differing_keys and all(v['exact'] for v in arrays.values())
    result = dict(seed=a['seed'], same_seed=a['seed']==b['seed'], exact=exact,
                  result_keys_differing_except_wall_s=differing_keys, arrays=arrays,
                  baseline_success=a['success'], optimized_success=b['success'],
                  baseline_error=a['error'], optimized_error=b['error'],
                  simulated_s=a['simulated_s'], baseline_wall_s=a['wall_s'], optimized_wall_s=b['wall_s'],
                  speedup=a['wall_s']/b['wall_s'],
                  timing_caveat='These exploratory full runs may overlap on one host; measure isolated throughput separately.',
                  result_hashes=[hashlib.sha256((p/'result.json').read_bytes()).hexdigest() for p in (before, after)],
                  state_hashes=[hashlib.sha256((p/'states.npz').read_bytes()).hexdigest() for p in (before, after)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', required=True); p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = compare(a.before, a.after)
    Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['exact'] or not result['same_seed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
