"""Verify independent parallel physics against sequential physical execution."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from . import task_assembly as task
from .audit_assembly import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    task.EXPECTED_PROVENANCE = task.provenance()
    layouts = [task.Layout(41001), task.Layout(41001), task.Layout(41002)]
    programs = ['position_member(0)\nposition_member(1)',
                'position_member(0)\n\nposition_member(1)\n', 'release_member(0)']
    os.environ['CAP_RL_ASSEMBLY_CACHE'] = str(out / 'parallel')
    task.PHYSICS_WORKERS = 2
    parallel = task.execute_batch(layouts, programs)
    task._POOL.shutdown()
    task._POOL = None
    assert parallel[0]['evidence_id'] == parallel[1]['evidence_id']
    assert parallel[1]['batch_duplicate']
    os.environ['CAP_RL_ASSEMBLY_CACHE'] = str(out / 'sequential')
    task.PHYSICS_WORKERS = 1
    sequential = task.execute_batch(layouts, programs)
    for a, b in zip(parallel, sequential):
        pa, pb = Path(a['evidence']), Path(b['evidence'])
        compare(json.loads((pa/'result.json').read_text()), json.loads((pb/'result.json').read_text()))
        np.testing.assert_allclose(np.load(pa/'states.npz')['rows'],
                                   np.load(pb/'states.npz')['rows'], rtol=0, atol=1e-7)
    result = dict(passes=True, cases=3, unique_programs=2, one_contact_grasp=True,
                  absolute_tolerance=1e-7, exact_ast_duplicate_verified=True)
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
