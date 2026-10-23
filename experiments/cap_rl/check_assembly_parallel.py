"""Verify independent parallel physics against sequential physical execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np
from . import task_assembly as task
from .audit_assembly import compare
from .assembly import SUPPORT_PROFILES, CONTROLLER_PROFILES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--support-profile', choices=SUPPORT_PROFILES, default='contact_only')
    p.add_argument('--controller-profile', choices=CONTROLLER_PROFILES, default='original')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    task.configure(args.support_profile, args.controller_profile)
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
    os.environ['CAP_RL_ASSEMBLY_CACHE'] = str(out / 'contended')
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'),
            initializer=task._worker_init,
            initargs=(task.EXPECTED_PROVENANCE, args.support_profile, args.controller_profile)) as pool:
        # Submit independently: this bypasses execute_batch's AST deduplication.
        futures = [pool.submit(task.execute, layouts[0], programs[0]) for _ in range(2)]
        contended = [f.result() for f in futures]
    assert sorted(r['cache_hit'] for r in contended) == [False, True]
    assert contended[0]['evidence_id'] == contended[1]['evidence_id']
    pc, ps = Path(contended[0]['evidence']), Path(sequential[0]['evidence'])
    compare(json.loads((pc/'result.json').read_text()), json.loads((ps/'result.json').read_text()))
    np.testing.assert_allclose(np.load(pc/'states.npz')['rows'],
                               np.load(ps/'states.npz')['rows'], rtol=0, atol=1e-7)
    result = dict(passes=True, cases=3, unique_programs=2, one_contact_grasp=True,
                  support_profile=args.support_profile, controller_profile=args.controller_profile,
                  independent_process_cache_contention_verified=True,
                  absolute_tolerance=1e-7, exact_ast_duplicate_verified=True)
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
