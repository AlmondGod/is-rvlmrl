"""Compare complete recordings from serial and process-parallel rollout prefixes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import time


def worker_init(backend, threads):
    import cv2
    from experiments.torso_assembly.acceleration import configure
    cv2.setNumThreads(threads)
    configure(backend)


def execute(job):
    from experiments.cap_rl.assembly import run
    seed, text, out, seconds = job
    with Path(out).with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log):
        result = run(seed, text, out, seconds=seconds,
                     support_profile='contact_and_nails', controller_profile='southwest_approach')
    return dict(seed=seed, wall_s=result['wall_s'], simulated_s=result['simulated_s'],
                error=result['error'], success=result['success'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--program', required=True)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 4, 8])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--seed-start', type=int, default=51001)
    p.add_argument('--layouts', type=int, default=8)
    p.add_argument('--backend', choices=['original', 'vectorized'], default='vectorized')
    a = p.parse_args()
    if min(*a.workers, a.threads, a.seconds, a.layouts) <= 0 or a.workers[0] != 1:
        p.error('Positive sizes required; the first worker count must be 1')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = str(a.threads)
    root = Path(a.out).resolve(); root.mkdir(parents=True, exist_ok=False)
    text = Path(a.program).read_text()
    from analysis.compare_assembly_speed import compare
    rows = []
    for workers in a.workers:
        destination = root/f'workers-{workers}'; destination.mkdir()
        jobs = [(seed, text, str(destination/str(seed)), a.seconds)
                for seed in range(a.seed_start, a.seed_start+a.layouts)]
        t = time.monotonic()
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_init, initargs=(a.backend, a.threads)) as pool:
            results = list(pool.map(execute, jobs))
        elapsed = time.monotonic()-t
        comparisons = []
        if workers != 1:
            for seed, *_ in jobs:
                item = compare(root/'workers-1'/str(seed), destination/str(seed))
                if not item['exact']:
                    raise RuntimeError('Parallel execution changed the recorded trajectory')
                comparisons.append(item)
        row = dict(workers=workers, elapsed_s=elapsed,
                   simulated_seconds_per_wall_second=sum(r['simulated_s'] for r in results)/elapsed,
                   result_rows=results, comparisons=comparisons)
        rows.append(row)
        (root/'summary.json').write_text(json.dumps(dict(configuration=vars(a), rows=rows,
            scope='Simulation prefixes, not task success. Includes pool startup; each world renders on GPU and integrates native MuJoCo on CPU.'), indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('result_rows','comparisons')}), flush=True)


if __name__ == '__main__':
    main()
