"""Profile continuous development episodes without changing their task or reward."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=41002)
    p.add_argument('--seconds', type=float, default=1200.)
    p.add_argument('--optimized', action='store_true')
    p.add_argument('--native-pixels', action='store_true', help='Exact compiled color predicates and sparse-mask scan; requires --optimized')
    a = p.parse_args()
    if a.native_pixels and not a.optimized: p.error('--native-pixels requires --optimized')
    from experiments.cap_rl.assembly import run
    from experiments.torso_assembly.environment import Env
    from experiments.torso_assembly.fasteners import NailBank
    from experiments.torso_assembly.frame_vision import MultiVision
    from experiments.wall_panel.perception import Vision
    from experiments.assembly_benchmark.executor import Executor
    from experiments.cap_rl.support_measurement import SupportEvaluation
    import mujoco
    if a.optimized:
        from experiments.torso_assembly.acceleration import configure
        configure('vectorized_pixels' if a.native_pixels else 'vectorized')
    stats = {}
    started = time.perf_counter()
    last_write = started
    report = Path(a.out).with_suffix('.profile.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(seed=a.seed, optimized=a.optimized, python=platform.python_version(),
                    native_pixels=a.native_pixels,
                    cpu_count=os.cpu_count(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    timing_scope='Inclusive nested wall times; do not sum parent and child categories.')
    sources = report.with_suffix('.sources')
    sources.mkdir(exist_ok=False)
    source_files = [Path(__file__)]
    if a.optimized:
        source_files.append(Path(__file__).with_name('assembly_speed.py'))
    metadata['execution_sources'] = {}
    for source in source_files:
        shutil.copy2(source, sources/source.name)
        metadata['execution_sources'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()

    def save(complete=False):
        temp = report.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(metadata, complete=complete,
                        elapsed_s=time.perf_counter()-started, timings=stats), indent=2)+'\n')
        temp.replace(report)

    def wrap(cls, name, label):
        original = getattr(cls, name)
        stats[label] = dict(calls=0, seconds=0.)
        @functools.wraps(original)
        def measured(self, *args, **kwargs):
            nonlocal last_write
            t = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                stats[label]['seconds'] += time.perf_counter()-t
                stats[label]['calls'] += 1
                if label == 'environment_step' and time.perf_counter()-last_write > 30:
                    metadata['simulated_s'] = float(self.d.time)
                    save()
                    last_write = time.perf_counter()
        setattr(cls, name, measured)

    for cls, name, label in [(Env, 'step', 'environment_step'), (Env, 'images', 'camera_render'),
                            (Env, 'ik', 'inverse_kinematics'), (Env, 'panel_metrics', 'metrics'),
                            (NailBank, 'apply', 'nail_forces'), (MultiVision, 'update', 'frame_vision'),
                            (Vision, 'update', 'panel_vision'), (Executor, 'command', 'controller'),
                            (SupportEvaluation, 'update', 'acceptance'),
                            (mujoco, 'mj_step', 'physics_kernel')]:
        wrap(cls, name, label)
    result = run(a.seed, Path(a.program).read_text(), a.out, seconds=a.seconds, progress=True,
                 support_profile='contact_and_nails', controller_profile='southwest_approach')
    metadata['result_sha256'] = hashlib.sha256((Path(a.out)/'result.json').read_bytes()).hexdigest()
    metadata['simulated_s'] = result['simulated_s']
    save(complete=True)
    print(json.dumps({k: result[k] for k in ['success', 'error', 'simulated_s', 'wall_s']}), flush=True)


if __name__ == '__main__':
    main()
