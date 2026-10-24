"""Success-only GRPO reward over the frozen full-assembly environment.

The physical task, dense diagnostic reward and certified trajectories are
unchanged. This external optimizer adapter records its exact source in each
run, so physics caches remain reusable without rewriting their provenance.
No demonstrations, program repair, or held-out-derived targets are supplied.
"""
import hashlib
import json
from pathlib import Path


def strict_results(results):
    return [dict(r, dense_reward=r.get('dense_reward', r['reward']), reward=float(r['success'])) for r in results]


def required_call_failure(text):
    """A necessary-condition proof of zero strict reward, never of success.

    The frozen executor requires four distinct corner visits before feeding,
    four released members before those visits, and positioning before release.
    Flipping requires a released sheet and all eight distinct nail visits.
    The evaluator requires completed flip AND finish. Missing any dependency
    therefore makes strict success impossible, regardless of the trajectory.
    Order/geometry/forces are deliberately left to the physical executor.
    """
    from experiments.cap_rl.assembly import parse_program
    try:
        calls = set(parse_program(text))
    except (SyntaxError, ValueError):
        return None  # The ordinary parser records syntax errors.
    required = {(name, i) for name in ('position_member','release_member','nail_corner','nail_panel')
                for i in range(4)}
    required.update((name, None) for name in ('park_grippers','feed_panel','position_panel',
                                            'release_panel','flip_structure','finish'))
    missing = sorted(required-calls)
    return dict(format='strict_necessary_calls_v1',missing=[list(x) for x in missing]) if missing else None


def install_execution_filter(task):
    """Guarded strict-only rejection; unmeasured dense reward stays unknown."""
    if getattr(task.execute_batch, '_strict_necessary_calls', False):
        return
    from experiments.cap_rl.assembly import ROOT
    expected = {
        'experiments/cap_rl/assembly.py': '9b84e2156320273004e7e90b818923e84766bdf128295cc371a58e2bd0acea2d',
        'experiments/torso_assembly/live.py': 'cf35af630c8ff75e920c49d95ed26b25fb77cac2d577ab09bb0e11db76b09499',
        'experiments/assembly_benchmark/executor.py': 'b85b412aebff6d779b216b891c7ef01789869ba8c697097df848636e72b2cd12',
        'experiments/assembly_benchmark/evaluation.py': '3758bb1ecef1754a8806049fb5797e993752c3777ae24329530ca1d27d8a578d'}
    for name,digest in expected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Revalidate strict necessary-call proof for changed source: '+name)
    original = task.execute_batch
    def execute_batch(layouts, texts):
        if len(layouts)!=len(texts):raise ValueError('Layout/program count mismatch')
        proofs=[required_call_failure(text) for text in texts]
        indices=[i for i,proof in enumerate(proofs) if proof is None]
        measured=iter(original([layouts[i] for i in indices],[texts[i] for i in indices]))
        return [next(measured) if proof is None else dict(reward=0.,success=False,valid=True,
            error='strict_missing_required_calls',physics_started=False,dense_reward=None,
            dense_reward_measured=False,static_rejection=proof) for proof in proofs]
    execute_batch._strict_necessary_calls=True
    task.execute_batch=execute_batch


def main():
    from experiments.cap_rl import train as trainer
    from experiments.cap_rl import task_assembly
    install_execution_filter(task_assembly)
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    original_rollout, original_train, original_test, original_dump = (
        trainer.rollout, trainer.train, trainer.test, trainer.dump)

    def rollout(*args, **kwargs):
        # The original trainer writes through its module-global rollout name,
        # which now points to this wrapper. Capture that fresh timing first.
        rollout.last_timing = {}
        sequences, attention, mask, texts, results = original_rollout(*args, **kwargs)
        rollout.last_timing = dict(rollout.last_timing or getattr(original_rollout, 'last_timing', {}))
        return sequences, attention, mask, texts, strict_results(results)

    def train(args):
        if args.task != 'assembly_full':
            raise ValueError('This optimizer adapter is restricted to assembly_full')
        args.optimization_reward = 'strict_success'
        args.static_strict_rejection = 'strict_necessary_calls_v1'
        args.optimization_adapter_sha256 = digest
        original_train(args)

    def test(args):
        metadata = json.loads((Path(args.out)/'metadata.json').read_text())
        if (metadata.get('optimization_reward') != 'strict_success' or
                metadata.get('optimization_adapter_sha256') != digest):
            raise ValueError('Use the exact recorded optimizer adapter for this run')
        original_test(args)

    def dump(path, value):
        original_dump(path, value)
        if Path(path).name == 'metadata.json':
            directory = Path(path).parent/'optimization_source'
            directory.mkdir(exist_ok=False)
            (directory/'train_assembly_strict.py').write_bytes(source)

    trainer.rollout, trainer.train, trainer.test, trainer.dump = rollout, train, test, dump
    trainer.main()


if __name__ == '__main__':
    main()
