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
    return [dict(r, dense_reward=r['reward'], reward=float(r['success'])) for r in results]


def main():
    from experiments.cap_rl import train as trainer
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    original_rollout, original_train, original_test, original_dump = (
        trainer.rollout, trainer.train, trainer.test, trainer.dump)

    def rollout(*args, **kwargs):
        sequences, attention, mask, texts, results = original_rollout(*args, **kwargs)
        return sequences, attention, mask, texts, strict_results(results)

    def train(args):
        if args.task != 'assembly_full':
            raise ValueError('This optimizer adapter is restricted to assembly_full')
        args.optimization_reward = 'strict_success'
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
