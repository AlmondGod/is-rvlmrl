"""Independently check the exact replay recovery receipt and preserved lineage.

This adds checks for recovery21 to the ordinary physical-record audit. It does
not replace that audit, regenerate physics, or infer unseen optimizer state.
"""
import argparse
import hashlib
import json
from pathlib import Path

RECOVERY_SOURCE_SHA256 = '263e5669129c0643247e9b27d0bc814ed4cbc530a161dd843630e73f418d4d32'
FIELDS = ('layout_id', 'prompt', 'completion', 'reward', 'success', 'evidence_id', 'result_sha256')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit(root):
    root = Path(root)
    metadata = read(root/'metadata.json')
    reconstruction = metadata['reconstruction']
    old = root/'interrupted_online_attempt'
    required = {'metadata.json', 'process_start.json', 'training.jsonl',
                'rollouts.jsonl', 'interruption.json', 'validation_baseline.json'}
    if set(reconstruction['interrupted_attempt_sha256']) != required:
        raise ValueError('Incomplete interrupted attempt receipt')
    for name, digest in reconstruction['interrupted_attempt_sha256'].items():
        if sha(old/name) != digest:
            raise ValueError('Interrupted source record changed: '+name)
    if not (sha(root/'reconstruction_source.py') == reconstruction['source_sha256'] == RECOVERY_SOURCE_SHA256):
        raise ValueError('Unreviewed reconstruction implementation')
    old_metadata = read(old/'metadata.json')
    for name in ('model', 'model_revision', 'optimizer_algorithm', 'training_layout_schedule',
                 'seed', 'lr', 'beta', 'sampling_temperature', 'microbatch', 'update_epochs',
                 'group_size', 'support_profile', 'controller_profile', 'assembly_backend',
                 'splits', 'warm_start_weights_sha256', 'reference_policy', 'online_source_sha256'):
        if metadata[name] != old_metadata[name]:
            raise ValueError('Recovery changed protocol: '+name)
    before = records(old/'rollouts.jsonl')
    after = records(root/'rollouts.jsonl')[:16]
    if len(before) != 16 or len(after) != 16:
        raise ValueError('Incomplete first rollout batch')
    for a, b in zip(before, after):
        if a['step'] != 1 or b['step'] != 1 or any(a[k] != b[k] for k in FIELDS):
            raise ValueError('First batch differs from interrupted run')
    prior = records(old/'training.jsonl')
    current = records(root/'training.jsonl')
    if len(prior) != 1 or prior[0]['optimizer_steps_total'] != 4 or current[0]['optimizer_steps_total'] != 4:
        raise ValueError('Expected replay of four initial updates')
    # Timing/cache fields legitimately differ when physical executions are reused.
    for name in ('loss', 'kl', 'grad_norm', 'epoch_metrics', 'reward', 'active_groups'):
        if prior[0][name] != current[0][name]:
            raise ValueError('First update numerical metrics differ: '+name)
    expected = read(old/'interruption.json')['checkpoint_sha256']
    actual = sha(root/'checkpoints/step-0001/adapter_model.safetensors')
    report = read(root/'reconstruction_verification.json')
    if not (actual == expected == reconstruction['checkpoint_sha256'] ==
            report['expected_weights_sha256'] == report['actual_weights_sha256']):
        raise ValueError('Reconstructed checkpoint differs')
    if (report['passes'] is not True or report['first_batch_records_equal'] is not True or
            report['source_sha256'] != RECOVERY_SOURCE_SHA256 or reconstruction['prior_updates'] != 4):
        raise ValueError('Invalid reconstruction receipt')
    return dict(passes=True, checkpoint_sha256=actual, first_batch_records=16,
                replayed_optimizer_updates=4, source_sha256=sha(__file__),
                inputs_sha256={name:sha(root/name) for name in (
                    'metadata.json', 'reconstruction_source.py', 'reconstruction_verification.json',
                    'rollouts.jsonl', 'training.jsonl')},
                interrupted_attempt_sha256=reconstruction['interrupted_attempt_sha256'],
                scope='Exact first-batch, four-update and checkpoint replay with unchanged protocol. '
                      'Prior interrupted time remains separate; no held-out or physical-replay claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = audit(args.run)
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
