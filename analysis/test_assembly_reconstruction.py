import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import audit_assembly_reconstruction as recovery


class ReconstructionAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/'interrupted_online_attempt'
        self.old.mkdir()
        (self.root/'checkpoints/step-0001').mkdir(parents=True)
        self.weights = self.root/'checkpoints/step-0001/adapter_model.safetensors'
        self.weights.write_bytes(b'original-checkpoint')
        (self.root/'reconstruction_source.py').write_text('reviewed recovery source')
        digest = recovery.sha(self.root/'reconstruction_source.py')
        self.patch = patch.object(recovery, 'RECOVERY_SOURCE_SHA256', digest)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        meta = {name:'same' for name in ('model', 'model_revision', 'optimizer_algorithm',
            'training_layout_schedule', 'seed', 'lr', 'beta', 'sampling_temperature',
            'microbatch', 'update_epochs', 'group_size', 'support_profile', 'controller_profile',
            'assembly_backend', 'splits', 'warm_start_weights_sha256', 'reference_policy', 'online_source_sha256')}
        rows = [dict(step=1, **{k:f'{i}:{k}' for k in recovery.FIELDS}) for i in range(16)]
        metrics = dict(optimizer_steps_total=4, loss=.1, kl=.2, grad_norm=.3,
                       epoch_metrics=[.1,.2], reward=.5, active_groups=2)
        for directory in (self.old, self.root):
            (directory/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (directory/'training.jsonl').write_text(json.dumps(metrics)+'\n')
        self.dump(self.old/'metadata.json', meta)
        self.dump(self.old/'process_start.json', {})
        self.dump(self.old/'validation_baseline.json', {})
        self.dump(self.old/'interruption.json', dict(checkpoint_sha256=recovery.sha(self.weights)))
        meta['reconstruction'] = dict(source_sha256=digest, prior_updates=4,
            checkpoint_sha256=recovery.sha(self.weights),
            interrupted_attempt_sha256={p.name:recovery.sha(p) for p in self.old.iterdir()})
        self.dump(self.root/'metadata.json', meta)
        self.dump(self.root/'reconstruction_verification.json', dict(passes=True,
            first_batch_records_equal=True, source_sha256=digest,
            expected_weights_sha256=recovery.sha(self.weights), actual_weights_sha256=recovery.sha(self.weights)))

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value))

    def test_valid_and_later_updates_do_not_change_recovery_prefix(self):
        with (self.root/'rollouts.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=2))+'\n')
        self.assertTrue(recovery.audit(self.root)['passes'])

    def test_replaced_weights_rejected(self):
        self.weights.write_bytes(b'different-checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            recovery.audit(self.root)

    def test_changed_first_prompt_rejected(self):
        path = self.root/'rollouts.jsonl'
        rows = recovery.records(path)
        rows[0]['prompt'] = 'changed'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'First batch differs'):
            recovery.audit(self.root)

    def test_changed_protocol_and_archived_records_rejected(self):
        path = self.root/'metadata.json'
        meta = recovery.read(path)
        meta['lr'] = 'changed'
        self.dump(path, meta)
        with self.assertRaisesRegex(ValueError, 'changed protocol: lr'):
            recovery.audit(self.root)
        meta['lr'] = 'same'
        self.dump(path, meta)
        self.dump(self.old/'process_start.json', {'changed':True})
        with self.assertRaisesRegex(ValueError, 'source record changed'):
            recovery.audit(self.root)


if __name__ == '__main__':
    unittest.main()
