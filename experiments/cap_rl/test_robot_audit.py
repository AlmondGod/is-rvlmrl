"""Ensure cached robot replays still check every evidence row for tampering."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
from analysis.cap_rl_audit_records import audit_records
from . import task_robot as task
from .robot_lift import asset_hashes


class RobotAuditTests(unittest.TestCase):
    def test_cached_duplicate_is_not_trusted_without_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            source = out / 'source'; source.mkdir()
            files = list(Path(task.__file__).parent.glob('*.py'))
            for p in files:
                (source/p.name).write_bytes(p.read_bytes())
            splits = task.datasets()
            metadata = {'task': 'robot_lift', 'group_size': 8, 'prompts_per_step': 2,
                        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                        'robot_assets_sha256': asset_hashes(),
                        'physics_runtime': {'mujoco': mujoco.__version__},
                        'splits': {k: [l.key for l in v] for k, v in splits.items()}}
            (out/'metadata.json').write_text(json.dumps(metadata))
            def row(l):
                return {'layout_id': l.key, 'prompt': task.prompt(l), 'completion': 'import os',
                        'truncated': False, **task.execute(l, 'import os')}
            validation = [row(l) for l in splits['validation']]
            (out/'validation_baseline.json').write_text(json.dumps({'rows': validation, 'metrics': {
                'n': 64, 'successes': 0, 'success_rate': 0., 'valid_rate': 0., 'mean_reward': 0.}}))
            rows = [{'step': 1, **row(l)} for l in splits['train'][:2] for _ in range(8)]
            path = out/'rollouts.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            (out/'training.jsonl').write_text(json.dumps({'step': 1, 'reward': 0., 'success_rate': 0.})+'\n')
            result = audit_records(out)
            self.assertEqual(result['cached_record_checks'], 14)
            self.assertEqual(result['unique_programs_executed'], 66)
            self.assertEqual(result['training_records_verified'], 16)
            self.assertEqual(audit_records(out, workers=2), result)
            rows[1]['success'] = True  # same cached program, tampered outcome
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaisesRegex(ValueError, 'Replay mismatch'):
                audit_records(out)
            with self.assertRaisesRegex(ValueError, 'Replay mismatch'):
                audit_records(out, workers=2)


if __name__ == '__main__':
    unittest.main()
