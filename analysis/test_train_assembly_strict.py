import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from analysis.train_assembly_strict import strict_results, main


class StrictRewardTests(unittest.TestCase):
    def test_partial_progress_cannot_beat_another_failed_program(self):
        rows=[dict(success=False,reward=r,valid=True) for r in (0.,.15,.8)]
        original=copy.deepcopy(rows)
        transformed=strict_results(rows)
        self.assertEqual([r['reward'] for r in transformed],[0.,0.,0.])
        self.assertEqual([r['dense_reward'] for r in transformed],[0.,.15,.8])
        self.assertEqual(rows,original)

    def test_only_measured_strict_success_receives_credit(self):
        rows=[dict(success=True,reward=.925,valid=True,error=None),
              dict(success=False,reward=0.,valid=False,error='truncated')]
        transformed=strict_results(rows)
        self.assertEqual([r['reward'] for r in transformed],[1.,0.])
        self.assertEqual(transformed[0]['dense_reward'],.925)
        self.assertEqual(transformed[1]['error'],'truncated')

    def test_wrapper_records_exact_source_and_leaves_physics_result_unchanged(self):
        import experiments.cap_rl
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'run'
            args=SimpleNamespace(task='assembly_full',out=str(out))
            raw=[dict(success=False,reward=.8,valid=True)]
            observed=[]
            fake=SimpleNamespace()
            fake.rollout=lambda: ('sequences','attention','mask','texts',raw)
            fake.dump=lambda path,value: Path(path).write_text(json.dumps(value))
            def original_train(arguments):
                out.mkdir()
                fake.dump(out/'metadata.json',vars(arguments))
                observed.append(fake.rollout())
            fake.train=original_train
            fake.test=lambda arguments: observed.append('tested')
            fake.main=lambda: (fake.train(args),fake.test(args))
            with patch.object(experiments.cap_rl,'train',fake,create=True):main()
            metadata=json.loads((out/'metadata.json').read_text())
            source=out/'optimization_source/train_assembly_strict.py'
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             metadata['optimization_adapter_sha256'])
            self.assertEqual(metadata['optimization_reward'],'strict_success')
            self.assertEqual(observed[0][:4],('sequences','attention','mask','texts'))
            self.assertEqual(observed[0][4][0]['reward'],0.)
            self.assertEqual(observed[0][4][0]['dense_reward'],.8)
            self.assertEqual(raw[0]['reward'],.8)
            self.assertEqual(observed[1],'tested')


if __name__=='__main__':unittest.main()
