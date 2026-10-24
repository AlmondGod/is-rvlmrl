import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from analysis.train_assembly_strict import strict_results, main, required_call_failure, install_execution_filter


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


class NecessaryCallTests(unittest.TestCase):
    def complete(self):
        return [f'{name}({i})' for name in ('position_member','release_member','nail_corner','nail_panel')
                for i in range(4)] + [name+'()' for name in
                ('park_grippers','feed_panel','position_panel','release_panel','flip_structure','finish')]

    def test_each_necessary_call_and_syntax(self):
        calls=self.complete()
        self.assertIsNone(required_call_failure('\n'.join(calls)))
        self.assertIsNone(required_call_failure('\n'.join(list(reversed(calls[:-1]))+calls[-1:])))
        for i in range(len(calls)):
            with self.subTest(call=calls[i]):
                proof=required_call_failure('\n'.join(calls[:i]+calls[i+1:]))
                self.assertEqual(len(proof['missing']),1)
        self.assertIsNone(required_call_failure('not python!!!'))

    def test_batch_order_unknown_dense_and_idempotence(self):
        complete='\n'.join(self.complete()); observed=[]
        def execute(layouts,texts):
            observed.append((layouts,texts))
            return [dict(layout=x,reward=.8,success=False) for x in layouts]
        task=SimpleNamespace(execute_batch=execute)
        install_execution_filter(task); install_execution_filter(task)
        rows=task.execute_batch([1,2,3],['finish()',complete,'stop()'])
        self.assertEqual(observed,[([2],[complete])])
        self.assertEqual(rows[1]['layout'],2)
        self.assertIsNone(rows[0]['dense_reward'])
        self.assertIsNone(strict_results(rows)[0]['dense_reward'])
        self.assertFalse(rows[2]['physics_started'])
        with self.assertRaises(ValueError):task.execute_batch([1],[])

    def test_changed_dependencies_fail_closed(self):
        task=SimpleNamespace(execute_batch=lambda a,b:[])
        with patch('analysis.train_assembly_strict.hashlib.sha256') as digest:
            digest.return_value.hexdigest.return_value='changed'
            with self.assertRaisesRegex(RuntimeError,'Revalidate'):install_execution_filter(task)


if __name__=='__main__':unittest.main()
