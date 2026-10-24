"""Packaging regression fixtures, not physical task evidence."""
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

from analysis.cap_rl_assembly_records import input_receipt
from analysis.package_cap_assembly import add_payload, scored_rows, verify_audit_receipts


class PackagingTests(unittest.TestCase):
    def fixture(self, root):
        run, replay = root/'run', root/'replay'
        run.mkdir(); replay.mkdir()
        def save(name, value): (run/name).write_text(json.dumps(value))
        save('metadata.json', {'fixture': True})
        save('selection.json', {'best_step': 0})
        (run/'training.jsonl').write_text('{}\n')
        (run/'rollouts.jsonl').write_text('{"evidence_id":"train"}\n')
        save('validation_baseline.json', {'rows':[{'evidence_id':'validation'}]})
        save('test_comparison.json', {'before':[{'evidence_id':'before'}],
                                      'after':[{'evidence_id':'after'}]})
        save('raw_test_baseline.json', {'rows':[{'evidence_id':'raw'}]})
        for split in ('validation', 'test'):
            save(f'unconstrained_{split}_comparison.json',
                 {'before':[{'evidence_id':split+'-free-before'}],
                  'after':[{'evidence_id':split+'-free-after'}]})
        inputs = input_receipt(run)
        save('record_audit.json', {'passes':True,'input_files_sha256':inputs})
        names = ('metadata.json','selection.json','test_comparison.json',
                 'raw_test_baseline.json','unconstrained_test_comparison.json')
        save('generation_audit.json', {'passes':True,
             'evaluation_inputs_sha256':{name:inputs[name] for name in names}})
        (replay/'audit.json').write_text(json.dumps({'passes':True,
            'test_comparison_sha256':inputs['test_comparison.json']}))
        return run, replay

    def test_all_decoder_and_raw_evidence_is_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, replay = self.fixture(Path(directory))
            verify_audit_receipts(run, replay)
            self.assertEqual({r['evidence_id'] for r in scored_rows(run)},
                {'train','validation','before','after','raw','validation-free-before',
                 'validation-free-after','test-free-before','test-free-after'})

    def test_changed_or_new_reports_invalidate_audits(self):
        for name in ('unconstrained_test_comparison.json', 'selection.json', 'validation_0002.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run, replay = self.fixture(Path(directory))
                (run/name).write_text('{"changed":true}')
                with self.assertRaises(ValueError): verify_audit_receipts(run, replay)

    def test_incomplete_generation_or_replay_is_rejected(self):
        for target, key in [('generation_audit.json','evaluation_inputs_sha256'),
                            ('replay','test_comparison_sha256')]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                run, replay = self.fixture(Path(directory))
                path = replay/'audit.json' if target == 'replay' else run/target
                report = json.loads(path.read_text());report.pop(key)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError): verify_audit_receipts(run, replay)

    def test_new_or_changed_weights_invalidate_record_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            run,replay=self.fixture(Path(directory))
            (run/'best_adapter').mkdir()
            (run/'best_adapter/adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_audit_receipts(run,replay)

    def test_parent_evidence_and_receipts_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            run,replay=self.fixture(Path(directory))
            parent=run/'parent_run';parent.mkdir()
            (parent/'rollouts.jsonl').write_text('{"evidence_id":"parent-failure"}\n')
            (parent/'metadata.json').write_text('{"parent":true}')
            self.assertIn('parent-failure',{r['evidence_id'] for r in scored_rows(run)})
            self.assertIn('parent_run/metadata.json',input_receipt(run))
            with self.assertRaises(ValueError):verify_audit_receipts(run,replay)

    def test_deduplicated_payloads_keep_every_path_and_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b'
            a.write_bytes(b'weights fixture'*100);b.write_bytes(a.read_bytes())
            archive=root/'payload.tar';seen={}
            with tarfile.open(archive,'w') as stream:
                first=add_payload(stream,a,'run/initial/weights',seen)
                second=add_payload(stream,b,'run/parent/weights',seen)
            self.assertEqual(first['sha256'],second['sha256'])
            self.assertEqual(second['content_link'],'run/initial/weights')
            with tarfile.open(archive) as stream:
                self.assertTrue(stream.getmember('run/parent/weights').islnk())
                self.assertEqual(stream.extractfile('run/parent/weights').read(),a.read_bytes())
                stream.extractall(root/'extract',filter='data')
            self.assertEqual((root/'extract/run/parent/weights').read_bytes(),a.read_bytes())


if __name__ == '__main__': unittest.main()
