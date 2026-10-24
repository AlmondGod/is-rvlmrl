"""Packaging regression fixtures, not physical task evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.cap_rl_assembly_records import input_receipt
from analysis.package_cap_assembly import scored_rows, verify_audit_receipts


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


if __name__ == '__main__': unittest.main()
