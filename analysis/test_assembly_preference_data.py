import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.assembly_preference_data import build
from experiments.cap_rl.assembly import parse_program
from experiments.cap_rl.task_assembly import Layout, prompt


class PreferenceDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.provenance = {'frozen': 'fixture'}
        self.meta = dict(task='assembly_full',assembly_split='varied_v2',optimization_reward='strict_success',
                        splits={'train':[Layout(51001).key], 'validation':[Layout(52001).key], 'test':[Layout(53001).key]})
        self.save('metadata.json',self.meta)
        self.save('assembly_provenance.json',self.provenance)
        self.rows = [self.sample('position_member(0)\nfinish()',True),
                     self.sample('position_member(1)\nfinish()',False)]
        self.flush()

    def tearDown(self):self.temp.cleanup()

    def save(self,name,obj):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj));return p

    def sample(self,code,success):
        identity=dict(seed=51001,calls=parse_program(code),provenance=self.provenance)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        directory=self.root/key;directory.mkdir()
        self.save(key+'/result.json',dict(seed=51001,success=success,passes=success))
        self.save(key+'/provenance.json',self.provenance)
        (directory/'program.py').write_text(code)
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        self.save(key+'/certificate.json',dict(identity_sha256=key,files=files))
        return dict(layout_id=Layout(51001).key,prompt=prompt(Layout(51001)),completion=code,
            reward=float(success),success=success,evidence=str(directory),evidence_id=key,
            result_sha256=files['result.json'],truncated=False,step=1)

    def flush(self):
        (self.root/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_deduplicate_and_pair_within_layout(self):
        self.rows+=self.rows[:];self.flush();data=build(self.root)
        self.assertEqual(data['source_rows'],4)
        self.assertEqual(len(data['pairs']),1)
        self.assertEqual(data['pairs'][0]['chosen'],self.rows[0]['completion'])
        self.assertEqual(data['pairs'][0]['rejected'],self.rows[1]['completion'])

    def test_reject_test_data(self):
        self.rows[0]['layout_id']=Layout(53001).key;self.flush()
        with self.assertRaisesRegex(ValueError,'Non-training'):build(self.root)

    def test_reject_altered_reward(self):
        self.rows[0]['reward']=.8;self.flush()
        with self.assertRaisesRegex(ValueError,'binary'):build(self.root)

    def test_reject_evidence_tampering(self):
        (Path(self.rows[0]['evidence'])/'program.py').write_text('stop()')
        with self.assertRaisesRegex(ValueError,'evidence changed'):build(self.root)

    def test_reject_program_mismatch(self):
        self.rows[0]['completion']='stop()';self.flush()
        with self.assertRaisesRegex(ValueError,'failure-prefix'):build(self.root)

    def test_reject_parent_provenance_mismatch(self):
        self.save('assembly_provenance.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'provenance'):build(self.root)

    def test_one_sided_layout_has_no_pairs(self):
        self.rows=self.rows[:1];self.flush()
        self.assertEqual(build(self.root)['pairs'],[])


if __name__=='__main__':unittest.main()
