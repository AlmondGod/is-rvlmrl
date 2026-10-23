"""Synthetic accounting fixtures only; no claim of physical execution."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from analysis import cap_rl_assembly_records as audit


class RecordsAuditTests(unittest.TestCase):
    def fixture(self, root):
        root=Path(root)
        layouts={k:[audit.task.Layout(i) for i in seeds] for k,seeds in
                 [('train',[41001,41002]),('validation',[42001,42002]),('test',[43001,43002])]}
        profiles=dict(support_profile='contact_and_nails',controller_profile='southwest_approach')
        manifest=dict(sha256={},profiles=profiles)
        def save(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        metadata=dict(task='assembly_full',**profiles,source_sha256={},steps=1,eval_every=1,
                      group_size=2,prompts_per_step=1,seed=42,
                      splits={k:[x.key for x in v] for k,v in layouts.items()})
        save('metadata.json',metadata);save('assembly_provenance.json',manifest)
        def row(layout,success):
            program='flip_structure()\nfinish()' if success else 'finish()'
            calls=audit.parse_program(program)
            identity=dict(seed=layout.seed,calls=calls,provenance=manifest)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            evidence=root/'evidence'/key
            evidence.mkdir(parents=True,exist_ok=True)
            raw=dict(seed=layout.seed,success=success,passes=success,valid=True,error=None,
                reward=float(success),verified_released_members=list(range(4)) if success else [],
                fasteners={'intact':8 if success else 0},preflip_stable_s=3. if success else 0.,
                inverted_hold_s=5. if success else 0.,peak_penetration_m=0.,
                nonfinite_or_warning=False,unexpected_robot_contacts=[],finished=True,flipped=success,
                airborne_relative_position_peak_m=0.,airborne_relative_angle_peak_deg=0.)
            (evidence/'program.py').write_text(program)
            for name,value in [('result.json',raw),('provenance.json',manifest)]:
                (evidence/name).write_text(json.dumps(value))
            files={p.name:audit.sha(p) for p in evidence.iterdir() if p.name!='certificate.json'}
            (evidence/'certificate.json').write_text(json.dumps(dict(identity_sha256=key,files=files)))
            return dict(layout_id=layout.key,prompt='fixture prompt',completion=program,**raw,
                evidence=str(evidence),evidence_id=key,result_sha256=files['result.json'],
                cache_reuse='exact_program',truncated=False)
        chosen=random.Random(42).sample(layouts['train'],1)[0]
        trainrows=[dict(step=1,**row(chosen,s)) for s in (False,True)]
        (root/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in trainrows))
        (root/'training.jsonl').write_text(json.dumps(dict(step=1,reward=.5,success_rate=.5)))
        for name,success in [('validation_baseline.json',False),('validation_0001.json',True)]:
            rows=[row(x,success) for x in layouts['validation']]
            save(name,dict(metrics=audit.metrics(rows),rows=rows))
        save('selection.json',dict(best_step=1,selection_split='validation',validation_success_rate=1.,validation_mean_reward=1.))
        for name in ('checkpoints/step-0001','best_adapter'):
            p=root/name;p.mkdir(parents=True);(p/'adapter_model.safetensors').write_bytes(b'fixture only')
        save('checkpoints/step-0001/checkpoint.json',dict(files_sha256={
            'adapter_model.safetensors':audit.sha(root/'best_adapter/adapter_model.safetensors')}))
        before=[row(x,False) for x in layouts['test']];after=[row(x,True) for x in layouts['test']]
        save('test_comparison.json',dict(before=before,after=after,baseline=audit.metrics(before),trained=audit.metrics(after),
            paired_wins=2,paired_losses=0,delta_success_rate=1.))
        return layouts,manifest

    def run_fixture(self, mutate=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);layouts,manifest=self.fixture(root)
            if mutate:mutate(root)
            with patch.object(audit.task,'provenance',return_value=manifest),\
                 patch.object(audit.task,'datasets',return_value=layouts),\
                 patch.object(audit.task,'prompt',return_value='fixture prompt'):
                return audit.audit(root)

    def test_valid_accounting(self):
        r=self.run_fixture();self.assertTrue(r['passes']);self.assertEqual(r['test_records'],4)

    def test_tampered_test_rows_rejected(self):
        for key,value in [('reward',0.),('completion','stop()'),('prompt','leaked prompt'),
                          ('layout_id','assembly-full-v1-41001'),('result_sha256','wrong')]:
            def mutate(root):
                p=root/'test_comparison.json';r=audit.read(p);r['after'][0][key]=value;p.write_text(json.dumps(r))
            with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):
                self.run_fixture(mutate)

    def test_changed_evidence_rejected(self):
        def mutate(root):
            row=audit.read(root/'test_comparison.json')['after'][0]
            p=Path(row['evidence'])/'result.json';r=audit.read(p);r['reward']=.5;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):self.run_fixture(mutate)

    def test_wrong_checkpoint_or_aggregate_rejected(self):
        for target in ('best_adapter/adapter_model.safetensors','selection.json','test_comparison.json'):
            def mutate(root):
                p=root/target
                if p.suffix=='.safetensors':p.write_bytes(b'different adapter')
                else:
                    r=audit.read(p)
                    if target=='selection.json':r['best_step']=0
                    else:r['paired_wins']=1
                    p.write_text(json.dumps(r))
            with self.subTest(target=target),self.assertRaises(ValueError):self.run_fixture(mutate)

if __name__=='__main__':unittest.main()
