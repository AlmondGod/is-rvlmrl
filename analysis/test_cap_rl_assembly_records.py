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

    def run_fixture(self, mutate=None, pretest=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);layouts,manifest=self.fixture(root)
            if mutate:mutate(root)
            with patch.object(audit.task,'provenance',return_value=manifest),\
                 patch.object(audit.task,'datasets',return_value=layouts),\
                 patch.object(audit.task,'prompt',return_value='fixture prompt'):
                return audit.audit(root,pretest=pretest)

    def test_valid_accounting(self):
        r=self.run_fixture();self.assertTrue(r['passes']);self.assertEqual(r['test_records'],4)

    def test_unexecuted_truncation_has_no_physics_credit(self):
        def mutate(root):
            p=root/'rollouts.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
            row=rows[0]
            for key in ('evidence','evidence_id','result_sha256','cache_reuse'):row.pop(key)
            row.update(truncated=True,physics_started=False,error='truncated',valid=False)
            p.write_text('\n'.join(json.dumps(r) for r in rows))
        self.assertTrue(self.run_fixture(mutate)['passes'])

    def enable_strict(self,root):
        source=Path(__file__).with_name('train_assembly_strict.py').read_bytes()
        directory=root/'optimization_source';directory.mkdir()
        (directory/'train_assembly_strict.py').write_bytes(source)
        p=root/'metadata.json';metadata=audit.read(p)
        metadata.update(optimization_reward='strict_success',
                        optimization_adapter_sha256=hashlib.sha256(source).hexdigest())
        p.write_text(json.dumps(metadata))
        rows=[json.loads(s) for s in (root/'rollouts.jsonl').read_text().splitlines()]
        # A synthetic high-progress failure: all hardware remains assembled,
        # but the unchanged relative-angle gate fails. It must earn zero for RL.
        row=rows[0];evidence=Path(row['evidence']);raw=audit.read(evidence/'result.json')
        raw.update(reward=.8,verified_released_members=list(range(4)),fasteners={'intact':8},
                   preflip_stable_s=3.,inverted_hold_s=5.,flipped=True,airborne_relative_angle_peak_deg=3.1)
        (evidence/'result.json').write_text(json.dumps(raw))
        cert=audit.read(evidence/'certificate.json');cert['files']['result.json']=audit.sha(evidence/'result.json')
        (evidence/'certificate.json').write_text(json.dumps(cert))
        row.update(raw,result_sha256=cert['files']['result.json'])
        def convert(values):
            for value in values:value.update(dense_reward=value['reward'],reward=float(value['success']))
        convert(rows)
        (root/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        for name in ('validation_baseline.json','validation_0001.json','test_comparison.json'):
            p=root/name;report=audit.read(p)
            for key in ('rows','before','after'):
                if key in report:convert(report[key])
            p.write_text(json.dumps(report))

    def test_strict_optimizer_keeps_dense_evidence_separate(self):
        result=self.run_fixture(self.enable_strict)
        self.assertEqual(result['optimization_reward'],'strict_success')

    def static_fixture(self,root):
        from analysis.train_assembly_strict import required_call_failure
        self.enable_strict(root)
        p=root/'metadata.json';metadata=audit.read(p)
        metadata['static_strict_rejection']='strict_necessary_calls_v1';p.write_text(json.dumps(metadata))
        p=root/'rollouts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
        row=rows[0]
        for key in ('evidence','evidence_id','result_sha256','cache_reuse'):row.pop(key)
        row.update(physics_started=False,error='strict_missing_required_calls',dense_reward=None,
                   dense_reward_measured=False,static_rejection=required_call_failure(row['completion']))
        p.write_text('\n'.join(json.dumps(r) for r in rows))

    def test_static_proof_and_unknown_dense_are_audited(self):
        self.assertTrue(self.run_fixture(self.static_fixture)['passes'])
        for key,value in [('reward',1.),('dense_reward',0.),('physics_started',True),
                          ('static_rejection',{'format':'strict_necessary_calls_v1','missing':[]})]:
            def mutate(root):
                self.static_fixture(root)
                p=root/'rollouts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                rows[0][key]=value;p.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_fixture(mutate)

    def test_rounds_are_distinct_from_optimizer_epochs(self):
        def mutate(root):
            p=root/'metadata.json';metadata=audit.read(p);metadata['update_epochs']=4;p.write_text(json.dumps(metadata))
            p=root/'training.jsonl';row=json.loads(p.read_text());row.update(optimizer_steps_total=4,epoch_metrics=[{'epoch':i} for i in range(1,5)])
            p.write_text(json.dumps(row))
        result=self.run_fixture(mutate)
        self.assertEqual(result['rollout_rounds'],1);self.assertEqual(result['optimizer_updates'],4)
        def broken(root):
            mutate(root)
            p=root/'training.jsonl';row=json.loads(p.read_text());row['optimizer_steps_total']=3;p.write_text(json.dumps(row))
        with self.assertRaises(ValueError):self.run_fixture(broken)

    def test_optional_raw_baseline_is_audited(self):
        def mutate(root):
            p=root/'metadata.json';metadata=audit.read(p);metadata['model_revision']='fixture';p.write_text(json.dumps(metadata))
            report=audit.read(root/'test_comparison.json')
            (root/'raw_test_baseline.json').write_text(json.dumps(dict(rows=report['before'],metrics=report['baseline'],model_revision='fixture')))
        result=self.run_fixture(mutate);self.assertEqual(result['raw_test_records'],2)
        def broken(root):
            mutate(root)
            p=root/'raw_test_baseline.json';raw=audit.read(p);raw['rows'][0]['reward']=1.;p.write_text(json.dumps(raw))
        with self.assertRaises((ValueError,AssertionError)):self.run_fixture(broken)

    def enable_ceiling_stop(self,root):
        self.enable_strict(root)
        p=root/'metadata.json';metadata=audit.read(p);metadata['steps']=3
        p.write_text(json.dumps(metadata))
        (root/'validation_ceiling_policy.json').write_text(json.dumps({'upper_bound':[1.,1.]}))
        names=['metadata.json','training.jsonl','rollouts.jsonl','validation_ceiling_policy.json',
               'validation_baseline.json','validation_0001.json']
        source=root/'selection_source/stop_assembly04_at_validation_ceiling.py'
        source.parent.mkdir();source.write_text('synthetic selection fixture')
        record=dict(format='cap_rl_validation_ceiling_stop_v1',upper_bound=[1.,1.],test_evaluated=False,
                    completed_updates=1,planned_updates=3,selected_step=1,
                    source_sha256=audit.sha(source),files_sha256={name:audit.sha(root/name) for name in names})
        (root/'validation_ceiling_stop.json').write_text(json.dumps(record))

    def test_modern_ceiling_finalizer_preserves_and_audits_checkpoint(self):
        from analysis import finalize_assembly_ceiling as finalizer
        def mutate(root):
            self.enable_strict(root)
            (root/'test_comparison.json').unlink();(root/'selection.json').unlink()
            p=root/'metadata.json';metadata=audit.read(p);metadata['steps']=3;p.write_text(json.dumps(metadata))
            p=root/'training.jsonl';update=audit.read(p);update['elapsed_seconds']=1.;p.write_text(json.dumps(update))
            (root/'validation_ceiling_policy.json').write_text(json.dumps({'upper_bound':[1.,1.]}))
            original=Path.read_bytes
            def read_bytes(path):
                if str(path)=='/proc/123456/cmdline':return ('python -m analysis.cap_rl_assembly_grammar --out '+str(root)).encode()
                return original(path)
            with patch.object(Path,'read_bytes',read_bytes),patch.object(finalizer,'running',return_value=False),patch.object(finalizer.os,'kill') as kill:
                finalizer.finalize(root,123456,1)
                kill.assert_called_once()
            self.assertEqual(audit.sha(root/'last_adapter/adapter_model.safetensors'),audit.sha(root/'best_adapter/adapter_model.safetensors'))
        result=self.run_fixture(mutate,pretest=True)
        self.assertTrue(result['passes']);self.assertTrue(result['stopped_at_validation_ceiling'])

    def test_ceiling_stop_preserves_planned_budget(self):
        result=self.run_fixture(self.enable_ceiling_stop)
        self.assertEqual(result['optimizer_updates'],1)
        self.assertEqual(result['planned_optimizer_updates'],3)
        self.assertTrue(result['stopped_at_validation_ceiling'])

    def test_pretest_selection_audit_requires_unused_test(self):
        def mutate(root):
            self.enable_ceiling_stop(root)
            (root/'test_comparison.json').unlink()
        result=self.run_fixture(mutate,pretest=True)
        self.assertTrue(result['passes']);self.assertEqual(result['selected_step'],1)
        with self.assertRaises(ValueError):self.run_fixture(self.enable_ceiling_stop,pretest=True)

    def test_invalid_ceiling_receipts_rejected(self):
        for key,value in [('upper_bound',[.9,.9]),('test_evaluated',True),
                          ('completed_updates',2),('selected_step',0),('files_sha256',{})]:
            def mutate(root):
                self.enable_ceiling_stop(root)
                p=root/'validation_ceiling_stop.json';record=audit.read(p);record[key]=value
                p.write_text(json.dumps(record))
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_fixture(mutate)

    def test_ceiling_stop_changed_records_rejected(self):
        def mutate(root):
            self.enable_ceiling_stop(root)
            p=root/'training.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):self.run_fixture(mutate)

    def test_strict_optimizer_source_and_dense_reward_are_verified(self):
        for target in ('source','reward'):
            def mutate(root):
                self.enable_strict(root)
                if target=='source':(root/'optimization_source/train_assembly_strict.py').write_text('changed')
                else:
                    p=root/'rollouts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                    rows[0]['dense_reward']=.7;p.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.subTest(target=target),self.assertRaises((ValueError,AssertionError)):
                self.run_fixture(mutate)

    def test_pretest_rejects_secondary_test_reports(self):
        for name in ('raw_test_baseline.json','unconstrained_test_comparison.json'):
            def mutate(root):
                (root/'test_comparison.json').unlink();(root/name).write_text('{}')
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Test already evaluated'):
                self.run_fixture(mutate,pretest=True)

    def test_secondary_transfer_binds_weights_source_and_all_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            metadata=dict(policy_decoding='assembly_permutations_v1',model_revision='fixture',splits={'test':['a','b']})
            (root/'metadata.json').write_text(json.dumps(metadata));(root/'selection.json').write_text('{}')
            for folder,value in [('initial_adapter',b'initial'),('best_adapter',b'selected')]:
                (root/folder).mkdir();(root/folder/'adapter_model.safetensors').write_bytes(value)
            source=root/'evaluation_source/cap_rl_assembly_unconstrained.py';source.parent.mkdir()
            source.write_bytes(Path(__file__).with_name('cap_rl_assembly_unconstrained.py').read_bytes())
            before=[dict(layout_id=k,success=s,reward=float(s),valid=True) for k,s in [('a',False),('b',True)]]
            after=[dict(layout_id=k,success=True,reward=1.,valid=True) for k in ('a','b')]
            report=dict(format='assembly_unconstrained_transfer_v1',split='test',evaluation_decoding='unrestricted_program',
                selection_role='diagnostic_only',model_revision='fixture',metadata_sha256=audit.sha(root/'metadata.json'),
                selection_sha256=audit.sha(root/'selection.json'),initial_weights_sha256=audit.sha(root/'initial_adapter/adapter_model.safetensors'),
                selected_weights_sha256=audit.sha(root/'best_adapter/adapter_model.safetensors'),evaluator_source_sha256=audit.sha(source),
                before=before,after=after,baseline=audit.metrics(before),trained=audit.metrics(after),paired_wins=1,paired_losses=0,delta_success_rate=.5)
            path=root/'unconstrained_test_comparison.json';path.write_text(json.dumps(report));seen=[]
            result=audit.audit_transfer(root,metadata,'test',lambda r,s:seen.append((r,s)))
            self.assertTrue(result['passes']);self.assertEqual(len(seen),4)
            report['selected_weights_sha256']='changed';path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):audit.audit_transfer(root,metadata,'test',lambda r,s:None)

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

class PreferenceReplayAuditTests(unittest.TestCase):
    # Reuse synthetic physical records to exercise new optimizer accounting.
    def preference_fixture(self,root):
        layouts,manifest=RecordsAuditTests().fixture(root)
        original=[json.loads(s) for s in (root/'rollouts.jsonl').read_text().splitlines()]
        parent=root/'parent_run';parent.mkdir();(parent/'rollouts.jsonl').write_text((root/'rollouts.jsonl').read_text())
        pair=dict(layout_id=original[0]['layout_id'],chosen_row=1,rejected_row=0)
        data={'pairs':[pair]}
        p=root/'metadata.json';metadata=audit.read(p)
        metadata.update(optimizer_algorithm='offline_dpo_v1',steps=48,eval_every=8,update_epochs=1,prompts_per_step=4)
        p.write_text(json.dumps(metadata))
        training=[];rows=[]
        for step in range(1,9):
            training.append(dict(step=step,optimizer_steps_total=step,pair_indices=[0]*4,new_training_physics_episodes=0,replayed_programs=8))
            for _ in range(4):
                for role,index in [('chosen',1),('rejected',0)]:
                    rows.append(dict(original[index],step=step,preference_pair_id=0,preference_role=role,parent_source_row=index,offline_replay=True))
        (root/'training.jsonl').write_text('\n'.join(json.dumps(r) for r in training))
        (root/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        (root/'validation_0001.json').rename(root/'validation_0008.json')
        (root/'checkpoints/step-0001').rename(root/'checkpoints/step-0008')
        p=root/'selection.json';selection=audit.read(p)
        selection.update(best_step=8,completed_optimizer_updates=8,planned_optimizer_updates=48,stopped_at_validation_ceiling=True)
        p.write_text(json.dumps(selection))
        return layouts,manifest,data

    def test_preference_updates_are_not_new_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);layouts,manifest,data=self.preference_fixture(root)
            with patch.object(audit.task,'provenance',return_value=manifest),\
                 patch.object(audit.task,'datasets',return_value=layouts),\
                 patch.object(audit.task,'prompt',return_value='fixture prompt'),\
                 patch.object(audit,'audit_preference_inputs',return_value=(data,{'passes':True})):
                report=audit.audit(root)
                self.assertEqual(report['rollout_rounds'],0);self.assertEqual(report['optimizer_updates'],8)
                self.assertEqual(report['training_samples'],0);self.assertEqual(report['replayed_training_programs'],64)
                self.assertTrue(report['stopped_at_validation_ceiling'])
                p=root/'rollouts.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                rows[0]['parent_source_row']=0;p.write_text('\n'.join(json.dumps(r) for r in rows))
                with self.assertRaises((AssertionError,ValueError)):audit.audit(root)
