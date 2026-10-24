"""Tests of balanced online layout coverage and source-bound audit inputs."""
import unittest
from analysis.train_assembly_visual_grpo import schedule

class BalancedScheduleTests(unittest.TestCase):
    def test_every_training_layout_exactly_once(self):
        keys=[f'assembly-full-v1-{seed}' for seed in range(51001,51033)]
        groups=schedule(keys)
        self.assertEqual(len(groups),8)
        self.assertTrue(all(len(g)==4 and len(set(g))==4 for g in groups))
        self.assertEqual(sorted(k for g in groups for k in g),keys)
        self.assertEqual(groups,schedule(keys))
        self.assertNotEqual(groups,schedule(keys,98))
    def test_missing_or_duplicated_layout_is_rejected(self):
        for keys in (list(range(31)),[0]*32):
            with self.assertRaises(ValueError):schedule(keys)

if __name__=='__main__':unittest.main()

class BalancedInputAuditTests(unittest.TestCase):
    def test_frozen_sources_schedule_and_warm_start(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from analysis import cap_rl_assembly_records as audit,assembly_visual_prompt as visual
        def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'parent_run';parent.mkdir()
            def save(p,value):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            save(parent/'metadata.json',{'parent':True});save(parent/'retirement.json',{'retired':True})
            save(root/'parent_record_audit.json',{'passes':True})
            for name in ('initial_adapter','warm_start_adapter','parent_run/initial_adapter'):
                p=root/name/'adapter_model.safetensors';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture-weights')
            warm=digest(root/'warm_start_adapter/adapter_model.safetensors')
            save(parent/'checkpoints/step-0006/checkpoint.json',{'files_sha256':{'adapter_model.safetensors':warm}})
            receipts={}
            for name in (*visual.SOURCES,'train_assembly_visual_grpo.py','cap_rl_assembly_records.py'):
                src=Path(audit.__file__).with_name(name);dst=root/'preference_source'/name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(src.read_bytes());receipts[name]=digest(dst)
            save(root/'visual_prompt_attempt/metadata.json',{'prompt_profile':visual.PROFILE,'optimizer_algorithm':'offline_dpo_v1'})
            save(root/'visual_prompt_attempt/selection.json',{'completed_optimizer_updates':48,'validation_success_rate':.5})
            save(root/'constant_prompt_attempt/selection.json',{'completed_optimizer_updates':48})
            keys=[f'assembly-full-v1-{i}' for i in range(51001,51033)]
            m=dict(optimizer_algorithm='grpo_visual_balanced_v1',parent_metadata_sha256=digest(parent/'metadata.json'),parent_retirement_sha256=digest(parent/'retirement.json'),online_source_sha256=receipts,warm_start_weights_sha256=warm,
                steps=8,eval_every=2,checkpoint_every=1,prompts_per_step=4,group_size=4,update_epochs=4,microbatch=8,lr=.0001,beta=.005,sampling_temperature=3.,seed=97,splits={'train':keys},training_layout_schedule=schedule(keys))
            for name in ('visual_prompt_attempt','constant_prompt_attempt'):m[name+'_sha256']={p.name:digest(p) for p in (root/name).iterdir()}
            with patch.object(audit,'audit',return_value={'passes':True}):
                self.assertEqual(audit.audit_visual_grpo_inputs(root,m),{'passes':True})
                altered=dict(m,lr=.1)
                with self.assertRaises(ValueError):audit.audit_visual_grpo_inputs(root,altered)
                altered=dict(m,online_source_sha256={})
                with self.assertRaises(ValueError):audit.audit_visual_grpo_inputs(root,altered)
                (root/'warm_start_adapter/adapter_model.safetensors').write_bytes(b'changed')
                with self.assertRaises(ValueError):audit.audit_visual_grpo_inputs(root,m)
