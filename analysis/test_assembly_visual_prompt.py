import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from analysis import assembly_visual_prompt as visual
from analysis.assembly_initial_observation import describe
from analysis.train_assembly_preferences import program_inventory
from experiments.cap_rl import task_assembly as task


class VisualPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        source=self.root/'preference_source';source.mkdir()
        for name in visual.SOURCES:shutil.copyfile(Path(visual.__file__).with_name(name),source/name)
        self.metadata=dict(prompt_profile=visual.PROFILE,
            visual_source_sha256={name:visual.sha(source/name) for name in visual.SOURCES},
            splits={'train':['assembly-full-v1-51001'],'validation':['assembly-full-v1-52001'],
                    'test':['assembly-full-v1-53001']})
        visual.dump(self.root/'assembly_provenance.json',{'fixture':'test-only'})
        self.observation=dict(format=visual.PROFILE,time_s=0.,members=[
            dict(member=i,observed=True,x_m=(-1 if i<2 else 1)*(.84 if i%2==0 else .68),y_m=0.,yaw_deg=0.)
            for i in range(4)])
        for key in self.metadata['splits']['train']+self.metadata['splits']['validation']:
            self.write_observation(key)
        visual.prepare(self.root,self.metadata,'development')
        self.metadata['observations_manifest_sha256']=visual.sha(self.root/'observations_manifest.json')

    def write_observation(self,key):
        directory=self.root/'observations'/key;directory.mkdir(parents=True)
        for i in (0,2,3):(directory/f'camera{i}.png').write_bytes(b'fixture image bytes')
        visual.dump(directory/'sensor_inputs.json',{})
        visual.dump(directory/'observation.json',self.observation)
        (directory/'prompt_fragment.txt').write_text(describe(self.observation)+'\n')
        visual.dump(directory/'receipt.json',dict(seed=visual.layout_seed(key),observation_time_s=0.,
            source_sha256=self.metadata['visual_source_sha256']['assembly_initial_observation.py'],
            physical_provenance=visual.read(self.root/'assembly_provenance.json'),
            files_sha256={p.name:visual.sha(p) for p in directory.iterdir()}))

    def test_hidden_metadata_cannot_change_prompt(self):
        secret=copy.deepcopy(self.observation)
        secret.update(seed=999123,true_poses='SECRET_POSES',success=True)
        secret['members'][0]['hidden_pose']='SECRET_POSES'
        self.assertEqual(describe(secret),describe(self.observation))
        secret['members'][0]['observed']=False
        self.assertIn('not visually localized',describe(secret))
        self.assertNotIn('SECRET_POSES',describe(secret))

    def test_sensor_tampering_is_rejected(self):
        key=self.metadata['splits']['train'][0]
        (self.root/'observations'/key/'camera0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'payload changed'):
            visual.load_observation(self.root,self.metadata,key)

    def test_source_tampering_is_rejected(self):
        (self.root/'preference_source'/visual.SOURCES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError,'frozen visual'):
            visual.verify_sources(self.root,self.metadata)

    def test_manifest_is_immutable(self):
        visual.dump(self.root/'observations_manifest.json',{})
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            visual.verify_sources(self.root,self.metadata)

    def test_test_capture_requires_learned_perfect_selection(self):
        for rate,step in ((0.5,8),(1.,0)):
            visual.dump(self.root/'selection.json',dict(validation_success_rate=rate,best_step=step))
            with self.assertRaisesRegex(ValueError,'learned 8/8'):
                visual.prepare(self.root,self.metadata,'test')
        self.assertFalse((self.root/'observations'/'assembly-full-v1-53001').exists())

    def test_context_restores_original_prompt_after_error(self):
        original=task.prompt
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with visual.installed(self.root,self.metadata):
                self.assertIn('Initial camera estimates',task.prompt(task.Layout(51001)))
                raise RuntimeError('deliberate')
        self.assertIs(task.prompt,original)

    def test_scene_augmentation_preserves_raw_programs_and_evidence(self):
        raw=dict(pairs=[dict(layout_id='assembly-full-v1-51001',prompt=task.prompt(task.Layout(51001)),
             chosen='good()',rejected='bad()',chosen_evidence_id='a',rejected_evidence_id='b')])
        before=copy.deepcopy(raw);data=visual.augment_preferences(raw,self.root,self.metadata)
        self.assertEqual(raw,before)
        for key in ('chosen','rejected','chosen_evidence_id','rejected_evidence_id'):
            self.assertEqual(raw['pairs'][0][key],data['pairs'][0][key])
        self.assertEqual(data['pairs'][0]['source_prompt_sha256'],hashlib.sha256(raw['pairs'][0]['prompt'].encode()).hexdigest())
        self.assertIn('Initial camera estimates',data['pairs'][0]['prompt'])

    def test_identical_code_under_distinct_contexts_is_not_deduplicated(self):
        data={'pairs':[dict(layout_id=key,chosen='good()',rejected='bad()') for key in ('a','b')]}
        self.assertEqual(len(program_inventory(data,'a',False)),2)
        self.assertEqual(program_inventory(data,'a',True),[('a','good()'),('a','bad()'),('b','good()'),('b','bad()')])


if __name__=='__main__':unittest.main()
