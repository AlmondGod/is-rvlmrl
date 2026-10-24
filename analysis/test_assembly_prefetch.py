import copy,unittest
from analysis.prefetch_assembly_training import select_candidates
from analysis.train_assembly_visual_grpo import schedule

class PrefetchTests(unittest.TestCase):
    def fixture(self):
        keys=[f'assembly-full-v1-{i}' for i in range(51001,51033)]
        m=dict(optimizer_algorithm='grpo_visual_balanced_v1',warm_start_weights_sha256='weights',model_revision='revision',splits={'train':keys,'validation':['assembly-full-v1-52001'],'test':['assembly-full-v1-53001']},training_layout_schedule=schedule(keys))
        r=dict(passes=True,weights_sha256='weights',model_revision='revision',rows=[dict(layout_id=k,exact=True,completion='fixture') for k in keys+['assembly-full-v1-52001']])
        return m,r
    def test_only_future_training_layouts_are_queued(self):
        m,r=self.fixture();keys,rows=select_candidates(m,r)
        self.assertEqual(len(keys),28);self.assertFalse(set(keys)&set(m['training_layout_schedule'][0]))
        self.assertEqual(set(rows),set(m['splits']['train']))
        self.assertFalse(set(keys)&set(m['splits']['validation']+m['splits']['test']))
    def test_unused_validation_mismatch_does_not_change_training_candidates(self):
        m,r=self.fixture();r['passes']=False;r['rows'][-1]['exact']=False
        keys,rows=select_candidates(m,r)
        self.assertEqual(len(keys),28);self.assertTrue(all(v['exact'] for v in rows.values()))
    def test_changed_source_incomplete_or_overlapping_inputs_rejected(self):
        for change in ('weights','duplicate','mismatch','overlap'):
            m,r=self.fixture()
            if change=='weights':r['weights_sha256']='other'
            elif change=='duplicate':r['rows'].append(copy.deepcopy(r['rows'][0]))
            elif change=='mismatch':r['rows'][0]['exact']=False
            else:m['splits']['test'].append(m['splits']['train'][0])
            with self.subTest(change=change),self.assertRaises(ValueError):select_candidates(m,r)

if __name__=='__main__':unittest.main()
