from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from analysis.assembly_prefix_rewards import group_programs,PendingGroups,verify_episode_prefix,sha
from analysis.benchmark_assembly_prefix import SnapshotFailure


def program(order=(0,1,2,3)):
    lines=[line for i in range(4) for line in (f'position_member({i})',f'release_member({i})')]
    lines+=['park_grippers()']+[f'nail_corner({i})' for i in range(4)]
    lines+=['feed_panel()','position_panel()','release_panel()']
    lines += [f'nail_panel({i})' for i in order]+['flip_structure()','finish()']
    return '\n'.join(lines)


class PrefixRewardTests(unittest.TestCase):
    def test_grouping_keeps_layout_and_physical_prefix_together(self):
        a,b=SimpleNamespace(seed=1),SimpleNamespace(seed=2)
        groups=group_programs([a,b,a],[program(),program(),program((3,2,1,0))])
        self.assertEqual([[r[0] for r in group] for group in groups],[[0,2],[1]])
        changed=program().replace('position_member(0)','position_member(1)',1)
        self.assertEqual(len(group_programs([a,a],[program(),changed])),2)

    def test_results_restore_sample_order_and_propagate_worker_errors(self):
        a,b=SimpleNamespace(seed=1),SimpleNamespace(seed=2)
        groups=group_programs([a,b,a],[program(),program(),program()+'\n'])
        f=Future();f.set_result([dict(reward=0),dict(reward=0)])
        g=Future();g.set_result([dict(reward=1)])
        pending=PendingGroups([(groups[0],f),(groups[1],g)],3,lambda:None)
        rows=pending.result()
        self.assertEqual([r['reward'] for r in rows],[0,1,0])
        self.assertEqual([r['batch_duplicate'] for r in rows],[False,False,True])
        failed=Future();failed.set_exception(RuntimeError('infrastructure failure'))
        with self.assertRaisesRegex(RuntimeError,'infrastructure'):
            PendingGroups([(groups[0],failed)],2,lambda:None).result()

    def test_reuse_audit_compares_actual_recorded_prefix_and_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);origin=root/'origin';target=root/'target';origin.mkdir();target.mkdir()
            for p,text in ((origin,program()),(target,program((3,2,1,0)))):
                (p/'program.py').write_text(text)
                (p/'provenance.json').write_text('{"frozen":true}')
                (p/'result.json').write_text('{"seed":51001}')
            rows=np.zeros((4,155));np.savez_compressed(origin/'states.npz',rows=rows)
            tail=rows.copy();tail[2:]=7;np.savez_compressed(target/'states.npz',rows=tail)
            (origin/'prefix_receipt.json').write_text(json.dumps(dict(reused=False,captured_step=7)))
            receipt=dict(reused=True,prefix_calls=16,restored_step=7,origin_episode=str(origin),
                source_sha256={'/frozen/a.py':'x'},origin_files_sha256={p.name:sha(p) for p in origin.iterdir()})
            (target/'prefix_receipt.json').write_text(json.dumps(receipt))
            expected=dict(prefix_calls=16,sources_sha256={'a.py':'x','assembly_prefix_rewards.py':'y'})
            self.assertTrue(verify_episode_prefix(target,expected)['reused'])
            tail[0,1]=1;np.savez_compressed(target/'states.npz',rows=tail)
            with self.assertRaisesRegex(SnapshotFailure,'physical prefix'):
                verify_episode_prefix(target,expected)
            receipt['source_sha256']={};(target/'prefix_receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(SnapshotFailure,'incomplete'):
                verify_episode_prefix(target,expected)


if __name__=='__main__':unittest.main()
