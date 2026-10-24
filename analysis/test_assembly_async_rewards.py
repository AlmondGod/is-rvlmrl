import unittest
import tempfile
from pathlib import Path
from concurrent.futures import Future
from analysis.assembly_async_rewards import RewardQueue, check_memory_budget


class Pool:
    def __init__(self): self.jobs=[]
    def submit(self, function, layout, text):
        future=Future(); self.jobs.append((layout,text,future)); return future


class AsyncRewards(unittest.TestCase):
    def setup_queue(self, shortcut=lambda *args:None, validate=lambda:None):
        pool=Pool()
        return pool,RewardQueue(pool,lambda *args:None,lambda layout,text:(layout,text),shortcut,validate)

    def test_slow_group_does_not_block_submitting_or_consuming_other_groups(self):
        pool,queue=self.setup_queue()
        slow=queue.submit([1,1],['a','b']);fast=queue.submit([2,2],['c','d'])
        self.assertEqual(len(pool.jobs),4)
        for _,_,f in pool.jobs[2:]:f.set_result({'reward':1.,'success':True})
        self.assertEqual(queue.wait_ready([slow,fast],timeout=0),[fast])
        self.assertFalse(slow.ready()); self.assertEqual(len(fast.result()),2)

    def test_duplicates_share_only_inflight_execution_and_keep_attribution(self):
        pool,queue=self.setup_queue()
        first=queue.submit([1,1],['a','a']);second=queue.submit([1],['a'])
        self.assertEqual(len(pool.jobs),1)
        pool.jobs[0][2].set_result({'reward':1.,'success':True})
        self.assertEqual([r['batch_duplicate'] for r in first.result()],[False,True])
        self.assertTrue(second.result()[0]['inflight_shared'])
        # A finished result must be checked again, not silently trusted forever.
        checked=[];queue.shortcut=lambda *args:checked.append(args) or {'reward':0.,'success':False}
        next_batch=queue.submit([1],['a'])
        self.assertEqual(len(checked),1);self.assertFalse(next_batch.result()[0]['success'])

    def test_infrastructure_error_is_not_zero_reward(self):
        pool,queue=self.setup_queue();batch=queue.submit([1],['a'])
        pool.jobs[0][2].set_exception(RuntimeError('worker failed'))
        with self.assertRaisesRegex(RuntimeError,'worker failed'):batch.result()

    def test_source_change_rejects_dispatch_and_collection(self):
        valid=[True]
        def validate():
            if not valid[0]:raise RuntimeError('source changed')
        pool,queue=self.setup_queue(validate=validate);batch=queue.submit([1],['a'])
        valid[0]=False
        with self.assertRaisesRegex(RuntimeError,'source changed'):queue.submit([2],['b'])
        with self.assertRaisesRegex(RuntimeError,'source changed'):batch.result()
        self.assertEqual(len(pool.jobs),1)

    def test_memory_guard_rejects_a_second_large_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'memory.max').write_text('31000000000')
            (root/'memory.current').write_text('22000000000')
            (root/'memory.stat').write_text('inactive_file 5000000000\nanon 17000000000\n')
            with self.assertRaisesRegex(RuntimeError,'Insufficient host RAM'):
                check_memory_budget(8,root)
            receipt=check_memory_budget(4,root)
            self.assertEqual(receipt['working_set_bytes'],17000000000)
            self.assertEqual(receipt['reserved_bytes'],12000000000)

    def test_missing_memory_limit_is_not_reported_as_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(check_memory_budget(8,Path(folder))['bounded'])


if __name__=='__main__':unittest.main()
