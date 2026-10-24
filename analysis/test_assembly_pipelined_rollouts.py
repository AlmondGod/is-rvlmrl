from concurrent.futures import Future
import unittest

from analysis.assembly_async_rewards import RewardQueue
from analysis.assembly_pipelined_rollouts import PendingPrograms,parameter_versions


class Pool:
    def __init__(self):self.jobs=[]
    def submit(self,*args):
        future=Future();self.jobs.append(future);return future


class PendingProgramTests(unittest.TestCase):
    def batch(self,queue,key):
        return PendingPrograms(queue.submit([key],['program']),str(key),[key],None,None,None,['program'],None,None,{},0.)
    def test_later_completed_rewards_can_be_consumed_without_waiting_for_slow_batch(self):
        pool=Pool();queue=RewardQueue(pool,None,lambda a,b:(a,b),lambda *a:None)
        slow=self.batch(queue,1);fast=self.batch(queue,2)
        pool.jobs[1].set_result(dict(success=True,reward=1.))
        self.assertFalse(slow.ready());self.assertTrue(fast.ready())
        self.assertEqual(fast.collect(),[dict(success=True,reward=1.,truncated=False,batch_duplicate=False,inflight_shared=False)])
        self.assertEqual(fast.policy_version,'2')
    def test_failed_worker_does_not_become_a_reward(self):
        pool=Pool();queue=RewardQueue(pool,None,lambda a,b:(a,b),lambda *a:None)
        batch=self.batch(queue,1);pool.jobs[0].set_exception(RuntimeError('infrastructure'))
        with self.assertRaisesRegex(RuntimeError,'infrastructure'):batch.collect()
    def test_parameter_mutation_changes_version_guard(self):
        class Parameter:requires_grad=True;_version=0
        class Model:
            parameter=Parameter()
            def named_parameters(self):return [('adapter',self.parameter)]
        model=Model();before=parameter_versions(model);model.parameter._version+=1
        self.assertNotEqual(before,parameter_versions(model))




try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'CPU tensor fixture requires PyTorch')
class DispatchFixtureTests(unittest.TestCase):
    def test_submit_precedes_logprobs_and_scores_are_frozen_on_cpu(self):
        from types import SimpleNamespace
        from contextlib import contextmanager
        from unittest.mock import patch
        from analysis import assembly_pipelined_rollouts as pipeline,cap_rl_assembly_grammar as grammar
        events=[]
        class Inputs(dict):
            def __getattr__(self,name):return self[name]
        class Grammar:
            def validate(self,row):
                if row!=[2,9]:raise ValueError('Invalid program')
            def decision_mask(self,mask):return mask
        class Model:
            training=True;reference=False
            def named_parameters(self):return []
            def eval(self):self.training=False
            def train(self,value):self.training=value
            def generate(self,**kwargs):return torch.tensor([[1,2,9]])
        model=Model()
        @contextmanager
        def reference(m):
            m.reference=True
            try:yield
            finally:m.reference=False
        def scores(m,*args):
            events.append('reference' if m.reference else 'old')
            return torch.full((1,2),2. if m.reference else 1.)
        pool=Pool();queue=RewardQueue(pool,None,lambda a,b:(a,b),lambda *a:None)
        original_submit=queue.submit
        def submit(*args):events.append('submit');return original_submit(*args)
        queue.submit=submit
        trainer=SimpleNamespace(encode=lambda *args:Inputs(input_ids=torch.tensor([[1]]),attention_mask=torch.tensor([[1]])),
            completion_mask=lambda *args:torch.ones((1,2),dtype=torch.bool),reference_policy=reference)
        tokenizer=SimpleNamespace(pad_token_id=9,eos_token_id=9,batch_decode=lambda *a,**k:['program'])
        with patch.object(grammar,'generation_kwargs',return_value={}),patch.object(grammar,'_GRAMMAR',Grammar()),patch.object(grammar,'constrained_logprobs',side_effect=scores):
            batch=pipeline.dispatch(trainer,model,tokenizer,[1],queue,policy_version='checkpoint-1',max_tokens=2)
        self.assertEqual(events,['submit','old','reference'])
        self.assertEqual(batch.old.tolist(),[[1.,1.]])
        self.assertEqual(batch.reference.tolist(),[[2.,2.]])
        self.assertEqual(batch.sequences.device.type,'cpu');self.assertTrue(model.training)
        self.assertFalse(batch.ready())
        pool.jobs[0].set_result(dict(reward=1.,success=True));self.assertEqual(batch.collect()[0]['reward'],1.)


if __name__=='__main__':unittest.main()
