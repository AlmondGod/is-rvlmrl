import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from analysis import assembly_cohort_grpo as cohort
from experiments.cap_rl import train as trainer


def logprobs(model,sequences,attention,length,temperature):
    return model(sequences).squeeze(-1)/temperature


class CohortTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(93)
        self.model=torch.nn.Embedding(20,1,dtype=torch.float64)
        self.batches=[];self.results=[]
        for n,length in enumerate((3,5)):
            seq=torch.randint(0,20,(8,length))
            with torch.no_grad():old=logprobs(self.model,seq,None,length,3.).clone()
            self.batches.append(SimpleNamespace(policy_version='frozen-0',
                layouts=[SimpleNamespace(key=str(n*2+i//4)) for i in range(8)],
                sequences=seq,attention=torch.ones_like(seq),mask=torch.ones_like(old),
                old=old,reference=old+.1))
            self.results.append([dict(reward=float(r),success=bool(r)) for r in (0,1,0,1,0,0,1,1)])

    def test_global_gradient_matches_independent_per_sequence_mean(self):
        expected=copy.deepcopy(self.model)
        opt=torch.optim.SGD(self.model.parameters(),lr=.03)
        reference_opt=torch.optim.SGD(expected.parameters(),lr=.03)
        with patch('analysis.cap_rl_assembly_grammar.constrained_logprobs',logprobs):
            records=cohort.update(trainer,self.model,opt,self.batches,self.results,microbatch=3,epochs=3)
        for _ in range(3):
            reference_opt.zero_grad(set_to_none=True)
            losses=[]
            for batch,rows in zip(self.batches,self.results):
                advantages=trainer.advantages(torch.tensor([r['reward'] for r in rows]),4)
                for i in range(8):
                    section=slice(i,i+1)
                    p=logprobs(expected,batch.sequences[section],None,batch.mask.shape[1],3.)
                    loss,_=trainer.policy_loss(p,batch.old[section],batch.reference[section],
                        batch.mask[section],advantages[section],.005)
                    losses.append(loss)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
            reference_opt.step()
        torch.testing.assert_close(self.model.weight,expected.weight,atol=1e-14,rtol=1e-14)
        self.assertEqual(len(records),3)
        self.assertTrue(all(r['sampled_programs']==16 for r in records))

    def test_rejects_mixed_versions_incomplete_groups_and_nonbinary_rewards(self):
        self.batches[1].policy_version='changed'
        with self.assertRaisesRegex(ValueError,'policy version'):
            cohort.validate_cohort(self.batches,self.results,4)
        self.batches[1].policy_version='frozen-0'
        self.batches[1].layouts[0].key='0'
        with self.assertRaisesRegex(ValueError,'complete group'):
            cohort.validate_cohort(self.batches,self.results,4)
        self.batches[1].layouts[0].key='2'
        self.results[0][0]['reward']=.8
        with self.assertRaisesRegex(ValueError,'binary'):
            cohort.validate_cohort(self.batches,self.results,4)

    def test_no_reward_collection_between_dispatches(self):
        groups=[b.layouts for b in self.batches]
        seen=[]
        def dispatch(*args,**kwargs):
            seen.append(args[3]);return SimpleNamespace(collect=lambda:self.fail('premature reward wait'))
        with patch('analysis.assembly_pipelined_rollouts.dispatch',dispatch):
            result=cohort.dispatch_all(None,self.model,None,groups,None,policy_version='0',max_tokens=512)
        self.assertEqual(seen,groups);self.assertEqual(len(result),2)


if __name__=='__main__':unittest.main()
