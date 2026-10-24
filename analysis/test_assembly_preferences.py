import random
import unittest
from analysis.train_assembly_preferences import sample_pairs
from analysis.retire_assembly_plateau import same_policy_results


class PreferenceSamplingTests(unittest.TestCase):
    def test_layout_uniform_not_pair_count_weighted(self):
        data={'pairs':[{'layout_id':'a'}]+[{'layout_id':'b'} for _ in range(100)]}
        draws=sample_pairs(data,random.Random(42),2000)
        a=sum(i==0 for i in draws)
        self.assertGreater(a,850);self.assertLess(a,1150)
        self.assertEqual(draws,sample_pairs(data,random.Random(42),2000))

    def test_empty_pairs_rejected(self):
        with self.assertRaises(ValueError):sample_pairs({'pairs':[]},random.Random(42),4)

    def test_changed_greedy_program_does_not_trigger_retirement(self):
        report={'metrics':{'success_rate':.5},'rows':[dict(layout_id='a',completion='A',success=False,reward=0.)]}
        other={'metrics':dict(report['metrics']),'rows':[dict(report['rows'][0],completion='B')]}
        self.assertTrue(same_policy_results(report,report));self.assertFalse(same_policy_results(report,other))

    def test_dpo_loss_and_gradient_direction(self):
        try:import torch
        except ImportError:self.skipTest('Torch GPU environment runs this numerical check')
        from analysis.train_assembly_preferences import pair_loss
        policy=torch.zeros((1,2),requires_grad=True);reference=torch.zeros((1,2))
        loss,margin=pair_loss(policy,reference,.1);loss.backward()
        self.assertAlmostEqual(float(loss),.69314718,places=6)
        self.assertLess(float(policy.grad[0,0]),0);self.assertGreater(float(policy.grad[0,1]),0)
        good,_=pair_loss(torch.tensor([[2.,-2.]]),reference,.1)
        bad,_=pair_loss(torch.tensor([[-2.,2.]]),reference,.1)
        self.assertLess(float(good),float(bad));self.assertIsNone(reference.grad)


if __name__=='__main__':unittest.main()
