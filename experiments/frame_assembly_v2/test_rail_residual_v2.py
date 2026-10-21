import unittest
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .rail_residual_v2 import BoundedResidual, SCALE, exploration_sigma, factory, new_model, select


class CorrectedResidualTests(unittest.TestCase):
    def test_network_is_exactly_zero_for_arbitrary_observations(self):
        torch.set_num_threads(1)
        vec=DummyVecEnv([factory(0)])
        model=new_model(vec,0)
        obs=np.random.default_rng(12).normal(size=(32,*vec.observation_space.shape)).astype(np.float32)
        actions=model.predict(obs,deterministic=True)[0]
        np.testing.assert_array_equal(actions,np.zeros_like(actions))
        self.assertFalse(model.policy.log_std.requires_grad)
        vec.close()

    def test_action_cap_and_exploration_schedule(self):
        # ActionWrapper's transform is independent of simulator state.
        a=BoundedResidual.action(None,np.array([-100.,-1.,0.,.5,100.]))
        np.testing.assert_allclose(a,[-SCALE,-SCALE,0.,SCALE/2,SCALE])
        self.assertEqual(exploration_sigma(0),.05)
        self.assertAlmostEqual(exploration_sigma(65536),.30)
        self.assertAlmostEqual(exploration_sigma(262144),.30)

    def test_selection_requires_validation_improvement(self):
        def rows(lifts,full=0):
            return [{'success':i<full,'lift_success':i<lifts,'max_penetration_m':.002} for i in range(20)]
        base=rows(7)
        self.assertEqual(select(base,{'equal':rows(7),'worse':rows(2)}),'baseline')
        self.assertEqual(select(base,{'better':rows(8),'tie':rows(8)}),'better')
        self.assertEqual(select(base,{'lift_only':rows(20),'placed':rows(1,1)}),'placed')


if __name__=='__main__':unittest.main(verbosity=2)
