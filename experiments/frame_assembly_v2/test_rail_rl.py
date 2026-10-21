import unittest
import mujoco
import numpy as np

from .rail_rl import RailEnv


class RailTests(unittest.TestCase):
    def test_zero_residual_is_exact_base_controller(self):
        a,b=RailEnv('residual'),RailEnv('base')
        a.reset(seed=17);b.reset(seed=17)
        for _ in range(30):
            oa,ra,*_=a.step(np.zeros(26));ob,rb,*_=b.step(np.zeros(26))
            np.testing.assert_array_equal(a.sim.data.qpos,b.sim.data.qpos)
            self.assertEqual(ra,rb)
        self.assertFalse(any(a.sim.data.eq_active[e] for e in a.sim.carries.values()))

    def test_observations_and_randomization_match_across_methods(self):
        a,b=RailEnv('residual'),RailEnv('direct')
        oa,_=a.reset(seed=31);ob,_=b.reset(seed=31)
        np.testing.assert_array_equal(oa,ob)
        self.assertEqual(a.friction,b.friction)
        np.testing.assert_array_equal(a.sim.data.xpos[a.bid],a.initial)
        self.assertEqual(a.observation_space.shape,(164,))

    def test_raised_center_with_one_end_on_table_is_not_lift(self):
        e=RailEnv();d,m=e.sim.data,e.sim.model
        theta=.7
        d.qpos[e.qa:e.qa+3]=[.4,0,.2]
        d.qpos[e.qa+3:e.qa+7]=[np.cos(theta/2),np.sin(theta/2),0,0]
        mujoco.mj_forward(m,d);lowest=e.measure()['lowest']
        d.qpos[e.qa+2]-=lowest
        mujoco.mj_forward(m,d)
        self.assertGreater(d.xpos[e.bid][2],.05)
        self.assertAlmostEqual(e.measure()['lowest'],0.)
        self.assertFalse(e.lift_achieved)

    def test_goal_pose_without_lift_history_is_not_success(self):
        e=RailEnv();d=e.sim.data
        d.qpos[e.qa:e.qa+3]=e.goal
        mujoco.mj_forward(e.sim.model,d)
        for _ in range(60):e.step(np.zeros(26))
        self.assertFalse(e.success)
        self.assertEqual(e.place_hold,0.)


if __name__=='__main__':unittest.main(verbosity=2)
