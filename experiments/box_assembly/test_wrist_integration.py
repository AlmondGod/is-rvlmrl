"""Local numerical regression for the formerly oscillating free wrist carrier."""
from dataclasses import replace
import unittest
import mujoco
import numpy as np
from .design import Config
from .fastening_environment import FasteningBoxAssembly


class WristIntegrationTest(unittest.TestCase):
 def test_free_space_hold_at_default_and_half_step(self):
  results=[]
  for dt in (Config().timestep,Config().timestep/2):
   with self.subTest(timestep=dt):
    e=FasteningBoxAssembly(replace(Config(),timestep=dt),allow_local=True);e.reset(seed=7)
    velocities=[]
    for i in range(100):
     scores=e.step(e.wrist_target.copy(),e.wrist_quaternion.copy(),e.finger_target.copy())
     if i>50:
      v=np.zeros(6);mujoco.mj_objectVelocity(e.model,e.data,mujoco.mjtObj.mjOBJ_BODY,e.wid,v,0)
      velocities.append(np.linalg.norm(v[:3]))
    self.assertLess(max(velocities),1e-3,'Uncommanded wrist rotation during hold')
    self.assertEqual(e.stage_saturation_steps,0)
    self.assertTrue(scores['finite_state']);self.assertFalse(any(scores['warnings']))
    results.append(e.data.xpos[e.wid].copy())
  self.assertLess(np.linalg.norm(results[0]-results[1]),1e-5)

if __name__=='__main__':unittest.main()
