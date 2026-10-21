import unittest
import mujoco,numpy as np
from .environment import Env
from .perception import Vision
from .skills import Skills

class Contracts(unittest.TestCase):
 def test_articulated_not_floating(self):
  e=Env();free=np.where(e.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0];self.assertEqual({e.m.joint(int(j)).name for j in free},{'wood0_free','wood1_free'});self.assertEqual(e.m.nu,23)
  for ids in e.ids:self.assertTrue(np.all(e.m.jnt_type[ids]==mujoco.mjtJoint.mjJNT_HINGE))
  self.assertFalse(any('mark' in (e.m.geom(i).name or '') for i in range(e.m.ngeom)))
 def test_native_linkages_do_not_attach_wood(self):
  e=Env();self.assertEqual(e.m.neq,6)
  for j in range(e.m.neq):
   if e.m.eq_type[j]==mujoco.mjtEq.mjEQ_CONNECT:
    self.assertTrue(all(int(b) not in e.woods for b in (e.m.eq_obj1id[j],e.m.eq_obj2id[j])))
 def test_unloaded_dynamics_and_motor_limits(self):
  e=Env();before=e.d.xpos[e.woods].copy()
  for _ in range(50):e.step()
  self.assertFalse(np.any(e.d.xfrc_applied));self.assertFalse(np.any(e.d.qfrc_applied));self.assertTrue(np.isfinite(e.d.qpos).all());self.assertLess(e.peak,.0005)
  np.testing.assert_allclose(e.d.xpos[e.woods][:,:2],before[:,:2],atol=1e-4)
  self.assertTrue(np.all(e.d.actuator_force<=e.m.actuator_forcerange[:,1]+1e-6))
  self.assertTrue(np.all(e.d.actuator_force>=e.m.actuator_forcerange[:,0]-1e-6))
 def test_black_rgb_never_falls_back(self):
  e=Env();v=Vision();cam={'K':np.eye(3),'world_to_camera':np.eye(4)};self.assertEqual(v.update(np.zeros((1,60,80,3),np.uint8),[cam],e.robot()),[None,None])
 def test_probe_requires_observed_lift(self):
  e=Env();s=Skills(e.robot());r=e.robot();estimate=[{'center':np.array([.38,-.21,.032]),'time':0.,'yaw':0.},None];s.state='probe';s.probe_start=estimate[0]['center'];s.tcp_start=r['tcp'][0].copy()
  for i in range(160):r['time']=i*.02;estimate[0]['time']=r['time'];s.command(r,estimate)
  self.assertEqual(s.failure,'probe_no_visual_lift');self.assertFalse(s.holding[0])
if __name__=='__main__':unittest.main()
