import unittest,numpy as np,mujoco
from .environment import Env,PANEL_SIZE,PANEL_GOAL,FIXTURE_HEIGHT
from .perception import Vision
from .skills import Skills
from .tool_plan import PanelPlanner,nail_target
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=Env()
 def test_five_free_workpieces_and_no_workpiece_constraints(self):
  e=self.e;self.assertEqual(sum(e.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE),5)
  for i in range(e.m.neq):self.assertFalse('wood' in (e.m.body(e.m.eq_obj1id[i]).name or '') or 'panel' in (e.m.body(e.m.eq_obj1id[i]).name or ''))
 def test_black_camera_has_no_state_fallback(self):self.assertIsNone(Vision().update(np.zeros((3,960,1280,3),np.uint8),[{}]*3))
 def test_fixture_and_mass_are_explicit(self):
  e=self.e;self.assertAlmostEqual(e.m.body_mass[e.panel],np.prod(PANEL_SIZE)*600);self.assertAlmostEqual(PANEL_GOAL[2],FIXTURE_HEIGHT+.07);self.assertEqual(e.m.joint('overhead_z').type,mujoco.mjtJoint.mjJNT_SLIDE)
 def test_nailing_requires_released_sheet(self):
  s=Skills(self.e.robot(),None)
  with self.assertRaises(ValueError):s.start('nail',0,self.e.robot(),{'center':PANEL_GOAL,'yaw':0})
 def test_abstract_fire_has_no_physics_mutation_and_rejects_duplicate(self):
  e=self.e;s=Skills(e.robot(),None);s.released=True;s.member=0;s.tip,s.toolR=nail_target(0);s.state='tool_seat';r=e.robot();r['tcp'][2]=s.tip;r['rotation'][2]=s.toolR;before=e.d.qpos.copy()
  for _ in range(27):s.command(r,{'center':PANEL_GOAL,'rotation':np.eye(3)},None)
  self.assertEqual(len(s.events),1);np.testing.assert_array_equal(before,e.d.qpos);s.state='done'
  with self.assertRaises(ValueError):s.start('nail',0,r,{'center':PANEL_GOAL,'yaw':0})
if __name__=='__main__':unittest.main()
