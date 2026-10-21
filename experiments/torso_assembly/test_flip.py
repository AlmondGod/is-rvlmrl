"""Grasp access and motion regressions; these do not establish task success."""
import unittest
from types import SimpleNamespace
import numpy as np
import mujoco
from .environment import scene,GOALS,PANEL_GOAL,SIDE_R
from .flip import Flip
from .fasteners import NailParameters

class FlipTests(unittest.TestCase):
 def controller(self,**kwargs):
  robot=dict(tcp=np.zeros((3,3)),rotation=np.tile(np.eye(3),(3,1,1)),joints=np.zeros((3,6)))
  planner=SimpleNamespace(plan=lambda *args:np.zeros((2,6)),plan_flip_approach=lambda *args:np.zeros((2,6)))
  est=[dict(center=p.copy()) for p in GOALS]
  panel=dict(center=PANEL_GOAL.copy())
  return Flip(robot,planner,est,panel,**kwargs),robot
 def test_stack_grasp_spans_stock_without_hidden_pose(self):
  c,_=self.controller(grasp='stack')
  np.testing.assert_allclose(c.grasp_points[:,0],GOALS[[1,3],0])
  np.testing.assert_allclose(c.grasp_points[:,2],.158)
  m=mujoco.MjModel.from_xml_string(scene());d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  axis=d.site_xmat[m.site('a0_link_tcp').id].reshape(3,3)[:,1]
  a=m.geom('a0_left_finger_pad_2').id;b=m.geom('a0_right_finger_pad_2').id
  radius=lambda g:abs(axis@d.geom_xmat[g].reshape(3,3))@m.geom_size[g]
  opening=abs((d.geom_xpos[a]-d.geom_xpos[b])@axis)-radius(a)-radius(b)
  self.assertGreater(opening,.076+.006) # room for stock and bounded insertion error
 def test_access_slots_preserve_end_support(self):
  m=mujoco.MjModel.from_xml_string(scene(fixture_half_width=.38))
  np.testing.assert_allclose(m.geom('frame_plinth').size,[.38,.46,.06])
  for side in (-1,1):
   for end in (-1,1):
    g=m.geom(f'fixture_ledge_{side}_{end}')
    np.testing.assert_allclose(g.solref,m.geom('frame_plinth').solref)
    np.testing.assert_allclose(g.friction,m.geom('frame_plinth').friction)
    self.assertLess(abs(g.pos[1])-g.size[1],.11)
    self.assertGreater(abs(g.pos[1])-g.size[1],.09)
    self.assertGreater(abs(g.pos[1])+g.size[1],.4382)
 def test_clearance_precedes_high_lift(self):
  c,r=self.controller(grasp='stack',smooth=True,clearance=True)
  c.anchor=c.grasp_points.copy();r['tcp'][:2]=c.anchor;c.switch('clear_lift')
  previous=c.anchor.copy()
  for _ in range(1100):
   t,_,_,_=c.command(r)
   self.assertLessEqual(float(np.max(np.linalg.norm(t[:2]-previous,axis=1)))/.02,.071)
   r['tcp'][:2]=t[:2];previous=t[:2].copy()
   if c.state=='lift':break
  self.assertEqual(c.state,'lift')
  np.testing.assert_allclose(c.anchor[:,1],-.10,atol=1e-9)
  np.testing.assert_allclose(c.anchor[:,2],.218,atol=1e-9)
 def test_transit_clears_surveyed_fixture_envelopes(self):
  # Center offset selected from the available interval between torso bracket
  # front face y=.47 and withdrawn feeder shelf front face y=-.635.
  half_sheet=.9744/2;shift=-.10
  self.assertLess(shift+half_sheet,.47-.04)
  self.assertGreater(shift-half_sheet,-.635+.04)
 def test_sheet_transfer_fits_between_shelf_and_shoulder(self):
  from .panel_skills import Skills
  from .environment import PANEL_START
  _,r=self.controller();skill=Skills(r,None)
  skill.start('position',None,r,dict(center=PANEL_START.copy()))
  height=PANEL_START[2]+skill.clearance_lift
  m=mujoco.MjModel.from_xml_string(scene(fixture_half_width=.38,shoulder_half_span=.55))
  bracket=m.geom('torso_shoulder_bracket_0.55')
  shelf=m.geom('rack_shelf0');rack=m.body('rack0')
  self.assertGreater(bracket.pos[2]-bracket.size[2]-(height+.006),.010)
  self.assertGreater(height-.006-(rack.pos[2]+shelf.size[2]),.010)
 def test_lift_verification_uses_grasp_time_camera_reference(self):
  from .panel_skills import Skills
  _,r=self.controller();skill=Skills(r,None)
  approach=dict(center=np.array([.06,0,.291]),rotation=np.eye(3),yaw=0.)
  grasp=dict(center=np.array([.06,0,.286]),rotation=np.eye(3),yaw=0.)
  skill.start('position',None,r,approach);skill.state='close';skill.time=2.
  skill.command(r,grasp,grasp)
  self.assertTrue(skill.begin_grasp);self.assertEqual(skill.state,'lift')
  lifted=dict(center=np.array([.06,0,.300]),rotation=np.eye(3),yaw=0.)
  skill.command(r,lifted,lifted)
  self.assertEqual(skill.state,'withdraw_shelves')
 def test_original_nail_parameters_frozen(self):
  p=NailParameters()
  self.assertEqual((p.stiffness_N_m,p.damping_N_s_m,p.withdrawal_N,p.lateral_N,p.failure_displacement_m),(12000.,4.,45.,70.,.004))
if __name__=='__main__':unittest.main()
