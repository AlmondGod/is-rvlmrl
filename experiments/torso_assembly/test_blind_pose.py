"""Regression tests for workpiece-pose information boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
from .discovery import discover
from .planner import Planner
from .tool_skills import Tool
from experiments.wall_panel.perception import HeldSheet

class BlindPoseTests(unittest.TestCase):
 def test_missing_images_fail_closed(self):
  frames=[np.zeros((200,300,3),np.uint8)]*4
  observed,debug=discover(frames,[None]*4,{})
  self.assertIsNone(observed)
  self.assertIn('need_two',debug['reason'])
 def test_planner_refuses_reset_fallback(self):
  p=object.__new__(Planner)
  with self.assertRaisesRegex(ValueError,'no reset fallback'):p.sync({},[None]*4)
 def test_sheet_height_comes_from_observation(self):
  r=dict(rotation=np.array([np.eye(3)]*2),tcp=np.array([[-.5,0,.4],[.5,0,.4]]))
  for z in (.23,.31):
   estimate=dict(center=np.array([.01,.02,z]),rotation=np.eye(3),yaw=0.)
   h=HeldSheet();h.begin(estimate,r)
   np.testing.assert_allclose(h.estimate(None,r)['center'],estimate['center'])
 def test_nail_targets_move_with_observed_material(self):
  class P:
   def plan(self,*args):return np.zeros((2,6))
  r=dict(tcp=np.zeros((3,3)),rotation=np.array([np.eye(3)]*3))
  estimates=[dict(center=np.array(p),yaw=0.) for p in ([0,-.438,.152],[-.438,0,.152],[0,.438,.152],[.438,0,.152])]
  panel=dict(center=np.array([0,0,.20]),rotation=np.eye(3))
  for kind in ('frame','panel'):
   before=Tool(r,P(),kind,0,estimates,panel).tip
   shift=np.array([.013,-.017,.009]);moved=[dict(e,center=e['center']+shift) for e in estimates]
   after=Tool(r,P(),kind,0,moved,dict(panel,center=panel['center']+shift)).tip
   np.testing.assert_allclose(after-before,shift,atol=1e-12)

if __name__=='__main__':unittest.main()
