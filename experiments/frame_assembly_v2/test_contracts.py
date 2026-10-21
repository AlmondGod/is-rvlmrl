import unittest
import numpy as np,mujoco
from .environment import Env,GOALS,YAWS
from .skills import Skills,nail_pose
from .perception import Vision

class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=Env()
 def estimates(self):return [{'center':p.copy(),'yaw':float(y),'time':0.} for p,y in zip(GOALS,YAWS)]
 def test_four_free_members_three_articulated_arms(self):
  m=self.e.m
  free=[m.joint(j).name for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
  self.assertEqual(free,[f'wood{i}_free' for i in range(4)])
  self.assertEqual(m.nu,23)
  self.assertEqual(m.neq,6) # native hand couplings only
  for i in range(m.neq):
   self.assertNotIn(int(m.eq_obj1id[i]),self.e.woods);self.assertNotIn(int(m.eq_obj2id[i]),self.e.woods)
  self.assertFalse(any('marker' in (m.geom(g).name or '') for g in range(m.ngeom)))
 def test_nailer_requires_assembled_released_frame(self):
  s=Skills(self.e.robot());est=self.estimates();est[1]['center'][0]+=.1
  with self.assertRaises(ValueError):s.start({'skill':'nail','corner':0},est,self.e.robot())
  est=self.estimates();s.holding[0]=True
  with self.assertRaises(ValueError):s.start({'skill':'nail','corner':0},est,self.e.robot())
 def test_one_abstract_event_per_corner_no_physics_mutation(self):
  e=self.e;s=Skills(e.robot());est=self.estimates();q=e.d.qpos.copy();eq=e.d.eq_active.copy()
  s.start({'skill':'nail','corner':0},est,e.robot());s.state='tool_seat';robot=e.robot();robot['tcp'][2]=s.tip;robot['rotation'][2]=s.toolR
  for _ in range(30):s.command(robot,est)
  self.assertEqual(len(s.events),1);self.assertFalse(s.events[0]['physical_nail_or_bond']);np.testing.assert_array_equal(q,e.d.qpos);np.testing.assert_array_equal(eq,e.d.eq_active)
  s.state='done'
  with self.assertRaises(ValueError):s.start({'skill':'nail','corner':0},est,robot)
 def test_tool_must_be_positioned_before_firing(self):
  s=Skills(self.e.robot());est=self.estimates();s.start({'skill':'nail','corner':0},est,self.e.robot());s.state='tool_seat'
  for _ in range(40):s.command(self.e.robot(),est)
  self.assertEqual(s.events,[])
 def test_release_cannot_skip_position_convergence(self):
  s=Skills(self.e.robot());s.holding[0]=True;est=self.estimates();est[0]['center'][0]+=.02
  with self.assertRaises(ValueError):s.start({'skill':'release','piece':0},est,self.e.robot())
 def test_black_rgb_has_no_ground_truth_fallback(self):
  camera={'K':[[900,0,639],[0,900,479],[0,0,1]],'world_to_camera':[[1,0,0,0],[0,-1,0,0],[0,0,-1,2.4],[0,0,0,1]]}
  out=Vision().update(np.zeros((3,960,1280,3),np.uint8),[camera]*3,self.e.robot())
  self.assertEqual(out,[None]*4)
 def test_corner_measurements_are_rigid_rotation_invariant(self):
  from .geometry import corner_errors
  angle=.23;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
  centers=[];rotations=[]
  for center,yaw in zip(GOALS,YAWS):
   R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
   centers.append(Q@center+np.array([.1,-.2,0]));rotations.append(Q@R)
  for c in corner_errors(centers,rotations):
   self.assertAlmostEqual(c['gap_m'],.002,places=8);self.assertAlmostEqual(c['outer_flush_m'],0.,places=8);self.assertTrue(c['ok'])
 def test_idle_joint_goals_do_not_walk_in_nullspace(self):
  e=Env();before=np.array(e.qtarget).copy()
  for _ in range(150):e.step(e.targets.copy(),e.rotations.copy(),np.zeros(2))
  np.testing.assert_allclose(e.qtarget,before,atol=1e-9)
  self.assertLess(e.peak,.0005)
  self.assertFalse(any(w.number for w in e.d.warning))
 def test_every_nailing_target_has_feasible_kinematics(self):
  for c in range(4):
   p,R=nail_pose(c);self.assertAlmostEqual(float(np.linalg.det(R)),1.)
   _,err=self.e.ik(2,p,R,iterations=250,multistart=True);self.assertLess(err,.0001)
if __name__=='__main__':unittest.main()
