import unittest
import numpy as np
from experiments.box_assembly.fastening import Frame,Connector
from .fasteners import ray_interval,NailParameters

class NailTests(unittest.TestCase):
 def test_ray_miss_and_finite_embed(self):
  f=Frame(np.zeros(3),np.eye(3))
  self.assertIsNone(ray_interval(np.array([2,0,0]),np.array([0,0,1]),f,[1,1,1],1))
  self.assertEqual(ray_interval(np.array([0,0,1]),np.array([0,0,-1]),f,[1,1,1],.6),(.5,.6))
 def test_finite_force_and_irreversible_break(self):
  p=NailParameters();a=Frame(np.zeros(3),np.eye(3));b=Frame(np.zeros(3),np.eye(3));c=Connector.at(a,b,np.zeros(3));b.position[2]=.001
  force,point,energy,loss=c.evaluate(a,b,np.array([0,0,1]),p.stiffness_N_m,p.damping_N_s_m,p.withdrawal_N,p.lateral_N,p.failure_displacement_m,.001)
  self.assertAlmostEqual(force[2],12.);self.assertAlmostEqual(energy,.006);self.assertFalse(c.failed)
  b.position[2]=.005;c.evaluate(a,b,np.array([0,0,1]),p.stiffness_N_m,p.damping_N_s_m,p.withdrawal_N,p.lateral_N,p.failure_displacement_m,.001);self.assertTrue(c.failed)
  b.position[:]=0;f,_,_,_=c.evaluate(a,b,np.array([0,0,1]),12000,4,45,70,.004,.001);np.testing.assert_array_equal(f,0)
 def test_damping_dissipates_relative_motion(self):
  a=Frame(np.zeros(3),np.eye(3));b=Frame(np.zeros(3),np.eye(3),linear=np.array([.1,0,0]));c=Connector.at(a,b,np.zeros(3));f,x,energy,loss=c.evaluate(a,b,np.array([0,0,1]),12000,4,45,70,.004,.001)
  self.assertLess(float(f@a.velocity(x)-f@b.velocity(x)),0);self.assertGreater(loss,0)
 def test_installation_is_pose_preserving_and_internal_forces_balance(self):
  import mujoco
  from types import SimpleNamespace
  from .environment import scene,GOALS,YAWS
  from .fasteners import NailBank
  m=mujoco.MjModel.from_xml_string(scene());d=mujoco.MjData(m)
  for i in range(4):
   q=int(m.joint(f'wood{i}_free').qposadr[0]);d.qpos[q:q+3]=GOALS[i];d.qpos[q+3:q+7]=[np.cos(YAWS[i]/2),0,0,np.sin(YAWS[i]/2)]
  mujoco.mj_forward(m,d);axis=np.array([0,-.8,-.6]);x=np.array([1.,0,0]);R=np.column_stack((x,np.cross(axis,x),axis));tcp=np.zeros((3,3));tcp[2]=[-.4382,-.3992,.187]
  e=SimpleNamespace(m=m,d=d,woods=[m.body(f'wood{i}').id for i in range(4)],panel=m.body('panel').id,robot=lambda:dict(tcp=tcp,rotation=np.array([np.eye(3),np.eye(3),R])))
  bank=NailBank(e);before=d.qpos.copy();bank.fire('frame',0);np.testing.assert_array_equal(d.qpos,before)
  q=int(m.joint('wood1_free').qposadr[0]);d.qpos[q]+=.001;mujoco.mj_forward(m,d);bank.apply()
  forces=[]
  for i in (0,1):
   v=int(m.joint(f'wood{i}_free').dofadr[0]);forces.append(d.qfrc_applied[v:v+3])
  self.assertGreater(np.linalg.norm(forces[0]),0);np.testing.assert_allclose(forces[0]+forces[1],0,atol=1e-10)
  with self.assertRaises(ValueError):bank.fire('frame',0)
  missed=bank.fire('frame',1);self.assertFalse(missed['physical_bond']);self.assertEqual(len(bank.connections),1)
 def test_flip_gate_requires_retention_inversion_and_clearance(self):
  from .flip_evaluator import passes
  x=dict(panel_up_dot=-1.,position_errors_m=[0.]*4,angle_errors_rad=[0.]*4,minimum_height_above_fixture_m=.1)
  self.assertTrue(passes(x,8,True));self.assertFalse(passes(x,7,True));self.assertFalse(passes(x,8,False))
  for key,value in [('panel_up_dot',0.),('minimum_height_above_fixture_m',0.),('position_errors_m',[.01]*4),('angle_errors_rad',[.1]*4)]:self.assertFalse(passes(dict(x,**{key:value}),8,True))
 def test_latent_nail_state_is_not_in_policy_packet(self):
  from .live import packet
  r={k:0 for k in ('time','tcp','rotation','joints','joint_velocity','gripper_command','shuttle_x','rack_slide0','rack_slide1')}
  p=packet(0,r,'done',1,None,{},[],[dict(kind='frame',index=0,triggered=True,physical_bond=True,point_m=[1,2,3],intact=True)],[])
  self.assertEqual(p['abstract_fasteners'],[dict(kind='frame',index=0,triggered=True)])
 def test_invalid_parameters_reject(self):
  with self.assertRaises(ValueError):NailParameters(withdrawal_N=-1)
if __name__=='__main__':unittest.main()
