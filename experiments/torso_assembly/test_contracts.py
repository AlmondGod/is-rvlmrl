"""Structural guarantees, independent of eventual task success."""
import unittest,ast
from pathlib import Path
import mujoco,numpy as np
from .environment import scene,STARTS,PANEL_START,UR,SHUTTLE_START
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=mujoco.MjModel.from_xml_string(scene())
 def test_single_fixed_torso(self):
  m=self.m;root=m.body('robot_torso').id;self.assertEqual(m.body_jntnum[root],0)
  for a in range(3):
   b=m.body(f'a{a}_link_base').id
   self.assertEqual(m.body_parentid[b],root)
   self.assertEqual(len([m.joint(f'a{a}_joint{k}') for k in range(1,7)]),6)
 def test_all_five_workpieces_free_and_initially_loose(self):
  m=self.m
  for i in range(4):
   j=m.joint(f'wood{i}_free').id;self.assertEqual(m.jnt_type[j],mujoco.mjtJoint.mjJNT_FREE);np.testing.assert_allclose(m.qpos0[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3],STARTS[i])
  j=m.joint('panel_free').id;self.assertEqual(m.jnt_type[j],mujoco.mjtJoint.mjJNT_FREE);np.testing.assert_allclose(m.qpos0[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3],PANEL_START+[SHUTTLE_START,0,0])
 def test_no_workpiece_equalities(self):
  # Native gripper connect/joint equalities never reference workpieces.
  for i in range(self.m.neq):
   typ=self.m.eq_type[i];self.assertIn(typ,(mujoco.mjtEq.mjEQ_CONNECT,mujoco.mjtEq.mjEQ_JOINT))
   accessor=self.m.body if typ==mujoco.mjtEq.mjEQ_CONNECT else self.m.joint
   for obj in (self.m.eq_obj1id[i],self.m.eq_obj2id[i]):self.assertTrue(accessor(obj).name.startswith(('a0_','a1_')))
 def test_native_ur_lengths_inertias_and_limits(self):
  import xml.etree.ElementTree as ET
  r=ET.parse(UR/'ur10e.xml').getroot()
  for a in range(3):
   for name in ('upper_arm_link','forearm_link','wrist_1_link'):
    source=r.find(f".//body[@name='{name}']");b=self.m.body(f'a{a}_ur_{name}').id
    np.testing.assert_allclose(self.m.body_pos[b],np.fromstring(source.get('pos'),sep=' '))
    self.assertAlmostEqual(self.m.body_mass[b],float(source.find('inertial').get('mass')))
 def test_gripper_geometry_defaults_preserved(self):
  from experiments.frame_assembly_v2.environment import scene as native_scene
  native=mujoco.MjModel.from_xml_string(native_scene())
  for i in range(2):
   root=self.m.body(f'a{i}_xarm_gripper_base_link').id
   for geom in range(self.m.ngeom):
    body=self.m.geom_bodyid[geom];ancestor=body
    while ancestor and ancestor!=root:ancestor=self.m.body_parentid[ancestor]
    if ancestor!=root:continue
    name=self.m.geom(geom).name;reference=native.geom(name).id
    self.assertEqual(self.m.geom_type[geom],native.geom_type[reference],name)
    np.testing.assert_allclose(self.m.geom_size[geom],native.geom_size[reference],err_msg=name)
    np.testing.assert_allclose(self.m.geom_friction[geom],native.geom_friction[reference],err_msg=name)
 def test_runtime_does_not_write_workpiece_poses(self):
  for name in ('run.py','live.py','frame_skills.py','panel_skills.py','tool_skills.py','frame_vision.py','flip.py'):
   tree=ast.parse((Path(__file__).parent/name).read_text())
   if name=='run.py':
    # Explicit, oracle-only development checkpoint initialization is exempt;
    # the uninterrupted deployment loop must never assign workpiece state.
    for parent in ast.walk(tree):
     if hasattr(parent,'body') and isinstance(parent.body,list):
      parent.body[:]=[n for n in parent.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='a.resume_diagnostic')]
   for n in ast.walk(tree):
    if isinstance(n,(ast.Assign,ast.AugAssign)):
     targets=n.targets if isinstance(n,ast.Assign) else [n.target]
     for t in targets:self.assertNotIn('.qpos',ast.unparse(t))
 def test_corner_geometry_is_invariant_to_member_end_reversal(self):
  import itertools
  from scipy.spatial.transform import Rotation
  from .environment import GOALS,YAWS
  from experiments.frame_assembly_v2.geometry import corner_errors
  centers=GOALS.copy();centers[0,:2]+=[.0004,-.0003]
  rotations=[Rotation.from_euler('z',y+.001).as_matrix() for y in YAWS]
  expected=corner_errors(centers,rotations)
  for flips in itertools.product((1,-1),repeat=4):
   actual=corner_errors(centers,[r@np.diag([s,s,1]) for r,s in zip(rotations,flips)])
   self.assertEqual(actual,expected)
  centers[2,1]+=.02
  self.assertFalse(all(c['ok'] for c in corner_errors(centers,rotations)))
 def test_astra_public_packet_excludes_accidental_object_state(self):
  from .live import packet
  robot={k:0 for k in ('time','tcp','rotation','joints','joint_velocity','gripper_command','shuttle_x','rack_slide0','rack_slide1')}
  robot.update(object_state='secret',contacts='secret',reward='secret',metrics='secret')
  p=packet(0,robot,'idle',0,None,{},[],[],[])
  self.assertNotIn('secret',str(p))
  self.assertEqual(set(p),{'observation_index','sim_time_s','phase','last_command_id','last_failure','robot','rgb','camera_calibration','abstract_fasteners','completed_commands'})
if __name__=='__main__':unittest.main()
