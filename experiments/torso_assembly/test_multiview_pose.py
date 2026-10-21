"""Image-only regression for tilted members and contact-adjacent clutter."""
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.astra_camera.perception import project
from experiments.box_assembly.design import beveled_vertices
from .environment import SIZES
from .multiview_pose import fit_member

class PoseTests(unittest.TestCase):
 def fixture(self):
  cameras=[None]*4;frames=[None]*4
  for i,pos in ((0,np.array([.35,-.2,1.25])),(2,np.array([0.,-1.2,1.]))):
   look=np.array([0.,.2,.15]);z=look-pos;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x);R=np.array([x,y,z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@pos
   cameras[i]={'K':[[800,0,639.5],[0,800,479.5],[0,0,1]],'world_to_camera':T.tolist()};frames[i]=np.full((960,1280,3),30,np.uint8)
  center=np.array([.002,.438,.181]);rot=Rotation.from_euler('xyz',[.008,-.017,-1.56]).as_matrix();priorR=Rotation.from_euler('z',-np.pi/2).as_matrix()
  est=[None,{'center':np.array([-.438,0,.152]),'rotation':np.eye(3),'yaw':0.},dict(center=center+[.004,-.006,.007],rotation=priorR,yaw=-np.pi/2),None]
  for i in (0,2):
   for j,p,R in ((1,est[1]['center'],est[1]['rotation']),(2,center,rot)):
    uv,_=project(np.array(beveled_vertices(SIZES[j],.0005))@R.T+p,cameras[i]);cv2.fillConvexPoly(frames[i],cv2.convexHull(np.rint(uv).astype(np.int32)),(150,95,50))
   uv,_=project(np.array([center]),cameras[i]);u,v=np.rint(uv[0]).astype(int);cv2.rectangle(frames[i],(u-22,v-55),(u+22,v+55),(80,85,90),-1)
  return frames,cameras,est,center,rot
 def test_tilted_member_with_neighbor_and_finger_occlusion(self):
  frames,cameras,est,p,R=self.fixture();fit=fit_member(frames,cameras,est,2)
  self.assertIsNotNone(fit)
  self.assertLess(np.linalg.norm(fit['center']-p),.0025)
  self.assertLess(Rotation.from_matrix(fit['rotation']@R.T).magnitude(),np.deg2rad(1.))
 def test_repeated_fits_use_fixed_grasp_roll_reference(self):
  frames,cameras,est,p,R=self.fixture();reference=est[2]['rotation'].copy()
  est[2]['rotation']=Rotation.from_rotvec(reference[:,1]*.06).as_matrix()@reference
  est[2]['roll_prior_rotation']=reference
  for _ in range(5):
   fitted=fit_member(frames,cameras,est,2);self.assertIsNotNone(fitted);est[2]=fitted
  self.assertLess(Rotation.from_matrix(est[2]['rotation']@R.T).magnitude(),np.deg2rad(1.))
  self.assertLess(np.linalg.norm(est[2]['center']-p),.0025)
 def test_planar_reacquisition_uses_images_and_surveyed_support(self):
  frames,cameras,est,_,_=self.fixture();p=np.array([.002,.438,.152]);R=Rotation.from_euler('z',-np.pi/2).as_matrix()
  est=[None,None,dict(center=p+[.006,-.004,.008],rotation=R,yaw=-np.pi/2),None]
  for ci in (0,2):
   frames[ci][:]=30
   uv,_=project(np.array(beveled_vertices(SIZES[2],.0005))@R.T+p,cameras[ci]);cv2.fillConvexPoly(frames[ci],cv2.convexHull(np.rint(uv).astype(np.int32)),(150,95,50))
  fit=fit_member(frames,cameras,est,2,fixed_support_z=.152)
  self.assertIsNotNone(fit);self.assertLess(np.linalg.norm(fit['center']-p),.0025);self.assertAlmostEqual(fit['center'][2],.152)
  frames[2][:]=30
  self.assertIsNone(fit_member(frames,cameras,est,2,fixed_support_z=.152))
 def test_held_neighbor_is_not_mistaken_for_new_grasp(self):
  from .frame_vision import Vision
  frames,cameras,est,p,R=self.fixture();observer=Vision();observer.previous=est
  robot=dict(time=0.,tcp=np.array([[0,0,.5],est[2]['center']+[.01,0,.06]]),rotation=np.array([np.eye(3)]*2),gripper_command=[0,180])
  observer.update(frames,cameras,robot,holding=(False,False,False,True),only=(2,))
  self.assertIsNone(observer.probe_reference[2])
 def test_missing_second_view_rejects_pose(self):
  frames,cameras,est,_,_=self.fixture();frames[2][:]=30
  self.assertIsNone(fit_member(frames,cameras,est,2))
 def test_released_rail_refinement_measures_displacement_and_requires_two_views(self):
  from .frame_vision import MultiVision
  frames,cameras,_,_,_=self.fixture()
  p=np.array([.002,.441,.152]);R=Rotation.from_euler('z',-np.pi/2+.004).as_matrix()
  prior=dict(center=p+[.001,-.002,0],rotation=R,yaw=-np.pi/2+.011,
             tcp_at_measurement=np.array([.9,.65,.65]),tcp_rotation_at_measurement=np.eye(3))
  for ci in (0,2):
   frames[ci][:]=30
   uv,_=project(np.array(beveled_vertices(SIZES[2],.0005))@R.T+p,cameras[ci])
   cv2.fillConvexPoly(frames[ci],cv2.convexHull(np.rint(uv).astype(np.int32)),(150,95,50))
  frames[3]=frames[2].copy();cameras[3]=cameras[2]
  robot=dict(time=2.,tcp=np.array([[-.9,.65,.65],[.9,.65,.65]]),
             rotation=np.array([np.eye(3)]*2),gripper_command=[0,0])
  observer=MultiVision();observer.previous[2]=prior;observer.released_prior[2]=prior
  fitted=observer.update(frames,cameras,robot)[2]
  self.assertIsNotNone(fitted)
  self.assertIn('released member',fitted['method'])
  self.assertLess(np.linalg.norm(fitted['center']-p),.0025)
  self.assertGreater(fitted['center'][1],.439) # must not snap to blueprint .438
  frames[2][:]=30
  self.assertIsNone(observer.update(frames,cameras,robot)[2])
 def test_front_member_with_one_profile_occluded(self):
  frames,cameras,_,_,_=self.fixture();center=np.array([.001,-.438,.20]);R=Rotation.from_euler('xyz',[.003,-.009,np.pi/2]).as_matrix()
  for ci in (0,2):
   pos=np.linalg.inv(cameras[ci]['world_to_camera'])[:3,3];z=np.array([0.,0.,.15])-pos;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x);C=np.array([x,y,z]);T=np.eye(4);T[:3,:3]=C;T[:3,3]=-C@pos;cameras[ci]['world_to_camera']=T.tolist()
  priorR=Rotation.from_euler('z',np.pi/2).as_matrix();est=[dict(center=center+[.002,-.005,.006],rotation=priorR,yaw=np.pi/2),None,None,None]
  for ci in (0,2):
   frames[ci][:]=30
   uv,_=project(np.array(beveled_vertices(SIZES[0],.0005))@R.T+center,cameras[ci]);cv2.fillConvexPoly(frames[ci],cv2.convexHull(np.rint(uv).astype(np.int32)),(150,95,50))
  uv,_=project(np.array([center+R[:,1]*SIZES[0,1]*.25]),cameras[0]);u,v=np.rint(uv[0]).astype(int)
  cv2.rectangle(frames[0],(u-18,v-50),(u+18,v+50),(80,85,90),-1)
  fit=fit_member(frames,cameras,est,0);self.assertIsNotNone(fit);self.assertLess(np.linalg.norm(fit['center']-center),.003) # half the6mm task gate; rasterized occluded geometry
 def test_released_member_recovers_after_occlusion_and_rejects_out_of_view_shift(self):
  from .frame_vision import Vision
  frames,cameras,_,_,_=self.fixture();cam=cameras[0]
  center=np.array([.001,-.438,.152]);R=Rotation.from_euler('z',np.pi/2).as_matrix()
  prior=dict(center=center.copy(),rotation=R,yaw=np.pi/2)
  observer=Vision();observer.previous[0]=prior;observer.released_prior[0]=prior;observer.rest_reference[0]=center.copy()
  robot=dict(time=0.,tcp=np.array([[-.9,.65,.65],[.9,.65,.65]]),rotation=np.array([np.eye(3)]*2),gripper_command=[0,0])
  def picture(p):
   im=np.full((960,1280,3),30,np.uint8)
   uv,_=project(np.array(beveled_vertices(SIZES[0],.0005))@R.T+p,cam)
   cv2.fillConvexPoly(im,cv2.convexHull(np.rint(uv).astype(np.int32)),(150,95,50));return im
  clean=picture(center);first=observer.update([clean],[cam],robot,only=(0,))[0]
  self.assertIsNotNone(first)
  for n in range(8):
   robot['time']+=.1
   self.assertIsNone(observer.update([picture(center+[0,.080,0])],[cam],robot,only=(0,))[0])
  self.assertIsNone(observer.update([np.full_like(clean,30)],[cam],robot,only=(0,))[0])
  recovered=observer.update([clean],[cam],robot,only=(0,))[0]
  self.assertIsNotNone(recovered);np.testing.assert_allclose(recovered['center'],first['center'],atol=1e-8)
  np.testing.assert_allclose(observer.released_prior[0]['center'],center)
if __name__=='__main__':unittest.main()
