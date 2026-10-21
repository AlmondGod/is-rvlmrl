"""Global RGB discovery. No reset coordinates, object IDs, seed or simulator.
Known inventory, calibrated cameras and surveyed support height are inputs/priors.
Role assignment uses observed side and measured length, never feeder lane index.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.astra_camera.perception import plane_points
from .environment import SIZES,FIXTURE_HEIGHT,ASSIGN
from .multiview_pose import fit_member

def discover(frames,cameras,robot):
 im=frames[0].astype(float);r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
 mask=((r>70)&(r>1.05*g)&(g>1.18*b)).astype(np.uint8)
 n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8);candidates=[]
 for label in range(1,n):
  if stats[label,cv2.CC_STAT_AREA]<1000:continue
  yy,xx=np.where(labels==label);points=np.c_[xx,yy].astype(np.float32)
  rect=cv2.minAreaRect(points);s=sorted(rect[1])
  if s[0]<3 or s[1]/s[0]<5:continue
  world=plane_points(points,cameras[0],FIXTURE_HEIGHT+SIZES[0,2]);rect=cv2.minAreaRect(world[:,:2].astype(np.float32));box=cv2.boxPoints(rect);edges=np.roll(box,-1,axis=0)-box;lengths=np.linalg.norm(edges,axis=1);axis=edges[np.argmax(lengths)]/max(lengths)
  if axis[1]<0:axis=-axis
  if not .70<max(lengths)<1.08:continue
  center=np.r_[np.array(rect[0]),FIXTURE_HEIGHT+SIZES[0,2]/2];yaw=float(np.arctan2(-axis[0],axis[1]));candidates.append(dict(center=center,yaw=yaw,length=float(max(lengths))))
 out=[None]*4
 # Left/right is a robot reach allocation, not a known object coordinate.
 for side,roles in ((-1,(0,1)),(1,(2,3))):
  found=[c for c in candidates if np.sign(c['center'][0])==side]
  if len(found)!=2:return None,dict(reason='need_two_visually_detected_members_per_side',candidates=candidates)
  found.sort(key=lambda c:c['length'],reverse=True)
  for i,c in zip(roles,found):
   a=ASSIGN[i];out[i]=dict(center=c['center'],yaw=c['yaw'],rotation=Rotation.from_euler('z',c['yaw']).as_matrix(),time=robot['time'],tcp_at_measurement=robot['tcp'][a].copy(),tcp_rotation_at_measurement=robot['rotation'][a].copy(),method='global RGB components; known inventory length/observed-side association')
 fits=[]
 for i in range(4):
  fit=fit_member(frames,cameras,out,i,camera_indices=(0,3))
  if fit is None:return None,dict(reason='discovery_multiview_rejected',piece=i,candidates=candidates)
  fits.append(fit)
 return fits,dict(reason='four_globally_discovered_RGB_members',candidates=candidates)
