"""Calibrated marker-free RGB, four known face-up cuboids; no simulator state.
ROI association comes from the specified separated feeder layout, not object IDs.
Height uses silhouette scale at pickup, then encoders only after visual lift.
"""
import cv2,itertools,numpy as np
from scipy.optimize import least_squares
from experiments.astra_camera.perception import project,plane_points
from .environment import SIZES,STARTS,YAWS,ASSIGN,GOALS

class Vision:
 def __init__(self):self.previous=[None]*4;self.bias=[None]*4;self.last_robot=None;self.diagnostics=[];self.probe_reference=[None]*4;self.probe_residual=[None]*4
 def update(self,frames,cameras,robot,holding=(False,False,False,False)):
  im=frames[0].astype(float);cam=cameras[0];r,g,b=im[:,:,0],im[:,:,1],im[:,:,2];mask=((r>70)&(r>1.02*g)&(g>1.18*b)).astype(np.uint8);out=[];diag=[]
  T=np.array(cam['world_to_camera'])
  for i,size in enumerate(SIZES):
   old=self.previous[i];p=STARTS[i].copy() if old is None else old['center'].copy();yaw=YAWS[i] if old is None else old['yaw'];u=np.array([-np.sin(yaw),np.cos(yaw),0]);v=np.array([u[1],-u[0],0]);top=p+[0,0,size[2]/2]
   partial=holding[i] or (old is not None and np.linalg.norm(p[:2]-GOALS[i,:2])<.025)
   extent=size[1]/2-(.06 if partial else -.008)
   poly=np.array([top+u*a+v*b for a,b in [(-extent,-.035),(-extent,.035),(extent,.035),(extent,-.035)]]);uv,_=project(poly,cam);roi=np.zeros_like(mask);cv2.fillConvexPoly(roi,np.rint(uv).astype(np.int32),1);mm=mask&roi
   for j,other in enumerate(self.previous):
    if j==i or other is None:continue
    ou=np.array([-np.sin(other['yaw']),np.cos(other['yaw']),0]);ov=np.array([ou[1],-ou[0],0]);ot=other['center']+[0,0,SIZES[j,2]/2];ol=SIZES[j,1]/2+.010;ow=SIZES[j,0]/2+.012
    uv,_=project(np.array([ot+ou*a+ov*b for a,b in [(-ol,-ow),(-ol,ow),(ol,ow),(ol,-ow)]]),cam);cv2.fillConvexPoly(mm,np.rint(uv).astype(np.int32),0)
   yy,xx=np.where(mm)
   if len(xx)<140:out.append(None);diag.append('wood_pixels_missing');continue
   pts=np.c_[xx,yy].astype(np.float32);rect=cv2.minAreaRect(pts);box=cv2.boxPoints(rect);vv=np.roll(box,-1,axis=0)-box;axis=vv[np.argmax(np.linalg.norm(vv,axis=1))];axis/=np.linalg.norm(axis)
   dw=T[:3,:3].T@np.r_[axis,0];observed_yaw=np.arctan2(-dw[0],dw[1]);observed_yaw=yaw+(observed_yaw-yaw+np.pi/2)%np.pi-np.pi/2
   if abs(observed_yaw-yaw)>.12:out.append(None);diag.append('unexpected_shape_or_tilt');continue
   u=np.array([-np.sin(observed_yaw),np.cos(observed_yaw),0]);v=np.array([u[1],-u[0],0]);R=np.column_stack((v,u,[0,0,1]));corners=(np.array(list(itertools.product((-1,1),repeat=3)))*size/2)@R.T;perp=np.array([-axis[1],axis[0]]);basis=np.column_stack((axis,perp));coords=pts@basis;observed=np.r_[coords.min(0)-.5,coords.max(0)+.5]
   def residual(q):
    uv,_=project(corners+q,cam);cc=uv@basis;return np.r_[cc.min(0),cc.max(0)]-observed
   # Use the end away from the assigned arm. The forearm can occlude the
   # near end during approach; treating that silhouette truncation as motion
   # would pull the grasp target along the rail.
   far=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])[i]
   indices=[2 if np.dot(dw,far)>0 else 0,1,3]
   height=size[2]/2;visual_z=height;arm=ASSIGN[i]
   closing=robot['gripper_command'][arm]>100 and np.linalg.norm(robot['tcp'][arm,:2]-p[:2])<.08
   if not closing:self.probe_reference[i]=None
   if closing and old is not None and self.probe_reference[i] is None:
    self.probe_reference[i]=old['center'].copy();self.probe_residual[i]=residual(old['center'])[indices].copy()
   if holding[i] and self.last_robot is not None and old is not None:
    predicted=old['center']+robot['tcp'][arm]-np.array(old.get('tcp_at_measurement',self.last_robot['tcp'][arm]));height=predicted[2]
   elif closing and self.probe_reference[i] is not None:
    # A straight vertical probe holds XY constant. Infer rise from the visible
    # endpoint's perspective displacement, without reading simulated height.
    ref=self.probe_reference[i]
    height=float(least_squares(lambda z:residual(np.r_[ref[:2],z[0]])[indices]-self.probe_residual[i],np.array([p[2]]),bounds=(0,.25),max_nfev=12).x[0]);visual_z=height
   if partial and old is not None:
    if not holding[i]:predicted=old['center'].copy()
    # Longitudinal ends merge at butt joints. Use the verified grasp's encoder
    # displacement along the member; RGB still corrects transverse position
    # and yaw. This is a no-longitudinal-slip assumption, not hidden state.
    xy=least_squares(lambda q:np.r_[residual(np.r_[q,height])[[1,3]],1000*np.dot(q-predicted[:2],u[:2])],predicted[:2],max_nfev=10).x
   else:xy=least_squares(lambda q:residual(np.r_[q,height])[indices],p[:2],max_nfev=10).x
   center=np.r_[xy,height]
   if old is not None and np.linalg.norm(center-old['center'])>.04:out.append(None);diag.append('visual_jump');continue
   estimate={'center':center,'yaw':float(observed_yaw),'time':robot['time'],'visual_height_m':float(visual_z),'height_from_verified_grip':bool(holding[i]),'tcp_at_measurement':robot['tcp'][arm].copy(),'longitudinal_from_verified_grip':bool(holding[i]),'joined_endpoint_occluded':bool(partial and not holding[i]),'method':'RGB silhouette; calibrated camera; known cuboid/support and verified-grip priors'}
   out.append(estimate);diag.append({'pixels':len(xx),'residual_px':residual(center).tolist()})
  for i,e in enumerate(out):
   if e is not None:self.previous[i]=e
  self.last_robot={'tcp':robot['tcp'].copy()};self.diagnostics=diag;return out
