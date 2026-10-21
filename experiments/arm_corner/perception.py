"""Marker-free calibrated overhead RGB silhouette geometry, not simulator IDs/depth.
Known face-up rectangular members, known dimensions and tabletop at acquisition.
Body centers are inferred from endpoint extent and camera geometry. Height can be
fused with measured gripper motion AFTER a visually verified pickup.
"""
import cv2,numpy as np
import itertools
from scipy.optimize import least_squares
from experiments.astra_camera.perception import project,plane_points
from .environment import SIZE

class Vision:
 def __init__(self):self.previous=[None,None];self.bias=[None,None];self.last_robot=None;self.last_mask=None;self.diagnostics=[]
 def update(self,frames,cameras,robot,holding=(False,False)):
  im=frames[0];cam=cameras[0];x=im.astype(float);r,g,b=x[:,:,0],x[:,:,1],x[:,:,2];mask=((r>70)&(r>1.02*g)&(g>1.18*b)).astype(np.uint8);self.last_mask=mask.copy();out=[];diag=[]
  K=np.array(cam['K']);T=np.array(cam['world_to_camera']);origin=-T[:3,:3].T@T[:3,3]
  partial=False
  if all(e is not None for e in self.previous):
   aa,bb=self.previous;ua=np.array([-np.sin(aa['yaw']),np.cos(aa['yaw']),0]);ub=np.array([np.sin(bb['yaw']),-np.cos(bb['yaw']),0]);corner=bb['center']+ub*(SIZE[1]/2-SIZE[0]/2)
   partial=np.linalg.norm((aa['center']+ua*SIZE[1]/2-corner)[:2])<.15
  for i in range(2):
   mm=mask.copy();old=self.previous[i]
   if old is None:
    ys,xs=np.where(mm);world=plane_points(np.c_[xs,ys],cam,.064);ok=(world[:,0]>.1)&(world[:,1]<.28) if i==0 else (world[:,0]<.3)&(world[:,1]>.18);mm[:]=0;mm[ys[ok],xs[ok]]=1
   else:
    u=np.array([-np.sin(old['yaw']),np.cos(old['yaw']),0]);v=np.array([u[1],-u[0],0]);p=old['center']+np.array([0,0,SIZE[2]/2]);corners=np.array([p+u*a+v*b for a,b in [(-.49,-.05),(-.49,.05),(.49,.05),(.49,-.05)]]);uv,z=project(corners,cam);roi=np.zeros_like(mm);cv2.fillConvexPoly(roi,np.rint(uv).astype(np.int32),1);mm&=roi
   other=self.previous[1-i]
   if other is not None:
    u=np.array([-np.sin(other['yaw']),np.cos(other['yaw']),0]);v=np.array([u[1],-u[0],0]);p=other['center']+np.array([0,0,SIZE[2]/2]);corners=np.array([p+u*a+v*b for a,b in [(-SIZE[1]/2,-.019),( -SIZE[1]/2,.019),(SIZE[1]/2,.019),(SIZE[1]/2,-.019)]]);uv,z=project(corners,cam);cv2.fillConvexPoly(mm,np.rint(uv).astype(np.int32),0)
   yy,xx=np.where(mm)
   if len(xx)<150:out.append(None);diag.append({'reason':'too_few_wood_pixels'});continue
   pts=np.c_[xx,yy].astype(np.float32)
   if partial and old is not None:
    u=np.array([-np.sin(old['yaw']),np.cos(old['yaw']),0]);pp,_=project(np.array([old['center']+[0,0,.032],old['center']+[0,0,.032]+u*.1]),cam);aa=pp[1]-pp[0];aa/=np.linalg.norm(aa);along=(pts-pp[0])@aa;pts=pts[along< -30] if i==0 else pts[along>30]
    if len(pts)<100:out.append(None);diag.append({'reason':'far_end_occluded'});continue
   rect=cv2.minAreaRect(pts);box=cv2.boxPoints(rect);vectors=np.roll(box,-1,axis=0)-box;v=vectors[np.argmax(np.linalg.norm(vectors,axis=1))];length=float(np.linalg.norm(v));axis=v/max(length,1e-8)
   if (i==0 and axis[1]>0) or (i==1 and axis[0]>0):axis=-axis
   delta_world=T[:3,:3].T@np.array([axis[0],axis[1],0]);yaw=float(np.arctan2(-delta_world[0],delta_world[1]));u=np.array([-np.sin(yaw),np.cos(yaw),0]);v=np.array([u[1],-u[0],0]);R=np.column_stack((v,u,[0,0,1]));corners=(np.array(list(itertools.product((-1,1),repeat=3)))*SIZE/2)@R.T
   perp=np.array([-axis[1],axis[0]]);coords=pts@np.column_stack((axis,perp));observed=np.r_[coords.min(axis=0)-.5,coords.max(axis=0)+.5]
   guess=plane_points(np.array([rect[0]]),cam,.064)[0];guess[2]=.032 if old is None else old['center'][2]
   def residual(p):
    uv,_=project(corners+p,cam);cc=uv@np.column_stack((axis,perp));return np.r_[cc.min(axis=0),cc.max(axis=0)]-observed
   fitted=old['center'].copy() if partial and old is not None else least_squares(residual,guess,max_nfev=20,diff_step=1e-4).x
   if self.bias[i] is None:self.bias[i]=fitted[2]-.032
   raw_height=fitted[2]-self.bias[i];height=raw_height
   if partial and old is not None:
    height=old['center'][2]+(robot['tcp'][i,2]-self.last_robot['tcp'][i,2] if holding[i] else 0.)
    if not holding[i]:height=.032
   elif old is not None and holding[i] and self.last_robot is not None:
    predicted=old['center'][2]+robot['tcp'][i,2]-self.last_robot['tcp'][i,2]
    if abs(height-predicted)>.025:out.append(None);diag.append({'reason':'shape_height_disagrees_with_grip','visual_z':height,'prediction':predicted});continue
    height=.2*height+.8*predicted
   # Refit image translation at the calibrated/fused height.
   indices=([0,1,3] if i==0 else [1,2,3]) if partial else [0,1,2,3]
   fitted2=least_squares(lambda xy:residual(np.r_[xy,height])[indices],fitted[:2],max_nfev=10).x;center=np.r_[fitted2,height]
   estimate={'center':center,'yaw':yaw,'time':robot['time'],'method':'RGB cuboid outline / known dimensions / face-up prior','visual_height_m':raw_height,'end_occluded':partial,'height_from_grip_during_end_occlusion':bool(partial and holding[i])}
   if old is not None and np.linalg.norm(center-old['center'])>.06:out.append(None);diag.append({'reason':'implausible_visual_jump'});continue
   out.append(estimate);diag.append({'axis':axis.tolist(),'observed':observed.tolist(),'predicted_residual':residual(center).tolist(),'rect':rect,'length_px':length,'wood_pixels':len(xx),'raw_height_m':raw_height})
  self.diagnostics=diag
  for i,e in enumerate(out):
   if e is not None:self.previous[i]=e
  self.last_robot={'tcp':robot['tcp'].copy()};return out
