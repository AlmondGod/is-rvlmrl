"""Calibrated marker-free RGB, four known face-up cuboids; no simulator state.
Initial association comes from global RGB discovery and known inventory sizes.
Subsequent ROIs follow observed poses, not reset coordinates or object IDs.
Height uses silhouette scale at pickup, then encoders only after visual lift.
"""
import cv2,itertools,numpy as np
from scipy.optimize import least_squares
from experiments.astra_camera.perception import project,plane_points
from .multiview_pose import fit_member
from .image_ops import mask_pixels,wood_mask
from .environment import SIZES,YAWS,ASSIGN,GOALS,FIXTURE_HEIGHT

class Vision:
 def __init__(self):self.previous=[None]*4;self.bias=[None]*4;self.last_robot=None;self.diagnostics=[];self.probe_reference=[None]*4;self.probe_residual=[None]*4;self.probe_tcp=[None]*4;self.probe_robot_R=[None]*4;self.rest_reference=[None]*4;self.probe_indices=[None]*4;self.released_prior=[None]*4
 def update(self,frames,cameras,robot,holding=(False,False,False,False),only=None):
  im=frames[0] if frames[0].dtype==np.uint8 else frames[0].astype(float);cam=cameras[0];mask=wood_mask(im,70,1.02);out=[];diag=[]
  T=np.array(cam['world_to_camera'])
  for i,size in enumerate(SIZES):
   if only is not None and i not in only:out.append(None);diag.append('not_requested');continue
   old=self.previous[i]
   if old is None:out.append(None);diag.append('not_visually_discovered');continue
   # Search from the observed release pose, never from accumulated occlusion
   # errors or a blueprint target. Every returned estimate still needs RGB.
   stationary=self.released_prior[i] if not holding[i] else None
   if stationary is not None:old=stationary
   p=old['center'].copy()
   carried_R=None
   if holding[i] and old is not None:
    a=ASSIGN[i];old_tcp=np.array(old['tcp_at_measurement']);old_robot_R=np.array(old.get('tcp_rotation_at_measurement',robot['rotation'][a]));delta_R=robot['rotation'][a]@old_robot_R.T
    p=robot['tcp'][a]+delta_R@(p-old_tcp)
    if self.probe_reference[i] is not None and self.probe_robot_R[i] is not None:
     # RGB verifies that the wood lifted. Propagation starts from the known
     # feeder support plane, not the noisy monocular lift-height fit.
     anchor=robot['tcp'][a]+robot['rotation'][a]@self.probe_robot_R[i].T@(self.probe_reference[i]-self.probe_tcp[i]);p[2]=anchor[2]
    old_R=np.array(old.get('rotation',[[np.cos(old['yaw']),-np.sin(old['yaw']),0],[np.sin(old['yaw']),np.cos(old['yaw']),0],[0,0,1.]]));carried_R=delta_R@old_R
   elif self.probe_tcp[i] is not None and robot['gripper_command'][ASSIGN[i]]>100:
    # Predict ONLY the image search window during the lift probe. The
    # subsequent height fit must still observe wood motion before holding.
    p[2]=self.probe_reference[i][2]+robot['tcp'][ASSIGN[i],2]-self.probe_tcp[i][2]
   yaw=old['yaw']
   if carried_R is not None:yaw=float(np.arctan2(-carried_R[0,1],carried_R[1,1]))
   u=np.array([-np.sin(yaw),np.cos(yaw),0]) if carried_R is None else carried_R[:,1];v=np.array([u[1],-u[0],0]) if carried_R is None else carried_R[:,0];top=p+(np.array([0,0,1.]) if carried_R is None else carried_R[:,2])*size[2]/2
   partial=holding[i] or stationary is not None or (old is not None and np.linalg.norm(p[:2]-GOALS[i,:2])<.025)
   extent=size[1]/2-(.06 if partial and not holding[i] else -.008)
   poly=np.array([top+u*a+v*b for a,b in [(-extent,-.035),(-extent,.035),(extent,.035),(extent,-.035)]]);uv,_=project(poly,cam);roi=np.zeros_like(mask);cv2.fillConvexPoly(roi,np.rint(uv).astype(np.int32),1);mm=mask&roi
   for j,other in enumerate(self.previous):
    if j==i or other is None:continue
    ou=np.array([-np.sin(other['yaw']),np.cos(other['yaw']),0]);ov=np.array([ou[1],-ou[0],0]);ot=other['center']+[0,0,SIZES[j,2]/2];ol=SIZES[j,1]/2+.010;ow=SIZES[j,0]/2+.012
    uv,_=project(np.array([ot+ou*a+ov*b for a,b in [(-ol,-ow),(-ol,ow),(ol,ow),(ol,-ow)]]),cam);cv2.fillConvexPoly(mm,np.rint(uv).astype(np.int32),0)
   yy,xx=mask_pixels(mm)
   if len(xx)<140:out.append(None);diag.append('wood_pixels_missing');continue
   pts=np.c_[xx,yy].astype(np.float32);rect=cv2.minAreaRect(pts);box=cv2.boxPoints(rect);vv=np.roll(box,-1,axis=0)-box;axis=vv[np.argmax(np.linalg.norm(vv,axis=1))];axis/=np.linalg.norm(axis)
   if carried_R is not None:
    endpoints,_=project(np.array([p-u*.2,p+u*.2]),cam);axis=endpoints[1]-endpoints[0];axis/=np.linalg.norm(axis);observed_yaw=yaw
   else:
    line_world=plane_points(np.array([pts.mean(axis=0),pts.mean(axis=0)+axis*100]),cam,p[2]+size[2]/2);dw=line_world[1]-line_world[0];dw/=np.linalg.norm(dw);observed_yaw=np.arctan2(-dw[0],dw[1]);observed_yaw=yaw+(observed_yaw-yaw+np.pi/2)%np.pi-np.pi/2
   if abs(observed_yaw-yaw)>.12:out.append(None);diag.append('unexpected_shape_or_tilt');continue
   u=np.array([-np.sin(observed_yaw),np.cos(observed_yaw),0]) if carried_R is None else carried_R[:,1];v=np.array([u[1],-u[0],0]) if carried_R is None else carried_R[:,0];R=np.column_stack((v,u,[0,0,1])) if carried_R is None else carried_R;corners=(np.array(list(itertools.product((-1,1),repeat=3)))*size/2)@R.T;perp=np.array([-axis[1],axis[0]]);basis=np.column_stack((axis,perp));coords=pts@basis;observed=np.r_[coords.min(0)-.5,coords.max(0)+.5]
   def residual(q):
    uv,_=project(corners+q,cam);cc=uv@basis;return np.r_[cc.min(0),cc.max(0)]-observed
   # Use the end away from the assigned arm. The forearm can occlude the
   # near end during approach; treating that silhouette truncation as motion
   # would pull the grasp target along the rail.
   far=np.array([[-1,0,0],[0,-1,0],[1,0,0],[0,-1,0]])[i]
   # Occlusion/shadow may erase one entire long edge. If the observed
   # width is too narrow, fit the edge consistent with the prior visible
   # pose rather than averaging a real edge with an occluder boundary.
   prior_residual=residual(p);edge_indices=[1,3]
   if prior_residual[3]-prior_residual[1]>2.:edge_indices=[min((1,3),key=lambda k:abs(prior_residual[k]))]
   # Select the endpoint consistent with the prior projection. The new
   # torso can occlude either end, unlike the old opposite-base fixture.
   endpoint=min((0,2),key=lambda k:abs(prior_residual[k]));indices=[endpoint]+edge_indices
   height=size[2]/2+FIXTURE_HEIGHT;visual_z=height;arm=ASSIGN[i]
   closing=robot['gripper_command'][arm]>100 and np.linalg.norm(robot['tcp'][arm,:2]-p[:2])<.08 and not any(holding[j] and ASSIGN[j]==arm for j in range(4) if j!=i)
   if not closing and not holding[i]:self.probe_reference[i]=None;self.probe_tcp[i]=None
   anchor=getattr(self,'pending_anchor',None)
   if anchor is not None and anchor['piece']==i:
    self.probe_reference[i]=np.array(anchor['center']).copy();self.probe_reference[i][2]=size[2]/2+FIXTURE_HEIGHT;self.probe_tcp[i]=np.array(anchor['tcp']).copy();self.probe_robot_R[i]=np.array(anchor['rotation']).copy();self.probe_indices[i]=[endpoint];self.probe_residual[i]=residual(self.probe_reference[i]).copy();self.pending_anchor=None
   if closing and old is not None and self.probe_reference[i] is None:
    self.probe_reference[i]=old['center'].copy();self.probe_reference[i][2]=size[2]/2+FIXTURE_HEIGHT;self.probe_tcp[i]=robot['tcp'][arm].copy();self.probe_robot_R[i]=robot['rotation'][arm].copy();self.probe_indices[i]=[endpoint];self.probe_residual[i]=residual(old['center']).copy()
   if holding[i] and self.last_robot is not None and old is not None:
    predicted=old['center']+robot['tcp'][arm]-np.array(old.get('tcp_at_measurement',self.last_robot['tcp'][arm]));height=p[2];predicted=p.copy()
   elif closing and self.probe_reference[i] is not None:
    # A straight vertical probe holds XY constant. Infer rise from the visible
    # endpoint's perspective displacement, without reading simulated height.
    ref=self.probe_reference[i];probe_indices=self.probe_indices[i]
    height=float(least_squares(lambda z:residual(np.r_[ref[:2],z[0]])[probe_indices]-self.probe_residual[i][probe_indices],np.array([p[2]]),bounds=(FIXTURE_HEIGHT,.40),max_nfev=12).x[0]);visual_z=height
   if partial and old is not None:
    if not holding[i]:predicted=old['center'].copy()
    # Longitudinal ends merge at butt joints. Use the verified grasp's encoder
    # displacement along the member; RGB still corrects transverse position
    # and yaw. This is a no-longitudinal-slip assumption, not hidden state.
    xy=least_squares(lambda q:np.r_[residual(np.r_[q,height])[edge_indices],1000*np.dot(q-predicted[:2],u[:2])],predicted[:2],max_nfev=10).x
   else:
    # Before a grasp, an occluding parked arm can hide both endpoints. A
    # previously RGB-localized feeder member has a stationary longitudinal
    # prior; use its visible long edge, not an occluder as a new endpoint.
    resting=self.rest_reference[i] is not None and not closing and np.linalg.norm(p[:2]-GOALS[i,:2])>.025
    if resting:
     ref=self.rest_reference[i]
     xy=least_squares(lambda q:np.r_[residual(np.r_[q,height])[edge_indices],1000*np.dot(q-ref[:2],u[:2])],p[:2],max_nfev=10).x
    else:xy=least_squares(lambda q:residual(np.r_[q,height])[indices],p[:2],max_nfev=10).x
   if closing and not holding[i] and self.probe_reference[i] is not None:
    # The probe is commanded vertically; the far-end image displacement
    # estimates height. Do not infer lateral travel from an occluded edge.
    xy=self.probe_reference[i][:2]+robot['tcp'][arm,:2]-self.probe_tcp[i][:2]
   if holding[i] and old is not None:
    correction=xy-predicted[:2]
    # Reject RGB edge jumps inconsistent with the verified grip; temporary
    # occlusion is not evidence that a held member teleported sideways.
    xy=predicted[:2]+(np.clip(correction,-.001,.001) if np.linalg.norm(correction)<.004 else 0.)
   center=np.r_[xy,height]
   if old is not None and np.linalg.norm(center-(p if holding[i] else old['center']))>.04:out.append(None);diag.append('visual_jump');continue
   if old is not None and not holding[i] and not closing and np.linalg.norm(old['center'][:2]-GOALS[i,:2])>.025 and self.rest_reference[i] is not None and np.linalg.norm(center[:2]-self.rest_reference[i][:2])>.008:
    out.append(None);diag.append('stationary_feeder_inconsistent_RGB');continue
   if stationary is not None and (np.linalg.norm(center[:2]-stationary['center'][:2])>.004 or abs(observed_yaw-stationary['yaw'])>.03):
    out.append(None);diag.append('released_pose_inconsistent_RGB');continue
   if self.rest_reference[i] is None:self.rest_reference[i]=center.copy()
   estimate={'center':center,'yaw':float(observed_yaw),'rotation':R.copy(),'time':robot['time'],'visual_height_m':float(visual_z),'height_from_verified_grip':bool(holding[i]),'tcp_at_measurement':robot['tcp'][arm].copy(),'tcp_rotation_at_measurement':robot['rotation'][arm].copy(),'longitudinal_from_verified_grip':bool(holding[i]),'joined_endpoint_occluded':bool(partial and not holding[i]),'method':'RGB silhouette; calibrated camera; known cuboid/support and verified-grip priors'}
   out.append(estimate);diag.append({'pixels':len(xx),'residual_px':residual(center).tolist()})
  for i,e in enumerate(out):
   if e is not None:self.previous[i]=e
  self.last_robot={'tcp':robot['tcp'].copy()};self.diagnostics=diag;return out


class MultiVision:
 """Calibrated RGB: top for side members, front for near/far members."""
 def __init__(self):self.top=Vision();self.front=Vision();self.low=Vision();self.diagnostics=[];self.previous=[None]*4;self.anchor_token=None;self.was_holding=[False]*4;self.released_prior=[None]*4;self.grasp_orientation=[None]*4
 def update(self,frames,cameras,robot,holding=(False,)*4,probe_anchor=None):
  if all(e is None for e in self.previous):
   from .discovery import discover
   initial,debug=discover(frames,cameras,robot);self.discovery_diagnostics=debug
   if initial is None:self.diagnostics=[debug]*4;return [None]*4
   self.previous=initial

  for i,held in enumerate(holding):
   if held and not self.was_holding[i] and self.previous[i] is not None:
    e=self.previous[i];self.grasp_orientation[i]=np.array(e['tcp_rotation_at_measurement']).T@np.array(e['rotation'])
   if self.was_holding[i] and not held and self.previous[i] is not None:
    reference=dict(self.previous[i]);reference['center']=reference['center'].copy()
    reference['center'][2]=SIZES[i,2]/2+FIXTURE_HEIGHT
    self.released_prior[i]=reference
   if held:self.released_prior[i]=None
   for view in (self.top,self.front,self.low):
    view.released_prior[i]=self.released_prior[i]
    if self.released_prior[i] is not None:view.rest_reference[i]=self.released_prior[i]['center'].copy()
  self.was_holding=list(holding)
  if probe_anchor is not None:
   token=(probe_anchor['piece'],tuple(probe_anchor['tcp']))
   if token!=self.anchor_token:
    for view in (self.top,self.front,self.low):view.pending_anchor=probe_anchor
    self.anchor_token=token
  # All cameras start each update from the same last accepted visual pose.
  # In particular, the lift-probe reference must not change when an arm
  # occludes the preferred camera and a different view takes over.
  for view in (self.top,self.front,self.low):
   view.previous=[None if e is None else dict(e) for e in self.previous]
  top=self.top.update(frames,cameras,robot,holding);front=self.front.update([frames[3]],[cameras[3]],robot,holding)
  for i in (0,1,3):
   known=front[i] if i==0 else top[i]
   if known is not None:self.low.previous[i]=known
  low=self.low.update([frames[2]],[cameras[2]],robot,holding,only=(2,));out=[];self.diagnostics=[]
  for i in range(4):
   candidates=[(top,0),(low,2),(front,3)] if i==2 else [(top,0),(front,3)]
   selected=next(((view[i],camera) for view,camera in candidates if view[i] is not None),None)
   e=None if selected is None else dict(selected[0],camera_index=selected[1]);out.append(e);self.diagnostics.append({'top':self.top.diagnostics[i],'front':self.front.diagnostics[i],'low':self.low.diagnostics[i]})
  # Reacquire an untouched feeder board from two current images when an
  # occluding arm defeats the fast single-view edge fit. Never substitute
  # its reset pose. The surveyed planar support is an explicit fixture prior.
  for i in range(4):
   prior=self.previous[i]
   if out[i] is None and not holding[i] and prior is not None and np.linalg.norm(prior['center'][:2]-GOALS[i,:2])>.1 and (probe_anchor is None or probe_anchor['piece']!=i):
    candidates=[dict(old) if cur is None and old is not None else cur for cur,old in zip(out,self.previous)]
    recovered=fit_member(frames,cameras,candidates,i,camera_indices=(0,3),fixed_support_z=FIXTURE_HEIGHT+SIZES[i,2]/2)
    if recovered is not None and np.linalg.norm(recovered['center'][:2]-prior['center'][:2])<.015:
     recovered.update(time=robot['time'],tcp_at_measurement=robot['tcp'][ASSIGN[i]].copy(),tcp_rotation_at_measurement=robot['rotation'][ASSIGN[i]].copy(),method='Two current RGB views; surveyed planar feeder support')
     out[i]=recovered
     for view in (self.top,self.front,self.low):view.rest_reference[i]=recovered['center'].copy()
  for i in (0,2):
   if holding[i] and out[i] is not None and np.linalg.norm(out[i]['center'][:2]-GOALS[i,:2])<.04 and abs(np.sin(out[i]['yaw']))>.98:
    if self.grasp_orientation[i] is not None:out[i]=dict(out[i],roll_prior_rotation=robot['rotation'][ASSIGN[i]]@self.grasp_orientation[i])
    fitted=fit_member(frames,cameras,out,i)
    out[i]=fitted
    if fitted is not None:
     fitted['camera_index']=0
     for view in (self.top,self.front,self.low):
      view.probe_reference[i]=fitted['center'].copy();view.probe_tcp[i]=robot['tcp'][ASSIGN[i]].copy();view.probe_robot_R[i]=robot['rotation'][ASSIGN[i]].copy()
   elif self.released_prior[i] is not None and out[i] is not None and np.linalg.norm(out[i]['center'][:2]-GOALS[i,:2])<.04:
    # The single-view bounding rectangle is biased by trimmed joint endpoints.
    # Re-measure the released front/rear rails from two current silhouettes.
    # Use the same surveyed planar support prior as the existing released-pose
    # estimator; never replace a failed fit with the desired pose or object truth.
    fitted=fit_member(frames,cameras,out,i,fixed_support_z=FIXTURE_HEIGHT+SIZES[i,2]/2)
    out[i]=fitted
    if fitted is not None:
     fitted.update(camera_index=0,method='Two current RGB silhouettes of released member; surveyed planar support')
  for i,e in enumerate(out):
   if e is not None:self.previous[i]=e
  return out

def probe_anchor(skills):
 """RGB pose and encoder sample captured after jaw closure, before lift."""
 if skills.state not in ('probe','lift','carry_route','align','lower','settle') or not hasattr(skills,'tcp_R_start'):return None
 return dict(piece=skills.piece,center=skills.probe_start,tcp=skills.tcp_start,rotation=skills.tcp_R_start)
