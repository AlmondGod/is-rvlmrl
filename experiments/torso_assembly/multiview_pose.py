"""Calibrated RGB silhouette profile fit for a known rigid member.
Inputs contain images, calibration and previous image/encoder estimates only.
No renderer, physics state, contact labels, depth buffers or object IDs.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiments.astra_camera.perception import project
from experiments.box_assembly.design import beveled_vertices
from .environment import SIZES
from .image_ops import mask_pixels

DIRECTIONS=np.c_[np.cos(np.arange(24)*2*np.pi/24),np.sin(np.arange(24)*2*np.pi/24)]

def measurement(image,camera,estimates,index):
 e=estimates[index];p=np.array(e['center']);R=np.array(e['rotation']);v=np.array(beveled_vertices(SIZES[index],.0005))
 uv,_=project(v@R.T+p,camera);roi=np.zeros(image.shape[:2],np.uint8)
 cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),1);target_region=cv2.dilate(roi,np.ones((7,7),np.uint8));roi=cv2.dilate(roi,np.ones((25,25),np.uint8));excluded=np.zeros_like(roi);T=np.array(camera['world_to_camera'])
 im=image if image.dtype==np.uint8 else image.astype(float);r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
 mask=((r>35)&(r>1.05*g)&(g>1.18*b)).astype(np.uint8)&roi
 for j,other in enumerate(estimates):
  if j==index or other is None:continue
  y=other['yaw'];oR=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1.]])
  uv,_=project(np.array(beveled_vertices(SIZES[j],.0005))@oR.T+other['center'],camera)
  omit=np.zeros_like(mask);cv2.fillConvexPoly(omit,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
  if (T@np.r_[p,1.])[2]<(T@np.r_[other['center'],1.])[2]-.02:omit=omit&(1-target_region)
  excluded|=omit;mask[omit!=0]=0
 n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
 if n<=1:return None
 cutoff=max(150,.10*float(stats[1:,cv2.CC_STAT_AREA].max()))
 keep=np.where(stats[:,cv2.CC_STAT_AREA]>=cutoff)[0];keep=keep[keep!=0];mask=np.isin(labels,keep).astype(np.uint8)
 yy,xx=mask_pixels(mask)
 if len(xx)<400:return None
 # Interior cross-sections avoid merged joint endpoints and central fingers.
 ends,_=project(np.array([p-R[:,1]*.2,p+R[:,1]*.2]),camera)
 axis=ends[1]-ends[0];axis/=np.linalg.norm(axis);basis=np.column_stack((axis,[-axis[1],axis[0]]));coords=np.c_[xx,yy]@basis
 lo,hi=coords[:,0].min(),coords[:,0].max();samples=lo+(hi-lo)*np.array([.15,.25,.75,.85]);values=[];visible_samples=[]
 for t in samples:
  cross=coords[abs(coords[:,0]-t)<1.5,1]
  if len(cross)<8:continue
  visible_samples.append(t)
  values.extend([float(cross.min()-.5),float(cross.max()+.5)])
 if len(visible_samples)<3:return None
 return basis,np.array(visible_samples),np.r_[lo-.5,hi+.5,values]


def fit_member(frames,cameras,estimates,index,camera_indices=(0,2),fixed_support_z=None):
 e=estimates[index]
 if e is None:return None
 p=np.array(e['center']);R=np.array(e['rotation'])
 if fixed_support_z is not None:p[2]=fixed_support_z;R=Rotation.from_euler('z',e['yaw']).as_matrix()
 vertices=np.array(beveled_vertices(SIZES[index],.0005));obs=[]
 for k in camera_indices:
  item=measurement(frames[k],cameras[k],estimates,index)
  if item is not None:obs.append((cameras[k],item))
 if len(obs)!=2:return None
 def residual(x):
  from scipy.spatial import ConvexHull
  world=vertices@(Rotation.from_rotvec(x[3:]).as_matrix()@R).T+p+x[:3];res=[]
  for c,(basis,samples,observed) in obs:
   uv=project(world,c)[0]@basis;h=uv[ConvexHull(uv).vertices];end=np.roll(h,-1,axis=0);values=[uv[:,0].min(),uv[:,0].max()]
   for t in samples:
    delta=end[:,0]-h[:,0];alpha=(t-h[:,0])/np.where(abs(delta)>1e-10,delta,1e-10);valid=(alpha>=0)&(alpha<=1)&(abs(delta)>1e-10);cross=h[:,1]+alpha*(end[:,1]-h[:,1]);cross=cross[valid]
    if len(cross)<2:values.extend([observed[2],observed[3]+100.])
    else:values.extend([cross.min(),cross.max()])
   res.extend(np.array(values)-observed)
  return np.array(res)
 def objective(x):
  # A rectangular silhouette weakly constrains roll about the long axis.
  # Regularize that ambiguous component to the verified-grasp encoder prior.
  prior=np.array(e.get('roll_prior_rotation',R));candidate=Rotation.from_rotvec(x[3:]).as_matrix()@R
  roll_error=np.dot(Rotation.from_matrix(candidate@prior.T).as_rotvec(),prior[:,1])
  return np.r_[residual(x),roll_error/.015]
 if fixed_support_z is None:
  fit=least_squares(objective,np.zeros(6),bounds=([-0.04]*3+[-.10]*3,[.04]*3+[.10]*3),x_scale=[.01]*3+[.03]*3,loss='soft_l1',f_scale=1.,max_nfev=40)
 else:
  # Explicit surveyed planar-support prior for an untouched feeder member.
  expand=lambda q:np.array([q[0],q[1],0.,0.,0.,q[2]])
  fit=least_squares(lambda q:residual(expand(q)),np.zeros(3),bounds=([-.04,-.04,-.1],[.04,.04,.1]),x_scale=[.01,.01,.03],loss='soft_l1',f_scale=1.,max_nfev=40)
  fit.x=expand(fit.x)
 errors=abs(residual(fit.x));pos=p+fit.x[:3];rot=Rotation.from_rotvec(fit.x[3:]).as_matrix()@R
 fit_member.last_debug={'center':pos.tolist(),'rotation':rot.tolist(),'median_px':float(np.median(errors)),'p90_px':float(np.percentile(errors,90)),'max_px':float(max(errors)),'correction':fit.x.tolist(),'nfev':fit.nfev}
 if np.median(errors)>1.5 or np.percentile(errors,90)>3. or np.linalg.norm(fit.x[:3])>.035:return None
 return dict(e,center=pos,rotation=rot,yaw=float(np.arctan2(-rot[0,1],rot[1,1])),height_from_verified_grip=False,
             visual_height_m=float(pos[2]),full_multiview_silhouette=True,reprojection_median_px=float(np.median(errors)),reprojection_p90_px=float(np.percentile(errors,90)),method='Two calibrated RGB silhouettes, known beveled cuboid dimensions, six-degree-of-freedom fit')
