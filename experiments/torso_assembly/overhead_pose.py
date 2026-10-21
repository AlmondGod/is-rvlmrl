"""Known-cuboid translation from calibrated RGB silhouette; no simulator state."""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from experiments.astra_camera.perception import project,plane_points
from .environment import SIZES

SIGNS=np.array(list(itertools.product((-1,1),repeat=3)))

def fit_member(image,camera,estimates,index):
 e=estimates[index]
 if e is None:return None
 p=np.array(e['center']);R=np.array(e['rotation']);corners=(SIGNS*SIZES[index]/2)@R.T
 uv,_=project(corners+p,camera);roi=np.zeros(image.shape[:2],np.uint8)
 cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
 roi=cv2.dilate(roi,np.ones((11,11),np.uint8))
 im=image.astype(float);r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
 mask=((r>35)&(r>1.02*g)&(g>1.18*b)).astype(np.uint8)&roi
 for j,other in enumerate(estimates):
  if j==index or other is None:continue
  y=other['yaw'];oR=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1.]])
  uv,_=project((SIGNS*SIZES[j]/2)@oR.T+other['center'],camera)
  cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),0)
 yy,xx=np.where(mask)
 if len(xx)<300:return None
 pts=np.c_[xx,yy].astype(np.float32);rect=cv2.boxPoints(cv2.minAreaRect(pts));edges=np.roll(rect,-1,axis=0)-rect;axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];axis/=np.linalg.norm(axis)
 mid=pts.mean(0);world=plane_points(np.array([mid,mid+axis*100]),camera,p[2]+SIZES[index,2]/2);u=world[1]-world[0];yaw=float(np.arctan2(-u[0],u[1]));yaw=e['yaw']+(yaw-e['yaw']+np.pi/2)%np.pi-np.pi/2
 R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
 corners=(SIGNS*SIZES[index]/2)@R.T
 basis=np.column_stack((axis,[-axis[1],axis[0]]));coords=np.c_[xx,yy]@basis
 observed=np.r_[coords.min(0)-.5,coords.max(0)+.5]
 def residual(q):
  uv,_=project(corners+q,camera);c=uv@basis;return np.r_[c.min(0),c.max(0)]-observed
 fit=least_squares(residual,p,bounds=(p-.04,p+.04),max_nfev=25)
 error=float(np.max(abs(residual(fit.x))))
 fit_member.last_debug={'error_px':error,'center':fit.x.tolist(),'prior':p.tolist(),'residual':residual(fit.x).tolist(),'pixels':len(xx)}
 if error>1.5 or np.linalg.norm(fit.x-p)>.035:return None
 return dict(e,center=fit.x,rotation=R,yaw=yaw,visual_height_m=float(fit.x[2]),height_from_verified_grip=False,
             full_overhead_silhouette=True,reprojection_max_px=error,
             method='RGB calibrated complete cuboid silhouette translation; known dimensions, image-estimated yaw and face-up orientation assumption')
