"""Marker-free calibrated RGB quadrilateral pose of known plywood geometry."""
import numpy as np,cv2
from .environment import PANEL_SIZE
class Vision:
 def __init__(self):self.last=None
 def update(self,frames,cameras):
  im=frames[0].astype(float);r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
  hsv=cv2.cvtColor(frames[0],cv2.COLOR_RGB2HSV);mask=((hsv[:,:,0]>=17)&(hsv[:,:,0]<=26)&(hsv[:,:,1]>25)&(hsv[:,:,1]<115)&(hsv[:,:,2]>65)).astype('uint8')*255
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contours=[c for c in contours if cv2.contourArea(c)>300]
  if not contours:return None
  # Fit the four outer edges even when the overhead arm hides one corner.
  lines=cv2.HoughLinesP(cv2.Canny(mask,50,150),1,np.pi/180,20,minLineLength=55,maxLineGap=20)
  if lines is None:return None
  pts=np.concatenate(contours)[:,0,:];mid=pts.mean(axis=0);horizontal=[];vertical=[]
  for x1,y1,x2,y2 in lines.reshape(-1,4):
   dx=x2-x1;dy=y2-y1
   l=np.cross([x1,y1,1.],[x2,y2,1.]);l/=max(np.linalg.norm(l[:2]),1e-9)
   if abs(dy)<abs(dx)*.25:horizontal.append((-(l[0]*mid[0]+l[2])/l[1],l))
   if abs(dx)<abs(dy)*.25:vertical.append((-(l[1]*mid[1]+l[2])/l[0],l))
  if len(horizontal)<2 or len(vertical)<2:return None
  self.debug={'horizontal':horizontal,'vertical':vertical}
  top=min(horizontal,key=lambda x:x[0])[1];bottom=max(horizontal,key=lambda x:x[0])[1];left=min(vertical,key=lambda x:x[0])[1];right=max(vertical,key=lambda x:x[0])[1]
  def cross(a,b):
   p=np.cross(a,b);return p[:2]/p[2]
  uv=np.array([cross(top,left),cross(top,right),cross(bottom,right),cross(bottom,left)])
  if cv2.contourArea(uv.astype('float32'))<8000:return None
  h=PANEL_SIZE[0]/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float);K=np.array(cameras[0]['K']);T=np.array(cameras[0]['world_to_camera'])
  ok,rv,tv=cv2.solvePnP(obj,uv,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
  projected=cv2.projectPoints(obj,rv,tv,K,None)[0].reshape(-1,2);rs=[rv];ts=[tv];errs=[[np.sqrt(np.mean((projected-uv)**2))]]
  if not ok:return None
  best=None
  for rv,tv,er in zip(rs,ts,errs):
   Rc=cv2.Rodrigues(rv)[0];R=T[:3,:3].T@Rc;p=T[:3,:3].T@(tv.ravel()-T[:3,3])-R[:,2]*PANEL_SIZE[2]/2
   self.debug.update(uv=uv,center=p,rotation=R)
   if R[2,2]<.9 or not -.05<p[2]<.6:continue
   score=float(er[0])+(0 if self.last is None else 10*np.linalg.norm(p-self.last['center']))
   item={'center':p,'rotation':R,'yaw':float(np.arctan2(R[1,0],R[0,0])),'reprojection_error_px':float(er[0]),'pixels':uv,'method':'RGB natural plywood outline + calibrated planar PnP'}
   if best is None or score<best[0]:best=(score,item)
  if best is None:return None
  self.last=best[1];return self.last

class HeldSheet:
 """Encoder propagation initialized from the observed RGB sheet pose.
 RGB supplies XY/yaw and verifies lift; this prior does not detect hidden slip.
 """
 def __init__(self):self.reference=None
 def begin(self,estimate,robot):
  c=np.array(estimate['center']).copy();R=np.array(estimate['rotation']).copy()
  self.reference=(np.array(robot['rotation'][:2]).copy(),(R.T@(np.array(robot['tcp'][:2])-c).T).T,R)
 def estimate(self,visual,robot):
  from scipy.spatial.transform import Rotation
  old,offsets,R0=self.reference;Rs=np.array([r@o.T for r,o in zip(robot['rotation'][:2],old)]);R=Rotation.from_matrix(Rs).mean().as_matrix()@R0
  if visual is not None:
   yaw=np.arctan2(R[1,0],R[0,0]);dy=visual['yaw']-yaw;c,s=np.cos(dy),np.sin(dy);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@R
  center=np.mean(np.array(robot['tcp'][:2])-(R@offsets.T).T,axis=0)
  if visual is not None:center[:2]=visual['center'][:2]
  return {'center':center,'rotation':R,'yaw':float(np.arctan2(R[1,0],R[0,0])),'method':'RGB XY/yaw, encoder-propagated height/tilt after RGB-initialized grasp','visual_available':visual is not None}
