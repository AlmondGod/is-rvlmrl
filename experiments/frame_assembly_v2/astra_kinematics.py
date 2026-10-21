import xml.etree.ElementTree as ET
import numpy as np
class R:
 def __init__(self,m): self.m=m
 def apply(self,v):return self.m@v
 def __mul__(self,r):return R(self.m@r.m)
 @staticmethod
 def identity():return R(np.eye(3))
 @staticmethod
 def from_rotvec(v):
  t=np.linalg.norm(v)
  if t<1e-12:return R.identity()
  x,y,z=v/t;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
  return R(np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K))
 @staticmethod
 def from_quat(q):
  q=q/np.linalg.norm(q); x,y,z,w=q
  return R(np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]))
root=ET.parse('third_party/wuji-description/hand2/hand2_2/body/mjcf/right.xml').getroot()
names=[a.get('joint') for a in root.find('actuator')]
limits=np.array([list(map(float,a.get('ctrlrange').split())) for a in root.find('actuator')])
def fk(q):
 out={}
 def walk(b,p,r):
  p=p+r.apply(np.fromstring(b.get('pos','0 0 0'),sep=' '))
  wxyz=np.fromstring(b.get('quat','1 0 0 0'),sep=' ');r=r*R.from_quat(wxyz[[1,2,3,0]])
  for j in b.findall('joint'):
   if j.get('name') in names:r=r*R.from_rotvec(np.fromstring(j.get('axis','0 0 1'),sep=' ')*q[names.index(j.get('name'))])
  for s in b.findall('site'):
   out[s.get('name')]=p+r.apply(np.fromstring(s.get('pos','0 0 0'),sep=' '))
  for child in b.findall('body'):walk(child,p,r)
 walk(root.find('worldbody/body'),np.zeros(3),R.identity())
 return out
if __name__=='__main__':
 for q in [np.zeros(20),np.array([0,0,0,0]+[.5,0,.5,.3]*4)]:
  print(q, {k: v.round(4).tolist() for k,v in fk(q).items() if 'tip' in k})
def ik(finger, target, start=None):
 idx=['thumb','index_finger','middle_finger','ring_finger','pinky'].index(finger)*4
 site='r_'+finger+'_tip'; best=None
 rng=np.random.default_rng(10)
 for trial in range(12):
  q=np.zeros(20);q[idx:idx+4]=start if trial==0 and start is not None else rng.uniform(limits[idx:idx+4,0],limits[idx:idx+4,1])
  for _ in range(65):
   p=fk(q)[site];err=target-p
   if np.linalg.norm(err)<.0001:break
   J=np.zeros((3,4))
   for j in range(4):
    qq=q.copy();qq[idx+j]+=.0001;J[:,j]=(fk(qq)[site]-p)/.0001
   dq=J.T@np.linalg.solve(J@J.T+np.eye(3)*1e-5,err)
   q[idx:idx+4]=np.clip(q[idx:idx+4]+np.clip(dq,-.15,.15),limits[idx:idx+4,0],limits[idx:idx+4,1])
  error=np.linalg.norm(fk(q)[site]-target)
  score=error+ .00001*np.linalg.norm(q)
  if best is None or score<best[0]:best=(score,q[idx:idx+4].copy(),fk(q)[site])
 return best
