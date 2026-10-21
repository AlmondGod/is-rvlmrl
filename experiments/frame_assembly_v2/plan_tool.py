"""Offline robot-kinematic path generation; nominal blueprint obstacles, no rollouts.
Joint paths use actual motor actuation at runtime, never direct qpos assignment.
"""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from .environment import Env,GOALS,YAWS
from .skills import nail_pose

class Planner:
 def __init__(self,e):
  self.m=e.m;self.d=mujoco.MjData(e.m)
  for arm in range(3):self.d.qpos[e.qids[arm]]=e.robot()['joints'][arm] # robot encoders only
  self.qids=e.qids;self.vids=e.vids;self.allids=e.ids;self.sites=e.sites;self.home=[e.d.qpos[q].copy() for q in e.qids];self.arm=2
  self.qa=e.qids[2];self.va=e.vids[2];self.ids=e.ids[2];self.sid=e.sites[2];self.rng=np.random.default_rng(731)
  self.lo=self.m.jnt_range[self.ids,0]+.03;self.hi=self.m.jnt_range[self.ids,1]-.03
  for i in range(4):
   q=int(self.m.joint(f'wood{i}_free').qposadr[0]);self.d.qpos[q:q+3]=GOALS[i];self.d.qpos[q+3:q+7]=[np.cos(YAWS[i]/2),0,0,np.sin(YAWS[i]/2)]
 def plan_home(self,arm,robot,est):
  # Deployment inputs: joint encoders and camera estimates only.
  self.arm=arm;self.qa=self.qids[arm];self.va=self.vids[arm];self.ids=self.allids[arm];self.sid=self.sites[arm];self.lo=self.m.jnt_range[self.ids,0]+.03;self.hi=self.m.jnt_range[self.ids,1]-.03
  for a in range(3):self.d.qpos[self.qids[a]]=robot['joints'][a]
  for i,e in enumerate(est):
   if e is None:continue
   q=int(self.m.joint(f'wood{i}_free').qposadr[0]);self.d.qpos[q:q+3]=e['center'];self.d.qpos[q+3:q+7]=[np.cos(e['yaw']/2),0,0,np.sin(e['yaw']/2)]
  start=np.array(robot['joints'][arm]);goal=self.home[arm]
  if not self.valid(start):raise ValueError('Parking start configuration has a modeled collision')
  if not self.valid(goal):raise ValueError('Parking goal has a modeled collision')
  path=[start,goal] if self.edge(start,goal) else self.rrt(start,goal)
  if path is None:raise ValueError('No collision-free parking route')
  return np.array(path)
 def valid(self,q):
  self.d.qpos[self.qa]=q;mujoco.mj_forward(self.m,self.d)
  for c in self.d.contact:
   names=[self.m.geom(c.geom1).name or '',self.m.geom(c.geom2).name or '']
   if any(n.startswith(f'a{self.arm}_') or (self.arm==2 and n.startswith('nailer')) for n in names) and c.dist<-.000005:return False
  return True
 def edge(self,a,b):
  return all(self.valid(a+(b-a)*t) for t in np.linspace(0,1,max(2,int(np.max(abs(b-a))/.04)+2)))
 def solve(self,p,R,start,goal_filter=None):
  jp=np.zeros((3,self.m.nv));jr=jp.copy();candidates=[]
  for attempt in range(100):
   q=start.copy() if attempt==0 else self.rng.uniform(np.maximum(self.lo,-3.1),np.minimum(self.hi,3.1))
   for it in range(180):
    self.d.qpos[self.qa]=q;mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d);cur=self.d.site_xmat[self.sid].reshape(3,3);err=np.r_[p-self.d.site_xpos[self.sid],.3*Rotation.from_matrix(R@cur.T).as_rotvec()]
    if np.linalg.norm(err)<.00002:break
    mujoco.mj_jacSite(self.m,self.d,jp,jr,self.sid);J=np.vstack((jp[:,self.va],.3*jr[:,self.va]));dq=J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),err);q=np.clip(q+np.clip(dq,-.15,.15),self.lo,self.hi)
   if np.linalg.norm(err)<.0001 and self.valid(q) and (goal_filter is None or goal_filter(q)):
    candidates.append(q.copy())
    if self.edge(start,q):return q,[start,q]
  for q in sorted(candidates,key=lambda q:np.linalg.norm(q-start)):
   path=self.rrt(start,q)
   if path is not None:return q,path
  raise RuntimeError('No collision-free tool approach found')
 def rrt(self,start,goal):
  if not self.valid(start):raise ValueError('Invalid start')
  trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
  def extend(tree,target):
   nodes,parents=tree;i=int(np.argmin([np.linalg.norm(n-target) for n in nodes]));delta=target-nodes[i];new=nodes[i]+delta*min(1.,.3/max(np.linalg.norm(delta),1e-9))
   if not self.edge(nodes[i],new):return None
   nodes.append(new);parents.append(i);return len(nodes)-1
  def chain(tree,i):
   path=[]
   while i>=0:path.append(tree[0][i]);i=tree[1][i]
   return path[::-1]
  for iteration in range(2500):
   a=iteration%2;b=1-a;sample=self.rng.uniform(np.maximum(self.lo,-3.1),np.minimum(self.hi,3.1)) if self.rng.random()>.2 else trees[b][0][0]
   ia=extend(trees[a],sample)
   if ia is None:continue
   target=trees[a][0][ia]
   for _ in range(60):
    ib=extend(trees[b],target)
    if ib is None:break
    if np.linalg.norm(trees[b][0][ib]-target)<1e-6:
     ca=chain(trees[a],ia);cb=chain(trees[b],ib);path=ca+cb[::-1] if a==0 else cb+ca[::-1]
     for _ in range(100):
      if len(path)<3:break
      i,j=sorted(self.rng.choice(len(path),2,replace=False))
      if j>i+1 and self.edge(path[i],path[j]):path=path[:i+1]+path[j:]
     return path
  return None

def main():
 e=Env();pl=Planner(e);home=e.d.qpos[e.qids[2]].copy();out={'description':'Offline nominal-blueprint/self-collision checked joint routes. Executed by force-limited motors. No wood teleportation.','home':home.tolist(),'routes':{}}
 for c in range(4):
  tip,R=nail_pose(c);q,path=pl.solve(tip-R[:,2]*.14,R,home);out['routes'][str(c)]=[x.tolist() for x in path];print(c,len(path),'waypoints',flush=True)
 Path('experiments/frame_assembly_v2/tool_routes.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
