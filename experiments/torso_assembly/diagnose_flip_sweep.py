"""Privileged offline rigid-payload collision diagnostic; never a physics trial."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from .environment import Env
from experiments.arm_corner.run import clean

def main():
 ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--time',type=float,default=48.9);ap.add_argument('--height',type=float,default=.75);ap.add_argument('--shoulder-half-span',type=float,default=.34);a=ap.parse_args();p=Path(a.source);raw=np.load(p/'raw.npz')['rows'];row=raw[np.argmin(abs(raw[:,0]-a.time))];e=Env(fixture_half_width=.38,shoulder_half_span=a.shoulder_half_span);m,d=e.m,e.d
 def restore():
  d.qpos[:]=row[1:1+m.nq];mujoco.mj_forward(m,d);e.qtarget=[d.qpos[q].copy() for q in e.qids];e.ik_cache={}
 restore();source_model=mujoco.MjModel.from_xml_path(str(p/'scene.xml'));source_data=mujoco.MjData(source_model);source_data.qpos[:]=row[1:1+source_model.nq];mujoco.mj_forward(source_model,source_data);source_sites=[source_model.site(f'a{i}_link_tcp').id for i in range(3)];base_r=source_data.site_xmat[source_sites].reshape(3,3,3).copy();base_tcp=source_data.site_xpos[source_sites].copy();pivot=np.mean(base_tcp[:2],axis=0);pieces=e.woods+[e.panel];positions=[d.xpos[b].copy() for b in pieces];rotations=[d.xmat[b].reshape(3,3).copy() for b in pieces];qids=[int(m.joint(name+'_free').qposadr[0]) for name in ['wood0','wood1','wood2','wood3','panel']];results=[]
 for y in (-.1,-.2,-.3,-.4,-.5):
  restore();shift=np.array([0,y-pivot[1],a.height-pivot[2]]);worst=0.;hits=set();max_ik=0.;first=None
  path=[(u*shift,0.) for u in np.linspace(0,1,31)]+[(shift,t) for t in np.linspace(0,np.pi,181)]
  for delta,t in path:
   R=Rotation.from_rotvec([t,0,0]).as_matrix()
   for arm in (0,1):
    q,err=e.ik(arm,pivot+delta+R@(base_tcp[arm]-pivot),R@base_r[arm],iterations=100);e.qtarget[arm]=q;d.qpos[e.qids[arm]]=q;max_ik=max(max_ik,err)
   for q,x,rot in zip(qids,positions,rotations):
    d.qpos[q:q+3]=pivot+delta+R@(x-pivot);quat=Rotation.from_matrix(R@rot).as_quat();d.qpos[q+3:q+7]=quat[[3,0,1,2]]
   mujoco.mj_forward(m,d)
   for c in d.contact:
    ns=[m.geom(g).name or '' for g in (c.geom1,c.geom2)];work=lambda n:n.startswith('wood') or n=='panel_geom'
    if t>0 and c.dist<-.0005 and any(work(n) for n in ns) and not all(work(n) for n in ns) and not any('finger_pad' in n for n in ns):
     hits.add(tuple(ns));worst=max(worst,-c.dist)
     if first is None:first=float(np.degrees(t))
  results.append(dict(center_y=y,maximum_ik_error=max_ik,first_collision_deg=first,maximum_collision_m=worst,pairs=list(hits)))
 print(json.dumps(clean(results),indent=2));(p/f'sweep_diagnostic_z{a.height:.2f}_span{a.shoulder_half_span:.2f}.json').write_text(json.dumps(clean(results),indent=2))
if __name__=='__main__':main()
