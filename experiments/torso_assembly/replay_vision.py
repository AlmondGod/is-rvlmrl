"""Offline RGB observer diagnosis on recorded motion; not a policy trial."""
import json,sys,numpy as np,mujoco
from pathlib import Path
from .environment import Env
from .frame_vision import MultiVision
folder=Path(sys.argv[1]);tr=json.loads((folder/'trace.json').read_text());rows=np.load(folder/'raw.npz')['rows'];e=Env.__new__(Env);e.m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));e.d=mujoco.MjData(e.m)
if '--low-front' in sys.argv:
 from scipy.spatial.transform import Rotation
 pos=np.array([0,-1.2,1.]);back=pos-[0,0,.15];back/=np.linalg.norm(back);right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right);cid=e.m.camera('camera3').id;e.m.cam_pos[cid]=pos;q=Rotation.from_matrix(np.column_stack((right,up,back))).as_quat();e.m.cam_quat[cid]=q[[3,0,1,2]];e.m.cam_fovy[cid]=55
ren=mujoco.Renderer(e.m,960,1280);v=MultiVision();holding=[False]*4;out=[]
for k,t in enumerate(tr):
 if len(sys.argv)>2 and t['time']<float(sys.argv[2]):continue
 stage=t['stage'].split(': ')
 if stage[0].startswith('frame ') and len(stage)==2:
  i=int(stage[0].split()[1]);phase=stage[1]
  if phase=='lift':holding[i]=True
  elif phase=='retreat':holding[i]=False
 if k%5:continue
 row=rows[k];e.d.qpos[:]=row[1:1+e.m.nq];mujoco.mj_forward(e.m,e.d);frames,cameras=e.images(ren);r=t['robot'];r={key:np.array(value) if isinstance(value,list) else value for key,value in r.items()};est=v.update(frames,cameras,r,holding)
 out.append({'time':t['time'],'xy_error_mm':[None if x is None else float(1000*np.linalg.norm(x['center'][:2]-np.array(t['metrics']['frame']['wood_evaluator_only'][i]['center'])[:2])) for i,x in enumerate(est)],'centers':[None if x is None else x['center'].tolist() for x in est],'diagnostics':v.diagnostics})
 if k%500==0:print(out[-1]['time'],out[-1]['xy_error_mm'],flush=True)
ren.close();(folder/'offline_vision_diagnostic.json').write_text(json.dumps(out,indent=2));print('FINAL',out[-1],flush=True)
