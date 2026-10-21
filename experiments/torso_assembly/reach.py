import xml.etree.ElementTree as ET
import numpy as np,mujoco
from experiments.frame_assembly_v2.environment import scene,Env,GOALS,STARTS,YAWS
from experiments.arm_corner.environment import down_rotation
root=ET.fromstring(scene());w=root.find('worldbody')
for a in range(3):
 b=w.find(f"body[@name='a{a}_link_base']");b.set('pos', ['-.34 .35 .6','.34 .35 .6','0 .35 .95'][a]);b.set('quat','0 1 0 0')
e=Env.__new__(Env);e.m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));e.d=mujoco.MjData(e.m);e.ikd=mujoco.MjData(e.m);m=e.m
e.ids=[np.array([m.joint(f'a{i}_joint{k}').id for k in range(1,8)]) for i in range(3)];e.qids=[m.jnt_qposadr[j] for j in e.ids];e.vids=[m.jnt_dofadr[j] for j in e.ids];e.sites=[m.site(f'a{i}_link_tcp').id for i in range(3)]
for a in range(3):e.d.qpos[e.qids[a]]=[0,-.4,0,1.1,0,.6,0]
for a in range(3):
 print('ARM',a,flush=True)
 for i,p in enumerate(STARTS):
  pos=p+np.array([0,0,.12]);q,err=e.ik(a,pos,down_rotation(YAWS[i]),250,True);print(i,np.round(pos,3),err,flush=True)
 for x,y in [(-.44,-.44),(.44,-.44),(.44,.44),(-.44,.44)]:
  q,err=e.ik(a,np.array([x,y,.2]),down_rotation(0),250,True);print('corner',x,y,err,flush=True)
