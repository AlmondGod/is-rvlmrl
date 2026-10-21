"""Nominal blueprint kinematic diagnostics, separate from camera performance."""
import json,sys,numpy as np
from pathlib import Path
from .environment import Env,STARTS,START_YAWS,GOALS,YAWS,ASSIGN,down_rotation,PANEL_GOAL,SIDE_R,OFFSETS
from .planner import Planner
E=Env();P=Planner(E);r=E.robot();out={'privileged_kinematic_diagnostic':True,'pickup_descent':[],'open_sheet_grasp':[]}
est=[dict(center=p,yaw=yaw) for p,yaw in zip(STARTS,START_YAWS)]
for i in range(4):
 try:path=P.plan_pick(ASSIGN[i],r,est,STARTS[i]+[0,0,.20],down_rotation(START_YAWS[i]));out['pickup_descent'].append({'member':i,'clear':True,'waypoints':len(path)})
 except Exception as ex:out['pickup_descent'].append({'member':i,'clear':False,'error':repr(ex)})
r.update(shuttle_x=0.,rack_slide0=.2,rack_slide1=.2);est=[dict(center=p,yaw=yaw) for p,yaw in zip(GOALS,YAWS)];sheet={'center':PANEL_GOAL,'rotation':np.eye(3)}
for a in (0,1):
 try:path=P.plan(a,r,est,PANEL_GOAL+OFFSETS[a],SIDE_R[a],sheet);out['open_sheet_grasp'].append({'arm':a,'clear':True,'waypoints':len(path)})
 except Exception as ex:out['open_sheet_grasp'].append({'arm':a,'clear':False,'error':repr(ex)})
print(json.dumps(out,indent=2));Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/torso_assembly/preflight.json').write_text(json.dumps(out,indent=2))
