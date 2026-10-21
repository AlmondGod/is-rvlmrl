"""Collision-checked nailer paths from encoders, known fixture and RGB sheet pose."""
import numpy as np
from experiments.frame_assembly_v2.plan_tool import Planner
from experiments.frame_assembly_v2.environment import GOALS,YAWS
from experiments.arm_corner.environment import down_rotation
from .environment import FIXTURE_HEIGHT,PANEL_GOAL,PANEL_SIZE
class PanelPlanner(Planner):
 def __init__(self,e):
  super().__init__(e)
  self.rail_q=int(self.m.joint('overhead_z').qposadr[0]);self.panel_q=int(self.m.joint('panel_free').qposadr[0]);self.rack_q=[int(self.m.joint(f'rack_slide{i}').qposadr[0]) for i in range(2)]
 def plan(self,robot,estimate,carriage,rack,p,R):
  for arm in range(3):self.d.qpos[self.qids[arm]]=robot['joints'][arm]
  self.d.qpos[self.rail_q]=carriage
  for q in self.rack_q:self.d.qpos[q]=rack
  for i in range(4):
   q=int(self.m.joint(f'wood{i}_free').qposadr[0]);self.d.qpos[q:q+3]=GOALS[i]+[0,0,FIXTURE_HEIGHT]
  from scipy.spatial.transform import Rotation
  self.d.qpos[self.panel_q:self.panel_q+3]=estimate['center'];q=Rotation.from_matrix(estimate['rotation']).as_quat();self.d.qpos[self.panel_q+3:self.panel_q+7]=q[[3,0,1,2]]
  return self.solve(p,R,np.array(robot['joints'][2]))[1]
def nail_target(member):
 p=GOALS[member].copy();p[2]=PANEL_GOAL[2]+PANEL_SIZE[2]/2+.003;return p,down_rotation(0)
