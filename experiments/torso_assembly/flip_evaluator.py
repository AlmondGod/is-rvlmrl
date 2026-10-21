"""Evaluator-only retention measurements; never passed to the controller."""
import numpy as np
from scipy.spatial.transform import Rotation
from .environment import SIZES,PANEL_SIZE,FIXTURE_HEIGHT

def reference(e):
 R=e.d.xmat[e.panel].reshape(3,3)
 return [(R.T@(e.d.xpos[b]-e.d.xpos[e.panel]),R.T@e.d.xmat[b].reshape(3,3)) for b in e.woods]

def measure(e,ref):
 R=e.d.xmat[e.panel].reshape(3,3)
 bottom=min(float(e.d.xpos[b,2]-abs(e.d.xmat[b].reshape(3,3)[2])@(s/2)) for b,s in zip(e.woods+[e.panel],list(SIZES)+[PANEL_SIZE]))
 return dict(panel_up_dot=float(R[2,2]),position_errors_m=[float(np.linalg.norm(R.T@(e.d.xpos[b]-e.d.xpos[e.panel])-q[0])) for b,q in zip(e.woods,ref)],angle_errors_rad=[float(Rotation.from_matrix(R.T@e.d.xmat[b].reshape(3,3)@q[1].T).magnitude()) for b,q in zip(e.woods,ref)],minimum_height_above_fixture_m=bottom-FIXTURE_HEIGHT)

def passes(value,intact,held_inverted):
 return bool(held_inverted and intact==8 and value['panel_up_dot']<-.98 and max(value['position_errors_m'])<.005 and max(value['angle_errors_rad'])<np.deg2rad(3) and value['minimum_height_above_fixture_m']>.02)
