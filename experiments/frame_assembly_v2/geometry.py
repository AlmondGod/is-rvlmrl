"""Shared geometry calculation; caller supplies measured or evaluator poses."""
import numpy as np
from .environment import SIZES,W,CORNER_PAIRS,CORNER_NAMES

def corner_errors(centers,rotations):
 # Rectangular members have no distinguished +Y end. Canonicalize the
 # in-plane axes before choosing inner faces and endpoints; a 180-degree
 # placement is physically identical and must not swap the evaluated joint.
 rotations=[np.array(r).copy() for r in rotations]
 for i,r in enumerate(rotations):
  reference=np.array([-1.,0.,0.]) if i in (0,2) else np.array([0.,1.,0.])
  if np.dot(r[:,1],reference)<0:r[:,:2]*=-1
 out=[]
 for c,(h,v) in enumerate(CORNER_PAIRS):
  sx=-1 if c in (0,3) else 1;sy=-1 if c in (0,1) else 1
  ph,pv=np.array(centers[h]),np.array(centers[v]);rh,rv=np.array(rotations[h]),np.array(rotations[v]);inner=ph-sy*rh[:,0]*W/2;end=pv+sy*rv[:,1]*SIZES[v,1]/2
  gap=np.dot(inner-end,sy*rh[:,0]);hend=ph-sx*rh[:,1]*SIZES[h,1]/2;outer=end+sx*rv[:,0]*W/2;flush=np.dot(outer-hend,-sx*rh[:,1]);angle=np.degrees(np.arcsin(np.clip(abs(np.dot(rh[:,1],rv[:,1])),0,1)));height=abs(ph[2]-pv[2])
  out.append({'corner':c,'name':CORNER_NAMES[c],'gap_m':float(gap),'outer_flush_m':float(flush),'square_error_deg':float(angle),'height_difference_m':float(height),'ok':bool(-.001<=gap<=.006 and abs(flush)<.006 and angle<1 and height<.003)})
 return out

def visual_corners(est):
 centers=[e['center'] for e in est];rot=[]
 for e in est:
  t=e['yaw'];rot.append(np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]]))
 return corner_errors(centers,rot)
