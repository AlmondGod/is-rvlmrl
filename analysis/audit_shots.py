"""Evaluator-only reconstruction of insertion gates from logged pre-shot poses."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from experiments.box_assembly.fastening import Frame
from experiments.torso_assembly.fasteners import ray_interval,NailParameters
from experiments.torso_assembly.environment import SIZES,PANEL_SIZE,CORNER_PAIRS
p=argparse.ArgumentParser();p.add_argument('trial');a=p.parse_args();folder=Path(a.trial)/'evaluator';P=NailParameters();out=[];prev=None
with gzip.open(folder/'trace.jsonl.gz','rt') as f:
 for row in map(json.loads,f):
  if prev and row['metrics']['fasteners']['inserted']>prev['metrics']['fasteners']['inserted']:
   stage=row['stage'];kind='frame' if stage.startswith('frame_nail') else 'panel';index=int(stage.split()[1].rstrip(':'))
   metrics=prev['metrics'];objects=metrics['frame']['wood_evaluator_only']+[{'center':metrics['center_evaluator_only'],'rotation':metrics['rotation_evaluator_only']}]
   origin=np.array(prev['robot']['tcp'][2]);axis=np.array(prev['robot']['rotation'][2])[:,2];hits=[]
   for body,(o,size) in enumerate(zip(objects,list(SIZES)+[PANEL_SIZE])):
    h=ray_interval(origin,axis,Frame(np.array(o['center']),np.array(o['rotation'])),size,P.length_m)
    if h is not None:hits.append({'object':body,'entry_m':h[0],'exit_m':h[1]})
   hits.sort(key=lambda x:x['entry_m']);expected=set(CORNER_PAIRS[index]) if kind=='frame' else {4,index};reason=None
   if len(hits)<2 or {h['object'] for h in hits[:2]}!=expected:reason='ray_does_not_cross_intended_materials'
   elif hits[0]['entry_m']>P.max_standoff_m:reason='tool_too_far_from_surface'
   elif hits[1]['entry_m']-hits[0]['exit_m']>P.max_gap_m:reason='joint_gap_too_large'
   elif hits[1]['exit_m']-hits[1]['entry_m']<P.min_embed_m:reason='insufficient_receiver_embed'
   recorded_miss=row['metrics']['fasteners']['missed']>prev['metrics']['fasteners']['missed']
   target=np.array(prev['command'][0][2]);tracking=float(np.linalg.norm(origin-target))*1000
   panel_pose=objects[4];normal=np.array(panel_pose['rotation'])[:,2];center=np.array(panel_pose['center']);surface_z=center[2]+(PANEL_SIZE[2]/2-normal[0]*(target[0]-center[0])-normal[1]*(target[1]-center[1]))/normal[2]
   out.append({'tool_position_tracking_error_mm':tracking,'commanded_tip_clearance_above_true_panel_mm':float(target[2]-surface_z)*1000 if kind=='panel' else None,'nominal_panel_tip_clearance_mm':3.0 if kind=='panel' else None,'kind':kind,'index':index,'pre_shot_time_s':prev['time'],'recorded_miss':recorded_miss,'reconstructed_reason':reason,'agrees_with_record':recorded_miss==(reason is not None),'hits':hits,'expected_objects':sorted(expected),'receiver_embed_mm':(hits[1]['exit_m']-hits[1]['entry_m'])*1000 if len(hits)>1 else None,'max_standoff_mm':P.max_standoff_m*1000,'max_gap_mm':P.max_gap_m*1000,'minimum_embed_mm':P.min_embed_m*1000})
  prev=row
report={'method':'Replay ray gates against pre-shot logged poses, independently verified against raw-state reconstruction. No controller input, re-simulation, parameter change or trial retry. Surrogate insertion only; no physical validation.','shots':out,'all_classifications_agree':all(x['agrees_with_record'] for x in out)}
(folder/'shot_gate_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
