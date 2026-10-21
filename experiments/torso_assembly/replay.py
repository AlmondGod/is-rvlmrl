"""Render saved execution states without stepping or changing its physics.
Brighter overview lighting is presentation-only; policy RGB is preserved.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,cv2
from .trace_io import iter_trace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--speed',type=float,default=4.);ap.add_argument('--start',type=float);ap.add_argument('--output-name',default='continuous_4x.mp4');args=ap.parse_args();folder=Path(args.run);raw=np.load(folder/'raw.npz');rows=raw['rows'];nq=int(raw['nq']);nv=int(raw['nv']);tr=[{'stage':x['stage']} for x in iter_trace(folder/'trace.json')];result=json.loads((folder/'result.json').read_text());accepted=json.loads((folder/'flip_audit.json').read_text())['passes'] if (folder/'flip_audit.json').exists() else result['success'];m=mujoco.MjModel.from_xml_string((folder/'scene.xml').read_text());d=mujoco.MjData(m)
 m.vis.headlight.ambient[:]=.65;m.vis.headlight.diffuse[:]=.55
 # Main camera prioritizes the work surface; a separate inset shows the torso.
 inset_camera='torso_context' if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'torso_context')>=0 else 'camera0'
 renderer=mujoco.Renderer(m,960,1280);inset_renderer=mujoco.Renderer(m,240,320);fps=30;writer=io.get_writer(folder/args.output_name,fps=fps,codec='libx264',quality=8)
 for index,t in enumerate(np.arange(rows[0,0] if args.start is None else max(rows[0,0],args.start),rows[-1,0],args.speed/fps)):
  k=min(int(np.searchsorted(rows[:,0],t)),len(rows)-1);row=rows[k];d.time=float(row[0]);d.qpos[:]=row[1:1+nq];d.qvel[:]=row[1+nq:1+nq+nv];mujoco.mj_forward(m,d);renderer.update_scene(d,camera='torso_overview');view=renderer.render().copy()
  if index==0:io.imwrite(folder/'shared_torso_overview.png',view)
  canvas=np.zeros((1120,1280,3),np.uint8);canvas[:960]=view;inset_renderer.update_scene(d,camera=inset_camera);canvas[10:250,950:1270]=inset_renderer.render();canvas[960:]=15;count=sum(ev['time']<=d.time for ev in result['events']);label='PRIVILEGED DIAGNOSTIC' if result['privileged_diagnostic'] else ('ACTUAL ASTRA + RGB SKILLS' if result.get('actual_astra') else 'CAMERA FEEDBACK BASELINE')
  lines=[f'ONE TORSO PILLAR | {label} REPLAY | {args.speed:g}x | {d.time:.1f}s',f'{tr[k]["stage"]} | Visits: {count}/8 | Run outcome: '+('PASS' if accepted else 'INCOMPLETE'), ('Breakable nail surrogate; provisional strengths; simplified insertion' if result.get('fastener_physics') is not None else 'Uninterrupted run; recorded physical poses; abstract nails do not create bonds')]
  for j,line in enumerate(lines):cv2.putText(canvas,line,(18,1000+35*j),cv2.FONT_HERSHEY_SIMPLEX,.66,(245,245,245),2)
  writer.append_data(canvas)
 writer.close();renderer.update_scene(d,camera='torso_overview');io.imwrite(folder/'finished_overview.png',renderer.render().copy());renderer.close();inset_renderer.close()
if __name__=='__main__':main()
