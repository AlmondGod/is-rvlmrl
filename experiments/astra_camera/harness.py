"""Cloud-only camera/proprio control, independently scored by hidden sim state.

No controller code is given the environment or any object/contact state.
The process boundary is an information-flow boundary, not an adversarial sandbox.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET
from unittest.mock import patch

import imageio.v2 as imageio
import mujoco
import numpy as np

from experiments.frame_assembly import environment as physics
from experiments.astra_closed_loop.harness import ControlEnv, dump
from .worker import serve, OBSERVATION_KEYS

OUT = physics.ROOT/'outputs/frame_assembly/astra_camera'
WIDTH, HEIGHT = 640, 480
CAMERA_POSITIONS = ((.85,-.55,.65), (.90,.60,.45))
LOOKAT = np.array([.40,0.,.08])


def require_cloud():
    if platform.system() != 'Linux' or os.environ.get('PHYLA_CLOUD_RUN') != '1':
        raise RuntimeError('Run on the authorized Linux cloud host with PHYLA_CLOUD_RUN=1; local simulation is disabled')


def make_environment(seed):
    build_xml = physics.build_xml
    def camera_xml(config=physics.Config()):
        root = ET.fromstring(build_xml(config))
        world = root.find('worldbody')
        for i, position in enumerate(CAMERA_POSITIONS):
            position = np.asarray(position)
            backward = position-LOOKAT; backward /= np.linalg.norm(backward)
            right = np.cross([0.,0.,1.], backward); right /= np.linalg.norm(right)
            up = np.cross(backward, right)
            ET.SubElement(world, 'camera', name=f'vision_{i}', pos=physics.fmt(position),
                xyaxes=physics.fmt(np.r_[right,up]), fovy='45')
        return ET.tostring(root, encoding='unicode')
    # Only fixed cameras are added. Physical parameters are the frozen firm_v1.
    with patch.object(physics, 'build_xml', camera_xml):
        return ControlEnv(seed, contact_profile='firm_v1')


def calibration(e):
    cameras=[]
    for i in range(len(CAMERA_POSITIONS)):
        cid=e.sim.model.camera(f'vision_{i}').id
        camera_to_world=e.sim.data.cam_xmat[cid].reshape(3,3)
        R=np.diag([1.,-1.,-1.])@camera_to_world.T
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@e.sim.data.cam_xpos[cid]
        focal=.5*HEIGHT/np.tan(np.deg2rad(e.sim.model.cam_fovy[cid])/2)
        K=np.array([[focal,0,(WIDTH-1)/2],[0,focal,(HEIGHT-1)/2],[0,0,1.]])
        cameras.append({'K':K,'world_to_camera':T})
    return cameras


def configuration(e):
    # All model-derived information here concerns robot hardware, not the object.
    return {'pose_prior':json.loads((physics.ROOT/'outputs/frame_assembly/rl/base_params.json').read_text()),
        'rail_size':[.04,.30,.02],'tabletop_height':0.,
        'target_search_bounds':[[.25,-.22,-.02],[.56,.22,.55]],
        'workspace_bounds':[[-.6,-.47,.04],[.6,.47,.75]],
        'motor_kp':e.sim.model.actuator_gainprm[:,0].copy(),
        'actuator_force_limits':e.sim.model.actuator_forcerange.copy(),
        'control_dt':.02}


def observation(e, frames, cameras, image_time):
    s,m,d=e.sim,e.sim.model,e.sim.data
    packet={'time':float(d.time),'image_time':image_time,'rgb':frames,'cameras':cameras,
        'wrist_position':d.xpos[e.wid].copy(),'wrist_quaternion':d.xquat[e.wid].copy(),
        'finger_q':d.qpos[s.finger_qadr].copy(),'finger_qvel':d.qvel[m.jnt_dofadr[s.joint_ids]].copy(),
        'wrist_target':d.mocap_pos[s.wrist_id].copy(),'wrist_target_quaternion':d.mocap_quat[s.wrist_id].copy(),
        'finger_targets':d.ctrl.copy(),'joint_limits':m.actuator_ctrlrange.copy()}
    assert set(packet)==OBSERVATION_KEYS
    return packet


def serializable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serializable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serializable(v) for v in x]
    return x


def episode(seed,out,condition='rgb',max_steps=1500):
    require_cloud()
    e=make_environment(seed);cams=calibration(e);cfg=configuration(e)
    folder=Path(out)/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'config.json',serializable(cfg));dump(folder/'camera_calibration.json',serializable(cams))
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    process=ctx.Process(target=serve,args=(child,cfg));process.start();child.close()
    action_rows=[];proprio_rows=[];logs=[];latencies=[];image_times=[];error=None
    frame0=None;start=time.monotonic();info=e.summary()
    option=mujoco.MjvOption();option.geomgroup[2]=0;option.sitegroup[:]=0
    try:
        if not parent.poll(30):raise TimeoutError('Controller did not start')
        ready=parent.recv()
        if not ready.get('ready'):raise RuntimeError(ready)
        with mujoco.Renderer(e.sim.model,height=HEIGHT,width=WIDTH) as renderer, imageio.get_writer(
                str(folder/'camera_views.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=1) as video:
            for step in range(max_steps):
                if step%5==0:
                    views=[]
                    for i in range(2):
                        renderer.update_scene(e.sim.data,camera=f'vision_{i}',scene_option=option)
                        views.append(renderer.render().copy())
                    actual=np.asarray(views);image_time=float(e.sim.data.time)
                    video.append_data(np.concatenate(list(actual),axis=1));image_times.append(image_time)
                    if frame0 is None:
                        frame0=actual.copy()
                        imageio.imwrite(folder/'initial_cameras.png',np.concatenate(list(actual),axis=1))
                    frames=np.zeros_like(actual) if condition=='black' else frame0.copy() if condition=='frozen' else actual
                packet=observation(e,frames,cams,image_time)
                tick=time.monotonic();parent.send(packet)
                if not parent.poll(15):raise TimeoutError('Controller exceeded 15-second command deadline')
                reply=parent.recv();latencies.append(time.monotonic()-tick)
                if 'error' in reply:raise RuntimeError(reply['error'])
                p,q,f=(np.asarray(a,dtype=float) for a in reply['action'])
                # Actions/robot feedback are separate from hidden evaluator logs.
                action_rows.append(np.r_[p,q,f]);proprio_rows.append(np.r_[packet['wrist_position'],packet['wrist_quaternion'],packet['finger_q'],packet['finger_qvel']])
                _,bad,info=e.apply((p,q,f))
                logs.append({'time':float(e.sim.data.time),'controller':serializable(reply.get('status',{})),
                    'evaluator_only':info})
                if bad or e.functional_success:break
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if process.is_alive():
            try:parent.send(None)
            except (BrokenPipeError,EOFError,OSError):pass
            process.join(2)
        if process.is_alive():process.terminate();process.join(2)
        parent.close()
    result={'seed':seed,'condition':condition,**info,'error':error,'wall_seconds':time.monotonic()-start,
        'command_latency_median_ms':float(np.median(latencies)*1000) if latencies else None,
        'command_latency_p95_ms':float(np.percentile(latencies,95)*1000) if latencies else None,
        'controller_observation_keys':sorted(OBSERVATION_KEYS),'privileged_object_state_to_controller':False,
        'controller_sha256':hashlib.sha256(Path(__file__).with_name('controller.py').read_bytes()).hexdigest()}
    if error:result['success']=False
    dump(folder/'result.json',result);dump(folder/'trace.json',logs)
    np.savez_compressed(folder/'actions_proprio.npz',actions=np.asarray(action_rows),proprio=np.asarray(proprio_rows),image_times=np.asarray(image_times))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--out',type=Path,default=OUT/'development_v1')
    p.add_argument('--condition',choices=['rgb','black','frozen'],default='rgb');p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args();require_cloud();rows=[]
    for seed in a.seeds:
        row=episode(seed,a.out,a.condition,a.steps);rows.append(row);print(json.dumps(row),flush=True)
        dump(a.out/'summary.json',{'episodes':rows,'successes':sum(r['success'] for r in rows),'count':len(rows)})
        if row['error']:raise RuntimeError(row['error'])


if __name__=='__main__':main()
