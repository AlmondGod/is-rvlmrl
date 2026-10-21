"""Local guided-joint diagnostic video, explicitly not an autonomous assembly.

Actual MuJoCo tool contact and single-axis panel motion. Front fixed in a test jig;
left on a linear guide. Uses the same JointProcess as the box workcell.
"""
import argparse
from dataclasses import replace,asdict
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from .design import Config,parts
from .fastening import Frame,JointProcess,Parameters,box_joint
from .environment import fmt,norm_clip


def model_xml(dt):
    c=Config();spec=parts(c);joint=box_joint(c)
    root=ET.Element('mujoco',model='guided_box_joint_diagnostic')
    ET.SubElement(root,'option',timestep=str(dt),integrator='implicitfast',gravity='0 0 -9.81',iterations='100',tolerance='1e-10',cone='elliptic')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.65 .65 .65',specular='.1 .1 .1')
    world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
    ET.SubElement(world,'light',pos='-.3 -.5 1',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='bench',type='box',pos='0 0 -.025',size='.45 .35 .025',rgba='.12 .16 .20 1')
    # Hardware-like supports make the otherwise ideal guide constraints visible.
    ET.SubElement(world,'geom',name='fixed_jig',type='box',pos='0 -.12 .02',size='.16 .016 .02',rgba='.28 .35 .42 1')
    ET.SubElement(world,'geom',name='rail',type='box',pos='-.145 .04 .004',size='.018 .14 .004',rgba='.32 .38 .44 1')
    for name in ('front','left'):
        item=spec[name];body=ET.SubElement(world,'body',name=name,pos=fmt(item['target']))
        if name=='left':
            ET.SubElement(body,'joint',name='guide',type='slide',axis='0 1 0',range='-.002 .25',damping='4')
            ET.SubElement(act,'position',name='clamp',joint='guide',kp='50000',kv='90',forcelimited='true',forcerange='-200 200',ctrllimited='true',ctrlrange='-.002 .25')
        ET.SubElement(body,'geom',name=name+'_panel',type='box',size=fmt(np.array(item['size'])/2),density='600',rgba='.69 .46 .25 1',
            friction='.45 .005 .0001',solref='.004 1',solimp='.99 .999 .0001',margin='0')
    for name,pos,quat,color in [('nailer',[-.1455,-.27,.084],[.70710678,-.70710678,0,0],'.23 .40 .58 1'),
        ('glue_tool',[-.1455,.05,.22],[.70710678,.70710678,0,0],'.86 .84 .70 1')]:
        body=ET.SubElement(world,'body',name=name,pos=fmt(pos),quat=fmt(quat))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'geom',name=name+'_housing',type='cylinder',size='.018 .025',mass='.6',rgba=color)
        ET.SubElement(body,'geom',name=name+'_nose',type='capsule',fromto='0 0 .025 0 0 .10',size='.002',mass='.04',rgba='.5 .53 .56 1')
        ET.SubElement(body,'site',name=name+'_tcp',pos='0 0 .102',size='.001',rgba='0 0 0 0')
    ET.SubElement(root,'contact')
    ET.indent(root);return ET.tostring(root,encoding='unicode')


class Rig:
    def __init__(self,dt=.0005):
        self.model=mujoco.MjModel.from_xml_string(model_xml(dt));self.data=mujoco.MjData(self.model)
        self.p=Parameters();self.process=JointProcess(box_joint(Config()),self.p)
        self.dt=dt;self.frame_dt=.02;self.spec=parts(Config());self.commands={};self.phase='Setup';self.chemical_scale=1.
        self.clamp_enabled=True;self.clamp_target=.18;self.clamp_setpoint=.18;self.feedforward=True;self.logs=[];self.peak_penetration=0.
        self.data.qpos[self.model.joint('guide').qposadr[0]]=.18
        self.data.ctrl[0]=self.clamp_target
        mujoco.mj_forward(self.model,self.data)
        self.targets={n:self.data.xpos[self.model.body(n).id].copy() for n in ('nailer','glue_tool')}
        self.quats={n:self.data.xquat[self.model.body(n).id].copy() for n in self.targets}
        self.last_nose=0.;self.last_pair=0.;self.nose_peak=0.;self.warning_peak=0

    def frame(self,name):
        m,d=self.model,self.data;b=m.body(name).id;v=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0)
        return Frame(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy(),v[3:]+np.cross(v[:3],d.xpos[b]-d.xipos[b]),v[:3].copy())

    def tcp_target(self,name,point):
        # Initial tool orientations are retained; target only translates the carrier.
        return np.asarray(point)-self.frame(name).rotation@np.array([0,0,.102])

    def advance(self,seconds,*,nailer=None,glue=None,trigger=False,dispense=False,clamp=None,chemistry=None):
        if clamp is not None:self.clamp_target=float(clamp)
        if chemistry is not None:self.chemical_scale=float(chemistry)
        frames=round(seconds/self.frame_dt)
        for _ in range(frames):
            self.clamp_setpoint+=float(np.clip(self.clamp_target-self.clamp_setpoint,-.04*self.frame_dt,.04*self.frame_dt))
            for name,target in [('nailer',nailer),('glue_tool',glue)]:
                if target is not None:self.targets[name]+=norm_clip(np.asarray(target)-self.targets[name],.10*self.frame_dt)
            for _ in range(round(self.frame_dt/self.dt)):
                self.substep(trigger,dispense)
            yield self

    def substep(self,trigger,dispense):
        m,d=self.model,self.data
        mujoco.mj_forward(m,d);d.xfrc_applied[:]=0;d.qfrc_applied[:]=0;d.ctrl[0]=self.clamp_setpoint
        for name in self.targets:
            b=m.body(name).id;f=self.frame(name)
            force=3000*(self.targets[name]-f.position)-60*f.linear-m.body_mass[b]*m.opt.gravity
            q=d.xquat[b].copy();q[1:]*=-1;error=np.empty(4);mujoco.mju_mulQuat(error,self.quats[name],q)
            if error[0]<0:error=-error
            angle=2*np.arctan2(np.linalg.norm(error[1:]),max(0.,error[0]))
            torque=24*error[1:]*angle/max(np.linalg.norm(error[1:]),1e-12)-.10*f.angular
            d.xfrc_applied[b]=np.r_[norm_clip(force,180),norm_clip(torque,5)]
        a,b=self.frame('front'),self.frame('left');n=a.rotation@self.process.s.normal_a
        nose=0.;pair=0.;force=np.zeros(6)
        for i,c in enumerate(d.contact):
            if c.dist>0:continue
            names={m.geom(c.geom1).name,m.geom(c.geom2).name}
            if names & {'front_panel','left_panel'}:self.peak_penetration=max(self.peak_penetration,-float(c.dist))
            mujoco.mj_contactForce(m,d,i,force)
            normal=max(0.,float(force[0]))*abs(float(c.frame[:3]@n))
            if names=={'front_panel','nailer_nose'}:nose+=normal
            if names=={'front_panel','left_panel'}:pair+=normal
        self.last_nose=nose;self.last_pair=pair;self.nose_peak=max(self.nose_peak,nose)
        # Explicit, logged acceleration of chemistry ONLY, after the bonds have formed.
        self.process.p=replace(self.p,chemistry_time_scale=self.chemical_scale)
        area=self.spec['left']['size'][0]*self.spec['left']['size'][2]
        loads=self.process.step(self.dt,a,b,nailer_position=d.site_xpos[m.site('nailer_tcp').id].copy(),
            nailer_axis=self.frame('nailer').rotation[:,2],nose_force_N=nose,trigger=trigger,
            glue_position=d.site_xpos[m.site('glue_tool_tcp').id].copy(),glue_axis=self.frame('glue_tool').rotation[:,2],
            dispense=dispense,patch_clamp_N=np.full(4,pair*self.process.s.patch_area_m2/area))
        for name,force,point in loads:mujoco.mj_applyFT(m,d,force,np.zeros(3),point,m.body(name).id,d.qfrc_applied)
        if self.feedforward:
            tool=m.body('nailer').id
            d.xfrc_applied[tool,:3]=norm_clip(d.xfrc_applied[tool,:3]-self.process.last_reaction,180.)
        mujoco.mj_step(m,d)
        self.warning_peak=max(self.warning_peak,sum(int(x.number) for x in d.warning))
        if not np.isfinite(d.qpos).all() or self.warning_peak:raise RuntimeError('Unstable physics')

    def visual(self,scene):
        a=self.frame('front');axis=a.rotation@self.process.s.normal_a
        def shape(kind,size,pos,color):
            g=scene.geoms[scene.ngeom];scene.ngeom+=1
            mujoco.mjv_initGeom(g,kind,np.array(size),np.asarray(pos),np.eye(3).ravel(),np.array(color,np.float32))
            return g
        if self.process.depth>0:
            entry=a.world(self.process.s.entry_a);head=entry-axis*(self.p.nail_length_m-self.process.depth)
            g=shape(mujoco.mjtGeom.mjGEOM_CAPSULE,[.0006]*3,head,[.67,.69,.72,1])
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0006,head,entry+axis*self.process.depth)
            shape(mujoco.mjtGeom.mjGEOM_SPHERE,[.0015]*3,head,[.7,.72,.75,1])
        for local,patch in zip(self.process.s.patches_a,self.process.glue):
            if patch.volume_m3>0:
                radius=.0015*min(1,patch.volume_m3/(self.process.s.patch_area_m2*self.p.glue_film_m))**.5
                shape(mujoco.mjtGeom.mjGEOM_SPHERE,[radius]*3,a.world(local)+axis*.0001,[.92,.87,.67,1])


def sequence(r):
    a=r.frame('front');s=r.process.s;entry=a.world(s.entry_a)
    r.phase='1  Apply glue to exposed mating face'
    # No panel or tool teleportation after initialization: translate targets through finite stages.
    for pa in s.patches_a:
        target=r.tcp_target('glue_tool',a.world(pa)+np.array([0,.0006,0]))
        yield from r.advance(2.,glue=target)
        yield from r.advance(.6,dispense=True)
    r.phase='2  Retract nozzle, close guided clamp'
    yield from r.advance(2.,glue=np.array([-.1455,.04,.25]))
    yield from r.advance(6.,clamp=-.0006)
    r.phase='3  Seat nailer against front panel'
    target=r.tcp_target('nailer',entry+np.array([0,.0012,0]))
    yield from r.advance(1.8,nailer=target)
    r.phase='4  Drive nail; finite force and reaction'
    yield from r.advance(.3,trigger=True)
    yield from r.advance(.2,trigger=False)
    if not r.process.evaluator_metrics()['nail_seated']:
        r.phase='4b  Joint shifted: increase clamp force, retry'
        yield from r.advance(.8,clamp=-.0018)
        for _ in range(5):
            yield from r.advance(.3,trigger=True)
            yield from r.advance(.1,trigger=False)
            if r.process.evaluator_metrics()['nail_seated']:break
    r.phase='5  Retract driver; hold clamp'
    yield from r.advance(1.5,nailer=np.array([-.1455,-.27,.084]))
    # Genuine 60 s continuous clamping in physics; video encodes a stated time-lapse.
    r.phase='6  Clamping - physics time-lapse'
    yield from r.advance(62.)
    r.phase='7  Glue curing - chemical clock accelerated'
    yield from r.advance(8.,chemistry=10800.)
    r.phase='8  Final joint inspection (still guided)'
    yield from r.advance(2.,chemistry=1.)


def render(r,renderer,camera):
    renderer.update_scene(r.data,camera=camera);r.visual(renderer.scene)
    return renderer.render().copy()


def cutaway(r,renderer,camera):
    saved=r.model.geom_rgba.copy()
    try:
        for name in ('nailer_housing','nailer_nose'):r.model.geom_rgba[r.model.geom(name).id,3]=0.
        for name in ('front_panel','left_panel'):r.model.geom_rgba[r.model.geom(name).id,3]=.25
        return render(r,renderer,camera)
    finally:r.model.geom_rgba[:]=saved


def caption(image,text,xy,scale=.62,color=(225,235,242)):
    cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--no-video',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);r=Rig();started=time.time()
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[-.125,-.065,.075];cam.distance=.62;cam.azimuth=225;cam.elevation=-28
    side=mujoco.MjvCamera();side.type=mujoco.mjtCamera.mjCAMERA_FREE;side.lookat[:]=[-.1455,-.095,.084];side.distance=.19;side.azimuth=70;side.elevation=-12
    writer=None;renderer=None;frame_count=0
    if not a.no_video:
        renderer=mujoco.Renderer(r.model,height=540,width=960)
        writer=imageio.get_writer(a.out/'nail_and_glue.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
    last_phase=None
    try:
        for i,_ in enumerate(sequence(r)):
            m=r.process.evaluator_metrics()
            if r.phase!=last_phase:
                print(json.dumps({'phase':r.phase,'time_s':r.data.time,'depth_mm':1000*r.process.depth,'coverage':[p.volume_m3 for p in r.process.glue],
                    'nose_N':r.last_nose,'clamp_contact_N':r.last_pair,'gate':r.process.last_gate,'glue':m['glue']}),flush=True)
                last_phase=r.phase
            if i%5==0:r.logs.append({'time':float(r.data.time),'phase':r.phase,'nose_N':r.last_nose,'pair_N':r.last_pair,'joint':m})
            # Normal motion at 1x (50 physics updates/s -> 25 video fps). Long physical hold at 20x.
            stride=40 if r.phase.startswith('6 ') else 1 if r.phase.startswith(('4 ','4b ')) else 2
            if writer and i%stride==0:
                cam.azimuth=225 if r.phase.startswith(('1 ','2 ')) else 45
                side.azimuth=225 if r.phase.startswith(('1 ','2 ')) else 70
                im=render(r,renderer,cam)
                inset=cutaway(r,renderer,side) if r.phase.startswith(('4 ','4b ')) else render(r,renderer,side)
                inset=cv2.resize(inset,(320,180))
                im[96:276,620:940]=inset
                cv2.rectangle(im,(0,0),(959,84),(15,23,30),-1)
                cv2.rectangle(im,(0,400),(959,539),(15,23,30),-1)
                caption(im,'WOODEN BOX JOINT | LOCAL MUJOCO SIMULATION',(20,30),.75)
                caption(im,r.phase,(20,62),.65)
                caption(im,'Fixed front + guided side panel. Scripted test rig; not autonomous assembly.',(20,424),.55)
                caption(im,f'Nail: {r.process.depth*1000:5.1f} / 25 mm   Nose load: {r.last_nose:5.1f} N   Clamp contact: {r.last_pair:5.1f} N',(20,450),.6)
                cured=np.mean([x['cure_fraction'] for x in m['glue']]);bonds=sum(x['bonded'] for x in m['glue'])
                caption(im,'Diagnostic cutaway (not sensor view)' if r.phase.startswith(('4 ','4b ')) else 'Joint close-up',(620,91),.43)
                caption(im,'MODEL CURE - evaluator telemetry',(620,299),.45)
                cv2.rectangle(im,(620,310),(940,324),(65,77,88),-1)
                cv2.rectangle(im,(620,310),(620+int(320*cured),324),(88,185,150),-1)
                caption(im,f'Chemical age: {r.process.chemical_clock_s/3600:.2f} hours',(620,345),.48)
                caption(im,f'Glue patches bonded: {bonds}/4   Model cure: {100*cured:5.1f}%   Physical time: {r.data.time:5.1f}s',(20,476),.6)
                rate='20x playback of actual clamp hold' if stride==40 else 'Nail driving shown at half speed' if stride==1 else f'Chemical aging: {r.chemical_scale:g}x; robot dynamics: 1x'
                caption(im,rate+' | Material parameters UNCALIBRATED',(20,503),.55,(240,194,115))
                caption(im,'Cure percentage is evaluator telemetry, not a camera-observable measurement.',(20,526),.48)
                writer.append_data(im);frame_count+=1
                if frame_count in (1,200,400):imageio.imwrite(a.out/f'preview_{frame_count}.png',im)
        final=r.process.evaluator_metrics()
        if not final['nail_seated'] or not all(x['bonded'] and not x['failed'] for x in final['glue']):
            raise RuntimeError('Diagnostic did not complete fastening; inspect trace before presenting as successful')
        report={'status':'completed','wall_seconds':time.time()-started,'physical_seconds':float(r.data.time),'video_frames':frame_count,
            'diagnostic_rig':'fixed front, side on a one-axis clamp guide; no autonomous manipulation',
            'parameters':asdict(r.p),'chemistry_schedule':'1x until bond formation; 10800x for 8 seconds; 1x thereafter',
            'peak_contact_penetration_m':r.peak_penetration,'warnings':r.warning_peak,'final':final,
            'events':r.process.events,'physical_validation':False,'runpod_used':False}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'trace.json').write_text(json.dumps(r.logs)+'\n')
        (a.out/'scene.xml').write_text(model_xml(r.dt))
        print(json.dumps(report),flush=True)
        if writer:imageio.imwrite(a.out/'final.png',im)
    finally:
        if writer:writer.close()
        if renderer:renderer.close()


if __name__=='__main__':main()
