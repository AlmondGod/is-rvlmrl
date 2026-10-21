"""Uncalibrated wooden box workcell. No object assistance/welds.

Cloud execution is the default; allow_local=True explicitly opts into local runs.

Finite-wrench wrist is a test positioning stage, not a real robot-arm model.
Object state is exposed only through explicitly named evaluator methods.
"""
from collections import deque
import itertools
import os
import platform
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .design import Config, ROOT, NAMES, parts, beveled_vertices

HAND=ROOT/'third_party/wuji-description/hand2/hand2_2/body/mjcf/right.xml'


def fmt(x):return ' '.join(f'{float(v):.10g}' for v in x)


def require_cloud():
    if platform.system()!='Linux' or os.environ.get('PHYLA_CLOUD_RUN')!='1':
        raise RuntimeError('Physics execution requires the authorized Linux cloud host and PHYLA_CLOUD_RUN=1')


def build_xml(c):
    root=ET.parse(HAND).getroot();root.set('model','wuji_box_dry_fit')
    root.find('compiler').set('meshdir',str(HAND.parent.parent/'meshes/right'))
    option=root.find('option');option.set('timestep',str(c.timestep));option.set('integrator','implicitfast')
    option.set('gravity','0 0 -9.81');option.set('cone','elliptic');option.set('iterations','100');option.set('tolerance','1e-9')
    # Flat beveled panel meshes need a contact manifold, not a single unstable corner.
    flags=option.find('flag')
    if flags is None:flags=ET.SubElement(option,'flag')
    flags.set('multiccd','enable')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6',specular='.1 .1 .1')
    assets=root.find('asset');world=root.find('worldbody');actuator=root.find('actuator')
    mount=world.find("body[@name='r_mount']");mount.set('pos','.42 0 .40')
    for i,geom in enumerate(mount.iter('geom')):
        if not geom.get('name'):geom.set('name',f'robot_geom_{i}')
    ET.SubElement(mount,'freejoint',name='wrist_free')
    ET.SubElement(world,'light',pos='0 0 1.5',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='bench',type='box',pos='0 0 -.025',size='.68 .55 .025',rgba='.20 .23 .25 1')
    for i,pos in enumerate(((.65,-.65,.55),(.65,.65,.55))):
        back=np.array(pos)-[0,0,.08];back/=np.linalg.norm(back)
        right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
        ET.SubElement(world,'camera',name=f'vision_{i}',pos=fmt(pos),xyaxes=fmt(np.r_[right,up]),fovy='65')
    colors=['.68 .46 .27 1','.72 .49 .28 1','.69 .47 .27 1','.71 .48 .28 1','.70 .46 .26 1']
    for (name,item),color in zip(parts(c).items(),colors):
        vertices=beveled_vertices(item['size'],c.edge_bevel)
        ET.SubElement(assets,'mesh',name=f'{name}_mesh',vertex=fmt(np.asarray(vertices).ravel()))
        body=ET.SubElement(world,'body',name=name,pos=fmt(item['start']),quat=fmt(item['start_quat']))
        ET.SubElement(body,'freejoint',name=f'{name}_free')
        mass_property={'mass':str(c.panel_masses[name])} if name in c.panel_masses else {'density':str(c.density)}
        ET.SubElement(body,'geom',name=f'{name}_panel',type='mesh',mesh=f'{name}_mesh',**mass_property,rgba=color,
            margin='0',gap='0',condim='3',solref=fmt([c.contact_time_constant,c.contact_damping_ratio]),solimp=fmt(c.contact_impedance))
    # Real collision stops and guided, retractable inner jaws; no panel constraints.
    # Guides are a rigid workholding approximation; their flexibility is uncalibrated.
    L,W,t=c.length,c.width,c.thickness;g=c.fixture_clearance
    for ix,iy in itertools.product((-1,1),repeat=2):
        tag=f'{ix}_{iy}'
        jaw=ET.SubElement(world,'body',name=f'fixture_{tag}',pos='0 0 0')
        ET.SubElement(jaw,'joint',name=f'jaw_{tag}',type='slide',axis='0 0 1',range=fmt([0,c.height+.06]),damping='10')
        ET.SubElement(world,'geom',name=f'fixture_guide_{tag}',type='box',
            pos=fmt([ix*(L/2+.04),iy*(W/2+.04),.25]),size='.008 .008 .25',rgba='.25 .30 .34 1')
        for axis in (0,1):
            p=[ix*(L/2-.025),iy*(W/2+.006+g),.045]
            size=[.025,.006,.045]
            if axis==0:p=[ix*(L/2+.006+g),iy*(W/2-.025),.045];size=[.006,.025,.045]
            ET.SubElement(jaw,'geom',name=f'stop_{tag}_{axis}',type='box',pos=fmt(p),size=fmt(size),density='800',rgba='.30 .35 .40 1')
        # One vertical guide per corner carries two inside jaw pads.
        z=t+.045
        for axis in (0,1):
            if axis==0:p=[ix*(L/2-t-.006-g),iy*(W/2-t-.025),z];size=[.006,.020,.025]
            else:p=[ix*(L/2-t-.025),iy*(W/2-t-.006-g),z];size=[.020,.006,.025]
            ET.SubElement(jaw,'geom',name=f'jaw_pad_{tag}_{axis}',type='box',pos=fmt(p),size=fmt(size),density='800',rgba='.40 .48 .53 1')
        # Visible rigid riser above the pads. It is part of the prismatic fixture.
        ET.SubElement(jaw,'geom',name=f'jaw_riser_{tag}',type='box',
            pos=fmt([ix*(L/2-t-.012),iy*(W/2-t-.012),c.height+.04]),size=fmt([.004,.004,c.height*.6]),
            density='800',rgba='.35 .40 .44 1')
        ET.SubElement(actuator,'position',name=f'fixture_motor_{tag}',joint=f'jaw_{tag}',kp='2000',kv='30',
            ctrllimited='true',ctrlrange=fmt([0,c.height+.06]),forcelimited='true',forcerange='-30 30')
    # Explicit pair coefficients avoid accidental max-friction mixing from hand defaults.
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    hand_geoms=[x.get('name') for x in mount.iter('geom') if x.get('name') and
        not (x.get('contype')=='0' and x.get('conaffinity')=='0')]
    fixtures=[x.get('name') for x in world.iter('geom') if (x.get('name') or '').startswith(('stop_','jaw_','fixture_guide_'))]
    def pair(a,b,mu):
        ET.SubElement(contact,'pair',geom1=a,geom2=b,condim='3',friction=fmt([mu,mu,.001,.0001,.0001]),
            solref=fmt([c.contact_time_constant,c.contact_damping_ratio]),solimp=fmt(c.contact_impedance),margin='0',gap='0')
    for a,b in itertools.combinations(NAMES,2):pair(a+'_panel',b+'_panel',c.wood_wood_friction)
    for name in NAMES:
        panel=name+'_panel';pair(panel,'bench',c.wood_bench_friction)
        for geom in hand_geoms:pair(panel,geom,c.wood_hand_friction)
        for geom in fixtures:pair(panel,geom,c.wood_fixture_friction)
    ET.indent(root);return ET.tostring(root,encoding='unicode')


def norm_clip(x,limit):
    x=np.asarray(x,dtype=float);return x*min(1.,limit/max(np.linalg.norm(x),1e-12))


class BoxAssembly:
    def __init__(self,config=Config(),*,xml_builder=build_xml,allow_local=False):
        if not allow_local:require_cloud()
        self.config=config;self.xml=xml_builder(config)
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        m=self.model
        self.wid=m.body('r_mount').id;self.wrist_joint=m.joint('wrist_free').id
        self.finger_actuators=np.array([i for i in range(m.nu) if not m.actuator(i).name.startswith('fixture_motor_')])
        self.finger_joint_ids=m.actuator_trnid[self.finger_actuators,0]
        self.finger_qadr=m.jnt_qposadr[self.finger_joint_ids];self.finger_vadr=m.jnt_dofadr[self.finger_joint_ids]
        self.fixture_actuators=np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('fixture_motor_')])
        self.fixture_joints=m.actuator_trnid[self.fixture_actuators,0]
        self.hand_bodies={i for i in range(m.nbody) if m.body(i).name.startswith('r_')}
        self.panel_ids={n:m.body(n).id for n in NAMES};self.spec=parts(config)
        self.hand_mass=sum(m.body_mass[i] for i in self.hand_bodies)
        self.reset()

    def reset(self,seed=0,*,diagnostic_assembled=False):
        m,d=self.model,self.data;c=self.config;mujoco.mj_resetData(m,d);rng=np.random.default_rng(seed)
        for n,item in self.spec.items():
            qa=int(m.joint(n+'_free').qposadr[0]);pos=np.array(item['target'] if diagnostic_assembled else item['start'],float)
            q=np.array([1.,0,0,0] if diagnostic_assembled else item['start_quat'])
            if not diagnostic_assembled:
                pos[:2]+=rng.uniform(-c.start_xy_jitter,c.start_xy_jitter,2)
                yaw=rng.uniform(-c.start_yaw_jitter,c.start_yaw_jitter);rotation=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
                updated=np.empty(4);mujoco.mju_mulQuat(updated,rotation,q);q=updated
            d.qpos[qa:qa+3]=pos;d.qpos[qa+3:qa+7]=q
        d.ctrl[self.finger_actuators]=np.clip(0,*m.actuator_ctrlrange[self.finger_actuators].T)
        d.qpos[self.finger_qadr]=d.ctrl[self.finger_actuators]
        d.ctrl[self.fixture_actuators]=0
        mujoco.mj_forward(m,d)
        self.wrist_target=d.xpos[self.wid].copy();self.wrist_quaternion=d.xquat[self.wid].copy()
        self.finger_target=d.ctrl[self.finger_actuators].copy();self.fixture_target=0.
        self.queue=deque();self.dry_fit_hold=0.;self.released_hold=0.;self.peak_penetration=0.;self.peak_panel_force=0.
        self.applied_wrench=np.zeros(6);self.stage_saturation_steps=0;self.steps=0
        self.diagnostic_assembled=diagnostic_assembled

    def fixture_release(self,release=True):
        """Station command: retract real jaw bodies through finite-force actuators."""
        self.fixture_target=self.config.height+.06 if release else 0.

    def _apply_stage_wrench(self):
        m,d=self.model,self.data;c=self.config;v=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.wid,v,0)
        force=c.wrist_position_gain*(self.wrist_target-d.xpos[self.wid])-c.wrist_position_damping*v[3:]
        force-=self.hand_mass*m.opt.gravity
        conjugate=d.xquat[self.wid].copy();conjugate[1:]*=-1
        error=np.empty(4);mujoco.mju_mulQuat(error,self.wrist_quaternion,conjugate)
        if error[0]<0:error=-error
        angle=2*np.arctan2(np.linalg.norm(error[1:]),max(error[0],0.))
        rotation=error[1:]*angle/max(np.linalg.norm(error[1:]),1e-12)
        torque=c.wrist_rotation_gain*rotation-c.wrist_rotation_damping*v[:3]
        self.stage_saturation_steps+=int(np.linalg.norm(force)>c.wrist_force_limit or np.linalg.norm(torque)>c.wrist_torque_limit)
        self.applied_wrench=np.r_[norm_clip(force,c.wrist_force_limit),norm_clip(torque,c.wrist_torque_limit)]
        d.xfrc_applied[:]=0;d.xfrc_applied[self.wid]=self.applied_wrench
        # These wrenches act on the robot mount only, never on an object.
        assert all(not np.any(d.xfrc_applied[b]) for b in self.panel_ids.values())

    def _additional_forces(self):
        """Extension hook for explicit tool/joint loads; no-op in dry-fit workcell."""
        pass

    def _render_extra(self,scene):
        pass

    def step(self,position,quaternion,fingers):
        p,q,f=(np.asarray(x,float).copy() for x in (position,quaternion,fingers));c=self.config;m,d=self.model,self.data
        if p.shape!=(3,) or q.shape!=(4,) or f.shape!=(20,) or not all(np.isfinite(x).all() for x in (p,q,f)) or np.linalg.norm(q)<1e-8:raise ValueError('Invalid motor command')
        q/=np.linalg.norm(q);self.queue.append((p,q,f))
        if len(self.queue)>c.action_delay_steps:
            p,q,f=self.queue.popleft()
            self.wrist_target+=norm_clip(np.clip(p,[-.60,-.48,.025],[.60,.48,.65])-self.wrist_target,c.wrist_speed*c.control_period)
            if np.dot(q,self.wrist_quaternion)<0:q=-q
            angle=2*np.arccos(np.clip(np.dot(q,self.wrist_quaternion),-1,1));a=min(1.,c.wrist_angular_speed*c.control_period/max(angle,1e-9))
            self.wrist_quaternion=(1-a)*self.wrist_quaternion+a*q;self.wrist_quaternion/=np.linalg.norm(self.wrist_quaternion)
            self.finger_target=np.clip(f,self.finger_target-c.finger_target_speed*c.control_period,self.finger_target+c.finger_target_speed*c.control_period)
            self.finger_target=np.clip(self.finger_target,*m.actuator_ctrlrange[self.finger_actuators].T)
        d.ctrl[self.finger_actuators]=self.finger_target
        d.ctrl[self.fixture_actuators]+=np.clip(self.fixture_target-d.ctrl[self.fixture_actuators],-.1*c.control_period,.1*c.control_period)
        for _ in range(round(c.control_period/c.timestep)):
            mujoco.mj_forward(m,d);self._apply_stage_wrench();self._additional_forces();mujoco.mj_step(m,d);self._audit()
        mujoco.mj_forward(m,d);self.steps+=1
        scores=self.evaluator_metrics();good=scores['instantaneous_dry_fit']
        self.dry_fit_hold=self.dry_fit_hold+c.control_period if good else 0.
        released=good and scores['fixture_retracted'] and not scores['fixture_touching_panels']
        self.released_hold=self.released_hold+c.control_period if released else 0.
        return self.evaluator_metrics()

    def _audit(self):
        m,d=self.model,self.data;ids=set(self.panel_ids.values());force=np.zeros(6)
        for i,contact in enumerate(d.contact):
            a,b=(int(m.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
            if a in ids or b in ids:
                self.peak_penetration=max(self.peak_penetration,-float(contact.dist))
                mujoco.mj_contactForce(m,d,i,force);self.peak_panel_force=max(self.peak_panel_force,float(max(0.,force[0])))

    def evaluator_metrics(self):
        m,d=self.model,self.data;c=self.config;errors={};speeds={};loads={};hand_touch=False;fixture_touch=False
        reverse={b:n for n,b in self.panel_ids.items()};f=np.zeros(6)
        for n,b in self.panel_ids.items():
            qa=int(m.joint(n+'_free').qposadr[0]);va=int(m.joint(n+'_free').dofadr[0])
            errors[n]={'position_m':float(np.linalg.norm(d.xpos[b]-self.spec[n]['target'])),
                'angle_rad':float(2*np.arccos(np.clip(abs(d.qpos[qa+3]),0,1)))}
            speeds[n]={'linear_m_s':float(np.linalg.norm(d.qvel[va:va+3])),'angular_rad_s':float(np.linalg.norm(d.qvel[va+3:va+6]))}
        for i,contact in enumerate(d.contact):
            a,b=(int(m.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
            if contact.dist>0 or not (a in reverse or b in reverse):continue
            hand_touch|=(a in self.hand_bodies or b in self.hand_bodies)
            names=[m.geom(g).name for g in (contact.geom1,contact.geom2)]
            fixture_touch|=any(n.startswith(('stop_','jaw_','fixture_guide_')) for n in names)
            if a in reverse and b in reverse:
                key='-'.join(sorted((reverse[a],reverse[b])));mujoco.mj_contactForce(m,d,i,f)
                loads[key]=loads.get(key,0.)+float(max(0.,f[0]))
        seated=all(x['position_m']<=c.success_position_tolerance and x['angle_rad']<=c.success_angle_tolerance for x in errors.values())
        supported=all(loads.get('-'.join(sorted(('bottom',n))),0.)>.01 for n in NAMES[1:])
        still=all(x['linear_m_s']<.005 and x['angular_rad_s']<.05 for x in speeds.values())
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        warnings=[int(w.number) for w in d.warning]
        good=seated and supported and still and not hand_touch and self.peak_penetration<c.success_penetration_tolerance and finite and not any(warnings)
        retracted=bool(np.all(d.qpos[m.jnt_qposadr[self.fixture_joints]]>c.height+.04))
        return {'instantaneous_dry_fit':bool(good),'dry_fit_success':bool(good and self.dry_fit_hold>=c.release_hold_seconds-1e-8),
            'released_unfastened_stability':bool(good and self.released_hold>=c.release_hold_seconds-1e-8),
            'full_assembly_success':False,'fastening_status':'not_modelled','diagnostic_initialization':self.diagnostic_assembled,
            'pose_errors':errors,'speeds':speeds,'panel_contact_loads_N':loads,'all_walls_supported_on_bottom':supported,
            'hand_touching_panels':hand_touch,'fixture_touching_panels':fixture_touch,'fixture_retracted':retracted,
            'dry_fit_hold_seconds':self.dry_fit_hold,'released_hold_seconds':self.released_hold,
            'peak_episode_penetration_m':self.peak_penetration,'peak_panel_contact_force_N':self.peak_panel_force,
            'stage_saturation_substeps':self.stage_saturation_steps,'applied_robot_wrench':self.applied_wrench.tolist(),
            'warnings':warnings,'finite_state':finite,'calibration_status':c.calibration_status}

    def camera_observation(self,renderer,width=640,height=480):
        m,d=self.model,self.data;images=[];cameras=[]
        option=mujoco.MjvOption();option.geomgroup[2]=0;option.sitegroup[:]=0
        for i in range(2):
            name=f'vision_{i}';renderer.update_scene(d,camera=name,scene_option=option);self._render_extra(renderer.scene);images.append(renderer.render().copy());cid=m.camera(name).id
            R=np.diag([1.,-1.,-1.])@d.cam_xmat[cid].reshape(3,3).T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@d.cam_xpos[cid]
            focal=.5*height/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
            cameras.append({'K':np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]]),'world_to_camera':T})
        return {'time':float(d.time),'image_time':float(d.time),'rgb':np.asarray(images),'cameras':cameras,
            'wrist_position':d.xpos[self.wid].copy(),'wrist_quaternion':d.xquat[self.wid].copy(),
            'finger_q':d.qpos[self.finger_qadr].copy(),'finger_qvel':d.qvel[self.finger_vadr].copy(),
            'wrist_target':self.wrist_target.copy(),'wrist_target_quaternion':self.wrist_quaternion.copy(),
            'finger_targets':self.finger_target.copy(),'joint_limits':m.actuator_ctrlrange[self.finger_actuators].copy()}
