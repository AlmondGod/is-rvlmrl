"""MuJoCo adapter for the single-joint fastening experiment.

Local execution requires an explicit allow_local=True constructor argument.

Two finite-wrench virtual tool stages; neither is an industrial arm or a verified
nailer. Tool collision is physical. Nail penetration/glue are constitutive loads.
"""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .design import Config
from .environment import BoxAssembly, build_xml, fmt, norm_clip
from .fastening import Parameters, JointProcess, Frame, box_joint


def tool_xml(config):
    root=ET.fromstring(build_xml(config));root.set('model','box_single_joint_fastening')
    world=root.find('worldbody');contact=root.find('contact')
    for name,x,color in [('nailer',.52,'.24 .32 .40 1'),('glue_tool',-.52,'.75 .75 .65 1')]:
        body=ET.SubElement(world,'body',name=name,pos=fmt([x,.35,.45]))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'geom',name=name+'_housing',type='cylinder',size='.025 .04',mass='1.2',rgba=color)
        ET.SubElement(body,'geom',name=name+'_nose',type='capsule',fromto='0 0 .04 0 0 .10',size='.003',mass='.08',rgba='.4 .4 .4 1')
        ET.SubElement(body,'site',name=name+'_tcp',pos='0 0 .103',size='.001',rgba='0 0 0 0')
        for part in ('bottom','front','back','left','right'):
            for geom in (name+'_housing',name+'_nose'):
                ET.SubElement(contact,'pair',geom1=part+'_panel',geom2=geom,condim='3',
                    friction='.3 .3 .001 .0001 .0001',solref=fmt([config.contact_time_constant,config.contact_damping_ratio]),
                    solimp=fmt(config.contact_impedance),margin='0',gap='0')
    ET.indent(root);return ET.tostring(root,encoding='unicode')


class FasteningBoxAssembly(BoxAssembly):
    def __init__(self,config=Config(),parameters=None,condition='combined',*,allow_local=False,xml_builder=tool_xml):
        self.fastening_parameters=parameters or Parameters();self.fastening_condition=condition
        super().__init__(config,xml_builder=xml_builder,allow_local=allow_local)
        self.tools={name:self.model.body(name).id for name in ('nailer','glue_tool')}
        self.reset()

    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        self.process=JointProcess(box_joint(self.config),self.fastening_parameters,self.fastening_condition)
        self.tool_commands={};self.tool_targets={};self.tool_quaternions={};self.tool_wrenches={}
        if hasattr(self,'tools'):
            for name,b in self.tools.items():
                self.tool_targets[name]=self.data.xpos[b].copy();self.tool_quaternions[name]=self.data.xquat[b].copy()

    def step(self,position,quaternion,fingers,*,tools=None):
        # Tool controls contain motor targets and triggers, never object IDs/poses.
        tools={} if tools is None else tools
        if set(tools)-set(self.tools):raise ValueError('Unknown tool')
        validated={}
        for name,command in tools.items():
            if set(command)-{'position','quaternion','trigger','dispense'}:raise ValueError('Unknown tool control')
            pos=np.asarray(command.get('position',self.tool_targets[name]),float)
            quat=np.asarray(command.get('quaternion',self.tool_quaternions[name]),float)
            if pos.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(np.r_[pos,quat]).all() or np.linalg.norm(quat)<1e-8:
                raise ValueError('Invalid tool target')
            validated[name]=(pos,quat/np.linalg.norm(quat))
        for name,(pos,quat) in validated.items():
            self.tool_targets[name]+=norm_clip(np.clip(pos,[-.65,-.5,.02],[.65,.5,.7])-self.tool_targets[name],.10*self.config.control_period)
            previous=self.tool_quaternions[name]
            if previous@quat<0:quat=-quat
            angle=2*np.arccos(np.clip(previous@quat,-1,1));fraction=min(1.,self.config.control_period/max(angle,1e-12))
            q=(1-fraction)*previous+fraction*quat;self.tool_quaternions[name]=q/np.linalg.norm(q)
        self.tool_commands=tools
        return super().step(position,quaternion,fingers)

    def _frame(self,name):
        m,d=self.model,self.data;b=m.body(name).id;v=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0)
        # MuJoCo body spatial velocity is centered on the body's inertial COM.
        origin=d.xpos[b];linear=v[3:]+np.cross(v[:3],origin-d.xipos[b])
        return Frame(origin.copy(),d.xmat[b].reshape(3,3).copy(),linear,v[:3].copy())

    def _additional_forces(self):
        m,d=self.model,self.data;d.qfrc_applied[:]=0
        for name,b in self.tools.items():
            frame=self._frame(name)
            force=1200*(self.tool_targets[name]-frame.position)-70*frame.linear-m.body_mass[b]*m.opt.gravity
            q=d.xquat[b].copy();q[1:]*=-1;error=np.empty(4)
            mujoco.mju_mulQuat(error,self.tool_quaternions[name],q)
            if error[0]<0:error=-error
            angle=2*np.arctan2(np.linalg.norm(error[1:]),max(0.,error[0]))
            angular_damping=2*np.sqrt(18*np.min(m.body_inertia[b]))
            torque=18*error[1:]*angle/max(np.linalg.norm(error[1:]),1e-12)-angular_damping*frame.angular
            wrench=np.r_[norm_clip(force,150.),norm_clip(torque,5.)]
            d.xfrc_applied[b]=wrench;self.tool_wrenches[name]=wrench.copy()
        a,b=self._frame(self.process.s.a),self._frame(self.process.s.b)
        n=a.rotation@self.process.s.normal_a;nose_force=0.;pair_force=0.;f=np.zeros(6)
        ga=m.geom('front_panel').id;gb=m.geom('left_panel').id;gn=m.geom('nailer_nose').id
        for i,c in enumerate(d.contact):
            if c.dist>0:continue
            ids={c.geom1,c.geom2}
            if ids not in ({ga,gb},{ga,gn}):continue
            mujoco.mj_contactForce(m,d,i,f)
            normal_component=max(0.,float(f[0]))*abs(float(np.asarray(c.frame[:3])@n))
            if ids=={ga,gn}:nose_force+=normal_component
            else:pair_force+=normal_component
        # Uniform face-pressure approximation, not local pressure sensing.
        face_area=self.spec['left']['size'][0]*self.spec['left']['size'][2]
        clamp=np.full(4,pair_force*self.process.s.patch_area_m2/face_area)
        nailer=self._frame('nailer');glue=self._frame('glue_tool')
        loads=self.process.step(self.config.timestep,a,b,
            nailer_position=d.site_xpos[m.site('nailer_tcp').id].copy(),nailer_axis=nailer.rotation[:,2],nose_force_N=nose_force,
            trigger=bool(self.tool_commands.get('nailer',{}).get('trigger',False)),
            glue_position=d.site_xpos[m.site('glue_tool_tcp').id].copy(),glue_axis=glue.rotation[:,2],
            dispense=bool(self.tool_commands.get('glue_tool',{}).get('dispense',False)),patch_clamp_N=clamp)
        # Same-step bounded compensation for the tool's own driving reaction.
        nailer_id=self.tools['nailer']
        d.xfrc_applied[nailer_id,:3]=norm_clip(d.xfrc_applied[nailer_id,:3]-self.process.last_reaction,150.)
        for name,force,point in loads:
            mujoco.mj_applyFT(m,d,force,np.zeros(3),point,m.body(name).id,d.qfrc_applied)
        # Base stage wrench still never directly carries panels. Joint loads above are
        # explicit finite interactions, separately applied and logged by the process.

    def evaluator_metrics(self):
        result=super().evaluator_metrics()
        if hasattr(self,'process'):
            result['fastening_status']='single_joint_reduced_order_uncalibrated'
            result['single_joint']=self.process.evaluator_metrics()
            # The old unfastened-release label must not describe a bonded episode.
            if self.process.nail is not None or any(p.connector is not None for p in self.process.glue):
                result['released_unfastened_stability']=False
        result['full_assembly_success']=False
        return result

    def tool_observation(self):
        """Separate versioned sensor packet; existing camera allowlist stays intact."""
        return {'schema':'box_tools_v1','time':float(self.data.time),
            'tools':{name:{'position':self.data.xpos[b].copy(),'quaternion':self.data.xquat[b].copy()} for name,b in self.tools.items()},
            'sensors':self.process.sensor_packet(),'sensor_model':'ideal_uncalibrated'}

    def _render_extra(self,scene):
        a=self._frame(self.process.s.a);axis=a.rotation@self.process.s.normal_a
        def sphere(pos,radius,color):
            if scene.ngeom>=scene.maxgeom:raise RuntimeError('Insufficient rendering geometry capacity')
            geom=scene.geoms[scene.ngeom];scene.ngeom+=1
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),pos,np.eye(3).ravel(),np.array(color,dtype=np.float32))
        if self.process.depth>0:
            head=a.world(self.process.s.entry_a)-axis*(self.process.p.nail_length_m-self.process.depth)
            sphere(head,.0015,[.55,.56,.58,1.])
        for local,patch in zip(self.process.s.patches_a,self.process.glue):
            if patch.volume_m3>0:
                # Fixed appearance: color never reveals latent cure or damage.
                sphere(a.world(local)+axis*.0001,.0015,[.82,.78,.61,1.])
