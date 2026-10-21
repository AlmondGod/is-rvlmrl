"""Reduced-order constitutive laws for ONE front/left box joint.

No simulator import. SI units throughout. All defaults are engineering placeholders,
not wood/fastener specifications. This module is evaluator-side, never a policy input.
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Parameters:
    nail_length_m: float = .025
    drive_intercept_N: float = 35.
    drive_slope_N_m: float = 1800.
    drive_drag_N_s_m: float = 500.
    drive_force_limit_N: float = 110.
    drive_energy_budget_J: float = 4.
    drive_timeout_s: float = .5
    nose_preload_N: float = 2.
    tool_position_tolerance_m: float = .002
    tool_angle_tolerance_rad: float = .12
    max_mating_gap_m: float = .0005
    mating_slip_tolerance_m: float = .001
    nail_stiffness_N_m: float = 12000.
    nail_damping_N_s_m: float = 4.
    nail_withdrawal_N: float = 45.
    nail_lateral_N: float = 70.
    nail_failure_displacement_m: float = .004
    nail_seating_tolerance_m: float = .0005
    glue_flow_m3_s: float = 1e-7
    glue_film_m: float = .00015
    glue_nozzle_radius_m: float = .003
    glue_open_time_s: float = 300.
    glue_clamp_force_per_patch_N: float = .5
    glue_continuous_clamp_s: float = 60.
    glue_cure_t95_s: float = 86400.
    glue_stiffness_full_N_m: float = 8000.
    glue_damping_N_s_m: float = 2.
    glue_tensile_strength_Pa: float = 4e5
    glue_shear_strength_Pa: float = 6e5
    glue_failure_displacement_m: float = .002
    glue_handling_fraction: float = .25
    chemistry_time_scale: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('All fastening parameters must be finite and positive')
        if not 0 < self.glue_handling_fraction <= 1:
            raise ValueError('Invalid handling fraction')
        if self.tool_angle_tolerance_rad >= math.pi/2:
            raise ValueError('Tool angle tolerance must be below pi/2')
        if self.glue_film_m > self.max_mating_gap_m:
            raise ValueError('Film exceeds supported bondline gap')


@dataclass
class Frame:
    position: np.ndarray
    rotation: np.ndarray
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def world(self, p): return self.position + self.rotation @ np.asarray(p)
    def local(self, p): return self.rotation.T @ (p-self.position)
    def velocity(self, p): return self.linear + np.cross(self.angular, p-self.position)


@dataclass(frozen=True)
class JointSpec:
    a: str
    b: str
    entry_a: tuple
    normal_a: tuple
    normal_b: tuple
    face_a: tuple
    face_b: tuple
    patches_a: tuple
    patches_b: tuple
    patch_area_m2: float
    entry_thickness_m: float


def box_joint(config):
    """Design intent only; runtime gates use moving panel frames, never world targets.

    One nail through the front wall into the left wall, four adhesive quadrature
    patches over a 30 mm tall region. This does not fasten the whole box.
    """
    from .design import parts
    panels=parts(config); a=panels['front']; b=panels['left']
    ta,tb=np.array(a['target']),np.array(b['target'])
    # Actual supplied dimensions change face offsets and consequently the bondline gap.
    x=tb[0]; z=config.thickness+.075
    fa=np.array([x,ta[1]+a['size'][1]/2,z])-ta
    fb=np.array([x,tb[1]-b['size'][1]/2,z])-tb
    width=min(b['size'][0],.009); height=.030
    if z-height/2<max(ta[2]-a['size'][2]/2,tb[2]-b['size'][2]/2) or z+height/2>min(ta[2]+a['size'][2]/2,tb[2]+b['size'][2]/2):
        raise ValueError('Chosen coupon region does not fit these panels')
    pa=tuple(tuple(fa+[sx*width/4,0,sz*height/4]) for sx in (-1,1) for sz in (-1,1))
    pb=tuple(tuple(fb+[sx*width/4,0,sz*height/4]) for sx in (-1,1) for sz in (-1,1))
    return JointSpec('front','left',tuple(fa-[0,a['size'][1],0]),(0,1,0),(0,-1,0),
                     tuple(fa),tuple(fb),pa,pb,width*height/4,a['size'][1])


@dataclass
class Connector:
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    failed: bool = False
    fracture_energy_J: float = 0.
    peak_utilization: float = 0.
    stiffness_work_J: float = 0.
    previous_k: float = 0.

    @classmethod
    def at(cls, a, b, point):
        # Coincident anchors capture current configuration without moving either part.
        return cls(a.local(point), b.local(point))

    def evaluate(self, a, b, axis, k, damping, normal_limit, shear_limit, displacement_limit, dt):
        pa,pb=a.world(self.anchor_a),b.world(self.anchor_b)
        point=(pa+pb)/2; delta=pb-pa
        if self.failed or k <= 0:
            return np.zeros(3),point,0.,0.
        # Common application point conserves total moment, including shear damping.
        # Isotropic elastic force is collinear with anchor separation, so this also
        # preserves its work. Damping uses velocity at that same common point.
        v=b.velocity(point)-a.velocity(point)
        elastic=k*delta; force=elastic+damping*v
        normal=float(force@axis); shear=float(np.linalg.norm(force-normal*axis))
        # Symmetric compression/tension cap is a conservative surrogate; wood contact
        # carries compression separately. It is not a calibrated mixed-mode criterion.
        u=math.sqrt((normal/normal_limit)**2+(shear/shear_limit)**2)
        self.peak_utilization=max(self.peak_utilization,u)
        self.stiffness_work_J+=.5*(k-self.previous_k)*float(delta@delta)
        self.previous_k=k
        energy=.5*k*float(delta@delta)
        if u>=1 or np.linalg.norm(delta)>=displacement_limit:
            self.failed=True; self.fracture_energy_J+=energy
            return np.zeros(3),point,0.,0.
        return force,point,energy,damping*float(v@v)*dt


@dataclass
class GluePatch:
    volume_m3: float = 0.
    deposited_at_s: float | None = None
    expired: bool = False
    clamp_s: float = 0.
    connector: Connector | None = None


def cure_fraction(age_s, t95_s):
    return -math.expm1(-math.log(20)*max(0.,age_s)/t95_s)


def driving_increment(depth, force, dt, p, energy_left):
    """Exact constant-force linear-resistance/viscous feed law, seat/energy capped.

    No inertial hammer model. Input work = penetration work + viscous dissipation.
    Returns depth increment, active time, input work, penetration dissipation.
    """
    resistance=p.drive_intercept_N+p.drive_slope_N_m*depth
    if force<=resistance or energy_left<=0 or depth>=p.nail_length_m:
        return 0.,0.,0.,0.
    rate=p.drive_slope_N_m/p.drive_drag_N_s_m
    equilibrium=(force-p.drive_intercept_N)/p.drive_slope_N_m
    delta=(equilibrium-depth)*(-math.expm1(-rate*dt))
    delta=min(delta,p.nail_length_m-depth,energy_left/force)
    active=-math.log1p(-delta/(equilibrium-depth))/rate
    work=force*delta
    penetration=resistance*delta+.5*p.drive_slope_N_m*delta*delta
    return delta,active,work,penetration


class JointProcess:
    """Evaluator-side operation state; only sensor_packet() may reach a policy."""
    def __init__(self,spec,parameters=None,condition='combined'):
        if condition not in ('unfastened','nails_only','glue_only','combined'):
            raise ValueError('Unknown experimental condition')
        self.s=spec; self.p=parameters or Parameters(); self.condition=condition
        if self.p.nail_length_m<=spec.entry_thickness_m+self.p.max_mating_gap_m:
            raise ValueError('Nail cannot engage receiving member')
        self.depth=0.; self.nail=None; self.nail_failed=False; self.receiver_entry_m=None
        self.drive_elapsed=0.; self.drive_active=False; self.previous_trigger=False
        self.drive_work=0.; self.penetration_work=0.; self.drive_viscous_work=0.
        self.glue=[GluePatch() for _ in spec.patches_a]
        self.clock_s=0.; self.chemical_clock_s=0.; self.events=[]
        self.damping_work=0.; self.stored_energy=0.; self.dispensed_volume=0.; self.wasted_volume=0.
        self.driver_force=0.; self.nose_contact=False; self.peak_connection_force=0.
        self.last_gate='idle'; self.last_reaction=np.zeros(3)

    def _event(self,kind,**kw): self.events.append({'time_s':self.clock_s,'event':kind,**kw})

    def _mating(self,a,b,pa,pb):
        p=self.p; n=a.rotation@self.s.normal_a; other=b.rotation@self.s.normal_b
        delta=b.world(pb)-a.world(pa); gap=float(delta@n)
        valid=(float(n@other)<-math.cos(p.tool_angle_tolerance_rad)
               and -.0001<=gap<=p.max_mating_gap_m
               and np.linalg.norm(delta-gap*n)<=p.mating_slip_tolerance_m)
        return bool(valid),gap

    def step(self,dt,a,b,*,nailer_position,nailer_axis,nose_force_N=0.,trigger=False,
             glue_position=None,glue_axis=None,dispense=False,patch_clamp_N=None):
        """Returns (body-name, world force, common world point) loads.

        Caller supplies actual tool poses/contact, never commanded target poses.
        Active driving transfers equal/opposite force to the tool and wood.
        """
        if not math.isfinite(dt) or dt<=0: raise ValueError('Invalid timestep')
        for f in (a,b):
            if not all(np.isfinite(v).all() for v in (f.position,f.rotation,f.linear,f.angular)):
                raise ValueError('Nonfinite body state')
        p=self.p; s=self.s; loads=[]; n=a.rotation@s.normal_a
        entry=a.world(s.entry_a); tip=np.asarray(nailer_position,float); axis=np.asarray(nailer_axis,float)
        if tip.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[tip,axis,nose_force_N]).all() or abs(np.linalg.norm(axis)-1)>1e-5:
            raise ValueError('Invalid tool state')
        self.clock_s+=dt; self.chemical_clock_s+=dt*p.chemistry_time_scale
        valid,gap=self._mating(a,b,s.face_a,s.face_b)
        self.nose_contact=nose_force_N>=p.nose_preload_N
        aligned=np.linalg.norm(tip-entry)<=p.tool_position_tolerance_m and float(axis@n)>=math.cos(p.tool_angle_tolerance_rad)
        gate=self.nose_contact and aligned and valid
        self.driver_force=0.; self.last_reaction=np.zeros(3)
        if trigger and not self.previous_trigger:
            if self.condition not in ('nails_only','combined'): self.last_gate='nails_disabled'
            elif self.nail_failed: self.last_gate='failed_nail_requires_replacement'
            elif self.depth>=p.nail_length_m-p.nail_seating_tolerance_m: self.last_gate='already_seated'
            elif not gate: self.last_gate='nose_contact_alignment_or_mating_invalid'
            else:
                self.drive_active=True; self.drive_elapsed=0.; self.last_gate='driving'
                self._event('drive_started')
        self.previous_trigger=bool(trigger)
        if self.drive_active:
            # Loss of actual contact stops a shot; re-trigger can continue an intact nail.
            if not gate:
                self.drive_active=False; self.last_gate='contact_lost'; self._event('drive_stopped',reason=self.last_gate,gap_m=gap,nose_force_N=float(nose_force_N),aligned=bool(aligned),mating_valid=bool(valid))
            else:
                duration=min(dt,max(0.,p.drive_timeout_s-self.drive_elapsed))
                delta,active,work,resistance_work=driving_increment(self.depth,p.drive_force_limit_N,duration,p,p.drive_energy_budget_J-self.drive_work)
                self.depth+=delta; self.drive_work+=work; self.penetration_work+=resistance_work
                self.drive_viscous_work+=max(0.,work-resistance_work); self.drive_elapsed+=dt
                self.driver_force=p.drive_force_limit_N*active/dt
                driven=self.driver_force*n
                # Equivalent resultant; split penetration reaction after receiver engagement.
                share=np.clip((self.depth-s.entry_thickness_m-max(gap,0.))/max(self.depth,1e-12),0.,1.)
                loads.extend([(s.a,(1-share)*driven,entry),(s.b,share*driven,entry),('nailer',-driven,entry)])
                self.last_reaction=-driven
                if self.depth>s.entry_thickness_m+max(gap,0.) and self.nail is None:
                    self.receiver_entry_m=s.entry_thickness_m+max(gap,0.)
                    self.nail=Connector.at(a,b,(a.world(s.face_a)+b.world(s.face_b))/2)
                    self._event('nail_receiver_engagement')
                if self.depth>=p.nail_length_m-1e-10 or self.drive_elapsed>=p.drive_timeout_s or active==0:
                    self.drive_active=False
                    self.last_gate='seated' if self.depth>=p.nail_length_m-p.nail_seating_tolerance_m else 'timeout_or_stall'
                    self._event('drive_stopped',depth_m=self.depth)
        self._glue_step(dt,a,b,glue_position,glue_axis,dispense,patch_clamp_N)
        self.stored_energy=0.
        if self.nail is not None:
            embed=max(0.,self.depth-self.receiver_entry_m)
            fraction=min(1.,embed/(p.nail_length_m-s.entry_thickness_m))
            if not self.nail.failed and fraction>0:
                self._connection(loads,self.nail,a,b,n,p.nail_stiffness_N_m*fraction,
                    p.nail_damping_N_s_m*fraction,p.nail_withdrawal_N*fraction,p.nail_lateral_N*fraction,
                    p.nail_failure_displacement_m,dt,'nail')
            if self.nail.failed:self.nail_failed=True
        for i,patch in enumerate(self.glue):
            if patch.connector is not None:
                alpha=cure_fraction(self.chemical_clock_s-patch.deposited_at_s,p.glue_cure_t95_s)
                coverage=min(1.,patch.volume_m3/(s.patch_area_m2*p.glue_film_m))
                scale=alpha*coverage
                self._connection(loads,patch.connector,a,b,n,p.glue_stiffness_full_N_m*scale,
                    p.glue_damping_N_s_m*scale,p.glue_tensile_strength_Pa*s.patch_area_m2*scale,
                    p.glue_shear_strength_Pa*s.patch_area_m2*scale,p.glue_failure_displacement_m,dt,f'glue_{i}')
        return loads

    def _connection(self,loads,connector,a,b,n,k,damping,normal_limit,shear_limit,limit,dt,label):
        was_failed=connector.failed
        force,point,energy,loss=connector.evaluate(a,b,n,k,damping,normal_limit,shear_limit,limit,dt)
        loads.extend([(self.s.a,force,point),(self.s.b,-force,point)])
        self.stored_energy+=energy; self.damping_work+=loss
        self.peak_connection_force=max(self.peak_connection_force,float(np.linalg.norm(force)))
        if connector.failed and not was_failed:self._event('connection_failed',connection=label)

    def _glue_step(self,dt,a,b,position,axis,dispense,clamp):
        p=self.p; s=self.s
        clamp=np.zeros(len(self.glue)) if clamp is None else np.asarray(clamp,float)
        if clamp.shape!=(len(self.glue),) or not np.isfinite(clamp).all() or np.any(clamp<0):
            raise ValueError('Invalid local clamp loads')
        if self.condition not in ('glue_only','combined'): return
        candidates=[]
        if dispense:
            if position is None or axis is None: raise ValueError('Dispensing needs actual nozzle pose')
            position,axis=np.asarray(position,float),np.asarray(axis,float)
            if position.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[position,axis]).all() or abs(np.linalg.norm(axis)-1)>1e-5:
                raise ValueError('Invalid nozzle state')
            n=a.rotation@s.normal_a
            for i,pa in enumerate(s.patches_a):
                delta=position-a.world(pa); normal=float(delta@n)
                if 0<=normal<=.002 and np.linalg.norm(delta-normal*n)<=p.glue_nozzle_radius_m and float(axis@n)<-math.cos(p.tool_angle_tolerance_rad):
                    candidates.append(i)
            supplied=p.glue_flow_m3_s*dt; self.dispensed_volume+=supplied; deposited=0.
            for i in candidates:
                patch=self.glue[i]
                # No automatic cleaning/rebonding of expired or broken adhesive.
                if patch.expired or patch.connector is not None:continue
                if patch.deposited_at_s is None:patch.deposited_at_s=self.chemical_clock_s
                added=min(supplied/max(1,len(candidates)),max(0.,s.patch_area_m2*p.glue_film_m-patch.volume_m3))
                patch.volume_m3+=added; deposited+=added
            self.wasted_volume+=supplied-deposited
        for i,patch in enumerate(self.glue):
            if patch.deposited_at_s is None or patch.connector is not None or patch.expired:continue
            age=self.chemical_clock_s-patch.deposited_at_s
            if age>p.glue_open_time_s:
                patch.expired=True; self._event('glue_open_time_expired',patch=i); continue
            valid,_=self._mating(a,b,s.patches_a[i],s.patches_b[i])
            covered=patch.volume_m3>=s.patch_area_m2*p.glue_film_m*(1-1e-9)
            patch.clamp_s=patch.clamp_s+dt if valid and covered and clamp[i]>=p.glue_clamp_force_per_patch_N else 0.
            if patch.clamp_s>=p.glue_continuous_clamp_s:
                point=(a.world(s.patches_a[i])+b.world(s.patches_b[i]))/2
                patch.connector=Connector.at(a,b,point); self._event('glue_bond_formed',patch=i)

    def sensor_packet(self):
        # Ideal tool sensors only; no latent joint strength, cure, damage or geometry.
        return {'nose_contact_switch':bool(self.nose_contact),'driver_stroke_m':float(self.depth),
                'driver_force_N':float(self.driver_force),'dispensed_volume_m3':float(self.dispensed_volume)}

    def evaluator_metrics(self):
        p=self.p; glue=[]
        for patch in self.glue:
            glue.append({'volume_m3':patch.volume_m3,'expired':patch.expired,'clamp_seconds':patch.clamp_s,
                'cure_fraction':0. if patch.deposited_at_s is None else cure_fraction(self.chemical_clock_s-patch.deposited_at_s,p.glue_cure_t95_s),
                'bonded':patch.connector is not None,'failed':patch.connector is not None and patch.connector.failed})
        glue_ready=all(x['bonded'] and not x['failed'] and x['cure_fraction']>=p.glue_handling_fraction for x in glue)
        seated=self.depth>=p.nail_length_m-p.nail_seating_tolerance_m and self.nail is not None and not self.nail_failed
        ready=(seated if self.condition=='nails_only' else glue_ready if self.condition=='glue_only' else seated and glue_ready if self.condition=='combined' else False)
        connectors=[x.connector for x in self.glue if x.connector is not None]+([self.nail] if self.nail is not None else [])
        return {'condition':self.condition,'nail_depth_m':self.depth,'nail_seated':bool(seated),'nail_failed':self.nail_failed,
                'glue':glue,'model_handling_ready':bool(ready),'physical_acceptance':False,
                'drive_work_J':self.drive_work,'penetration_dissipation_J':self.penetration_work,
                'drive_viscous_dissipation_J':self.drive_viscous_work,'connection_damping_dissipation_J':self.damping_work,
                'stored_elastic_energy_J':self.stored_energy,
                'fracture_dissipation_J':sum(x.fracture_energy_J for x in connectors),
                'stiffness_change_work_J':sum(x.stiffness_work_J for x in connectors),
                'peak_connection_force_N':self.peak_connection_force,'wasted_glue_m3':self.wasted_volume,
                'physical_time_s':self.clock_s,'chemical_time_s':self.chemical_clock_s,
                'last_drive_gate':self.last_gate,'calibration_status':'unmeasured_placeholder_parameters'}
