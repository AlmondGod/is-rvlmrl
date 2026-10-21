"""Geometry/specification only. No simulator import or execution."""
from dataclasses import asdict, dataclass, field
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('bottom', 'front', 'back', 'left', 'right')


@dataclass(frozen=True)
class Config:
    length: float = .300
    width: float = .200
    height: float = .150
    thickness: float = .009
    corner_gap: float = .0003
    edge_bevel: float = .00025
    density: float = 600.
    wood_wood_friction: float = .45
    wood_bench_friction: float = .50
    wood_hand_friction: float = .65
    wood_fixture_friction: float = .30
    contact_time_constant: float = .008
    contact_damping_ratio: float = 1.
    contact_impedance: tuple = (.90,.99,.001,.5,2.)
    timestep: float = .0005  # 1 ms causes explicit wrist damping oscillation; see hold audit.
    control_period: float = .02
    release_hold_seconds: float = 3.
    fixture_clearance: float = .0005
    finger_target_speed: float = 1.5
    wrist_speed: float = .15
    wrist_angular_speed: float = 1.
    wrist_force_limit: float = 35.
    wrist_torque_limit: float = 3.
    wrist_position_gain: float = 600.
    wrist_position_damping: float = 35.
    wrist_rotation_gain: float = 12.
    wrist_rotation_damping: float = 1.2
    action_delay_steps: int = 2
    start_xy_jitter: float = .008
    start_yaw_jitter: float = .05
    success_position_tolerance: float = .0015
    success_angle_tolerance: float = .035
    success_penetration_tolerance: float = .0005
    calibration_status: str = 'uncalibrated_design_prior'
    panel_dimensions: dict = field(default_factory=dict)
    panel_masses: dict = field(default_factory=dict)

    def __post_init__(self):
        positive=('length','width','height','thickness','density','timestep','control_period',
            'contact_time_constant','contact_damping_ratio','release_hold_seconds','finger_target_speed',
            'wrist_speed','wrist_angular_speed','wrist_force_limit','wrist_torque_limit')
        if any(not math.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):raise ValueError('Physical sizes/rates must be finite and positive')
        if any(not math.isfinite(getattr(self,k)) or getattr(self,k)<0 for k in (
                'wood_wood_friction','wood_bench_friction','wood_hand_friction','wood_fixture_friction',
                'wrist_position_gain','wrist_position_damping','wrist_rotation_gain','wrist_rotation_damping',
                'start_xy_jitter','start_yaw_jitter')):raise ValueError('Invalid material/control parameter')
        if 2*self.thickness+2*self.corner_gap>=min(self.length,self.width):raise ValueError('Invalid panel fit')
        if self.height<=self.thickness:raise ValueError('No wall height')
        if not 0<=self.edge_bevel<self.thickness/3:raise ValueError('Bevel too large')
        if self.corner_gap<0 or self.fixture_clearance<0 or self.action_delay_steps<0:raise ValueError('Negative clearance/delay')
        if self.contact_time_constant<2*self.timestep:raise ValueError('Contact time constant below numerical resolution')
        if abs(self.control_period/self.timestep-round(self.control_period/self.timestep))>1e-8:raise ValueError('Nonintegral control substeps')
        for name,size in self.panel_dimensions.items():
            if name not in NAMES or len(size)!=3 or any(x<=2*self.edge_bevel for x in size):raise ValueError('Invalid measured panel dimensions')
        if any(n not in NAMES or mass<=0 for n,mass in self.panel_masses.items()):raise ValueError('Invalid panel masses')


def parts(c):
    L,W,H,t=c.length,c.width,c.height,c.thickness
    wall=H-t
    result={
        'bottom':{'size':[L,W,t],'target':[0,0,t/2],'start':[0,-.34,t/2],'start_quat':[1,0,0,0]},
        'front':{'size':[L,t,wall],'target':[0,-(W-t)/2,t+wall/2],'start':[-.38,-.27,t/2],'start_quat':[2**-.5,2**-.5,0,0]},
        'back':{'size':[L,t,wall],'target':[0,(W-t)/2,t+wall/2],'start':[-.38,.27,t/2],'start_quat':[2**-.5,2**-.5,0,0]},
        'left':{'size':[t,W-2*t-2*c.corner_gap,wall],'target':[-(L-t)/2,0,t+wall/2],'start':[.38,-.23,t/2],'start_quat':[2**-.5,0,2**-.5,0]},
        'right':{'size':[t,W-2*t-2*c.corner_gap,wall],'target':[(L-t)/2,0,t+wall/2],'start':[.38,.23,t/2],'start_quat':[2**-.5,0,2**-.5,0]},
    }
    for name,size in c.panel_dimensions.items():
        result[name]['size']=list(size)
        vertical_axis=2 if name=='bottom' else 1 if name in ('front','back') else 0
        result[name]['start'][2]=size[vertical_axis]/2
    return result


def beveled_vertices(size,bevel):
    """Convex chamfered cuboid, not a concave-joint approximation."""
    h=[v/2 for v in size]
    if not bevel:return [[s[i]*h[i] for i in range(3)] for s in itertools.product((-1,1),repeat=3)]
    return [[s[i]*(h[i]-(0 if i==axis else bevel)) for i in range(3)]
            for s in itertools.product((-1,1),repeat=3) for axis in range(3)]


def manifest(c):
    return {'design':'Open-top wooden box: four butt-jointed walls on a full bottom panel',
        'stage':'dry_fit_in_physical_fixture_before_fastening','config':asdict(c),'parts':parts(c),
        'confirmed_construction':{'parts':'separate wooden panels','fastening':['nails','wood glue']},
        'fastening_specification':'experiments/box_assembly/fastening_spec.json',
        'fastening_physics_implemented':True,
        'fastening_enabled_in_default_workcell':False,
        'fastening_scope':'Optional FasteningBoxAssembly: one front-left joint, uncalibrated reduced-order laws',
        'measured_parameters':[],
        'assumed_parameters':'All dimensions, material/contact/actuation and sensor parameters are design priors until measured.',
        'not_modelled':['impact-tool dynamics, nail plastic bending and head pull-through','adhesive fluid flow and temperature-dependent cure','wood bending/crushing/splintering',
            'moisture deformation','real arm kinematics, transmission and dynamics'],
        'full_assembly_success_available':False,
        'robot':'Official Wuji Hand 2.2 on force/torque-limited virtual positioning stage; no real arm claim',
        'policy_observation':'RGB plus robot proprioception; evaluator-only object state',
        'sources':['https://mujoco.readthedocs.io/en/3.3.3/modeling.html',
            'https://research.fs.usda.gov/fpl/wood-handbook']}


def schematic(c):
    def project(p):
        x,y,z=p;return (250+700*(x-.65*y),290+700*(.35*x+.40*y-z))
    colors=['#c8a369','#af783f','#c7955c','#deb57b','#bd8850']
    groups=[]
    for offset,title,explode in [(0,'Nominal fitted geometry',False),(550,'Exploded panel layout',True)]:
        shapes=[]
        for (name,item),color in zip(parts(c).items(),colors):
            center=list(item['target'])
            if explode:
                if name=='bottom':center[2]-=.045
                else:
                    center[0]*=1.5;center[1]*=1.8;center[2]+=.055
            corners=[tuple(center[i]+sign[i]*item['size'][i]/2 for i in range(3)) for sign in itertools.product((-1,1),repeat=3)]
            for face in [(1,3,7,5),(2,3,7,6),(4,5,7,6)]:
                points=' '.join(f'{project(corners[i])[0]+offset:.1f},{project(corners[i])[1]:.1f}' for i in face)
                shapes.append(f'<polygon points="{points}" fill="{color}" fill-opacity=".80" stroke="#563c25" stroke-width="1.2"/>')
        groups.append(f'<text x="{offset+25}" y="58" fill="#e5edf5" font-size="20">{title}</text>'+''.join(shapes))
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="440" viewBox="0 0 1100 440">'+\
        '<rect width="1100" height="440" fill="#15212c"/><g font-family="Arial,sans-serif">'+''.join(groups)+\
        f'<text x="25" y="375" fill="#e5edf5" font-size="18">Assumed design: {c.length*1000:.0f} × {c.width*1000:.0f} × {c.height*1000:.0f} mm; {c.thickness*1000:.0f} mm panels</text>'+\
        '<text x="25" y="408" fill="#b8c8d8" font-size="16">Geometry schematic only. Fixture/robot omitted. No simulation or assembly success demonstrated.</text></g></svg>'


def export(out,c=Config()):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'design.json').write_text(json.dumps(manifest(c),indent=2)+'\n')
    (out/'box_design.svg').write_text(schematic(c))


if __name__=='__main__':export(ROOT/'outputs/box_assembly/design')
