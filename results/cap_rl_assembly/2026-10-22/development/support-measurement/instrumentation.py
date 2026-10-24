"""Read-only instrumentation of a scripted development replay; never policy input."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'is-rvlmrl'))
from experiments.assembly_benchmark.evaluation import Evaluation
from experiments.cap_rl.assembly import run
from experiments.torso_assembly.fasteners import NailBank
original_apply = NailBank.apply

def apply(self):
    original_apply(self)
    if self.env.d.time >= 400:
        joint = self.env.m.body_jntadr[self.env.panel]
        dof = self.env.m.jnt_dofadr[joint]
        self.measured_panel_z_N = float(self.env.d.qfrc_applied[dof+2])
NailBank.apply = apply
original = Evaluation.update
rows = []
last = -1

def update(self, env, metrics):
    global last
    original(self, env, metrics)
    tick = round(env.d.time*10)
    if tick == last or env.d.time < 400 or self.ref is not None:
        return
    last = tick
    joint = env.m.body_jntadr[env.panel]
    dof = env.m.jnt_dofadr[joint]
    rows.append(dict(time_s=float(env.d.time), stable_s=self.stable,
        contact_frame_N=metrics['forces']['frame'],
        nail_applied_z_N=getattr(env.nails,'measured_panel_z_N',None),
        gravity_N=metrics['panel_mass_kg']*9.81,
        panel_z_acceleration=float(env.d.qacc[dof+2]),
        intact_nails=env.nails.metrics()['intact'],
        gripper_N=metrics['forces']['grip'], rack_N=metrics['forces']['rack']))
Evaluation.update = update
out=Path(sys.argv[1])
result=run(41001,Path(__file__).with_name('assembly-development.py').read_text(),out,progress=True,seconds=535.)
(out/'support-audit.json').write_text(json.dumps(dict(
    scope='Read-only measurement of existing controller/evaluator during scripted training-layout replay. Measurement at the original internal force application, before the simulator clears its force buffer. Truncated at 535 simulated seconds for diagnosis only; not a full benchmark trial. No changes to dynamics, control, acceptance or reward; no policy input.',
    instrumentation_source=Path(__file__).read_text(),rows=rows),indent=2))
print(json.dumps({k:result[k] for k in ('success','reward','preflip_stable_s','wall_s')}))
