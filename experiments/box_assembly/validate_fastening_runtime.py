"""Run ONLY on an authorized non-RunPod Linux host. No success-demo claim.

Compile/contact/render smoke check. This is not a full manipulation benchmark or
physical-validation substitute. Deliberately never creates a pre-fastened reset.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from .design import Config
from .environment import require_cloud
from .fastening_environment import FasteningBoxAssembly


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require_cloud();a.out.mkdir(parents=True,exist_ok=False)
    report={'cases':[],'policy_demonstration':False,'physical_validation':False}
    for dt in (.001,.0005,.00025):
        e=FasteningBoxAssembly(replace(Config(),timestep=dt))
        assert e.model.neq==0
        assert len(e.finger_actuators)==20
        for assembled in (False,True):
            e.reset(diagnostic_assembled=assembled)
            for _ in range(100):
                metrics=e.step(e.wrist_target.copy(),e.wrist_quaternion.copy(),e.finger_target.copy())
            assert metrics['finite_state'] and not any(metrics['warnings'])
            assert e.process.nail is None and all(x.connector is None for x in e.process.glue)
            assert not metrics['full_assembly_success']
            assert all(np.linalg.norm(w[:3])<=150+1e-9 and np.linalg.norm(w[3:])<=5+1e-9 for w in e.tool_wrenches.values())
            report['cases'].append({'dt_s':dt,'diagnostic_assembled':assembled,'metrics':metrics})
        if dt==.001:
            with mujoco.Renderer(e.model,height=480,width=640) as r:
                obs=e.camera_observation(r)
                imageio.imwrite(a.out/'diagnostic_tool_workcell.png',np.concatenate(list(obs['rgb']),axis=1))
            (a.out/'scene.xml').write_text(e.xml)
    report['status']='uncommanded_runtime_smoke_passed_active_tool_and_release_validation_still_required'
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
