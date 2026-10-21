"""Cloud physics diagnostics, not a successful policy demonstration."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from .design import Config,ROOT,manifest
from .environment import BoxAssembly,require_cloud


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/box_assembly/cloud_validation_v1')
    a=p.parse_args();require_cloud();a.out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','policy_demonstration':False,'design':manifest(Config()),'cases':[]}
    for dt in [.001,.0005,.00025]:
        env=BoxAssembly(replace(Config(),timestep=dt))
        # Constructibility and model invariants before interpreting dynamics.
        assert len(env.finger_actuators)==20
        assert env.model.neq==0,'This workcell must contain no equality attachment'
        assert all(env.model.body_mass[b]>0 for b in env.panel_ids.values())
        for assembled in [False,True]:
            env.reset(seed=10,diagnostic_assembled=assembled)
            initial=env.evaluator_metrics()
            assert not initial['full_assembly_success']
            if not assembled:assert not initial['dry_fit_success']
            for _ in range(200):
                m=env.step(env.wrist_target.copy(),env.wrist_quaternion.copy(),env.finger_target.copy())
            before_release=m
            env.fixture_release(True)
            for _ in range(300):
                m=env.step(env.wrist_target.copy(),env.wrist_quaternion.copy(),env.finger_target.copy())
            assert m['finite_state'] and not any(m['warnings']),m
            assert np.linalg.norm(env.applied_wrench[:3])<=env.config.wrist_force_limit+1e-8
            assert np.linalg.norm(env.applied_wrench[3:])<=env.config.wrist_torque_limit+1e-8
            row={'dt':dt,'diagnostic_assembled':assembled,'before_release':before_release,'after_release':m,
                'masses_kg':{n:float(env.model.body_mass[b]) for n,b in env.panel_ids.items()},
                'scene_sha256':hashlib.sha256(env.xml.encode()).hexdigest()}
            report['cases'].append(row)
            (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        if dt==.001:
            env.reset(diagnostic_assembled=False)
            with mujoco.Renderer(env.model,height=480,width=640) as renderer:
                obs=env.camera_observation(renderer)
                assert not any(k in obs for k in ['board_position','object_pose','contact_forces','metrics','reward'])
                imageio.imwrite(a.out/'initial_cameras.png',np.concatenate(list(obs['rgb']),axis=1))
                env.reset(diagnostic_assembled=True)
                obs=env.camera_observation(renderer)
                imageio.imwrite(a.out/'diagnostic_assembled_geometry.png',np.concatenate(list(obs['rgb']),axis=1))
            (a.out/'scene.xml').write_text(env.xml)
    report['status']='diagnostics_completed_not_hardware_calibrated'
    # A stable diagnostic placement is not evidence a robot assembled the box.
    report['policy_demonstration']=False
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
