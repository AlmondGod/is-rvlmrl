"""Synthetic numerical regression for fused substeps, never a task reward."""
import gc
import json
import pickle
import time
from unittest.mock import patch

import numpy as np


def main():
    from analysis.assembly_async_rewards import check_memory_budget
    budget=check_memory_budget(2)
    from experiments.cap_rl import task_assembly as task
    from experiments.cap_rl.support_measurement import install_observer
    from experiments.torso_assembly.environment import Env
    from experiments.torso_assembly import native_substeps,acceleration
    from experiments.box_assembly.fastening import Connector
    task.configure(assembly_backend='vectorized_fused')
    original_advance=native_substeps.advance
    records=[]
    for observed in (False,True):
        baseline=Env(dt=.0005,layout_seed=51002,physical_nails=True)
        if observed:install_observer(baseline)
        for k,(a,b) in enumerate(zip(baseline.woods[:3],baseline.woods[1:])):
            fa,fb=baseline.nails.frame(a),baseline.nails.frame(b)
            baseline.nails.connections.append(dict(a=a,b=b,connector=Connector.at(fa,fb,(fa.position+fb.position)/2),
                axis_a=np.array([0.,0.,1.]),fraction=.4+.2*k,key=('frame',k)))
        data=pickle.dumps(baseline,protocol=5);candidate=pickle.loads(data);del data
        def select(env,*args):
            return False if env is baseline else original_advance(env,*args)
        timing=[0.,0.]
        with patch.object(native_substeps,'advance',select):
            for step in range(100):
                if step==40:
                    for env in (baseline,candidate):
                        for joint in env.nails.connections:joint['connector'].anchor_a[0]+=.01
                for i,env in enumerate((baseline,candidate)):
                    override={0:env.qtarget[0].copy()} if step%3==0 else None
                    start=time.monotonic()
                    env.step(env.targets.copy(),env.rotations.copy(),env.gripper.copy(),override)
                    timing[i]+=time.monotonic()-start
                for name in ('qpos','qvel','ctrl','qacc','qacc_warmstart','xpos','xmat','qfrc_applied','actuator_force'):
                    np.testing.assert_array_equal(getattr(baseline.d,name),getattr(candidate.d,name),err_msg=f'{observed}:{step}:{name}')
                assert baseline.nails.metrics()==candidate.nails.metrics()
                assert (baseline.peak,baseline.worst,baseline.saturation)==(candidate.peak,candidate.worst,candidate.saturation)
                if observed:assert baseline.nails.panel_support_z_N==candidate.nails.panel_support_z_N
        assert len(baseline.nails.failures)==3
        records.append(dict(observed_support=observed,steps=100,internal_steps=4000,failures=3,exact=True,seconds=timing))
        # Reject native invariants as infrastructure, outside the skill-error catch.
        candidate.d.xfrc_applied[0,0]=1.
        try:
            ids=np.concatenate(candidate.aids);limits=candidate.m.actuator_forcerange[ids,1]-.01
            original_advance(candidate,ids,limits,candidate.nails.apply)
        except native_substeps.NativeBackendError:pass
        else:raise AssertionError('Invalid applied force was accepted')
        del candidate,baseline;gc.collect()
    manifest=task.provenance()
    assert manifest['execution_backend']=='vectorized_fused'
    assert all(name in manifest for name in ('native_pixels','native_connectors','native_substeps'))
    for backend in ('original','vectorized','vectorized_pixels','vectorized_native'):
        try:acceleration.configure(backend)
        except RuntimeError:pass
        else:raise AssertionError('Fused backend silently disabled')
    print(json.dumps(dict(passes=True,rows=records,memory_budget=budget,native=manifest['native_substeps'],
        scope='Synthetic numerical/controller regression, no physical task reward or episode throughput claim.')),flush=True)


if __name__=='__main__':main()
