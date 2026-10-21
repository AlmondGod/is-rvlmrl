"""Offline constitutive regression report. Does not import/execute MuJoCo."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .fastening import Parameters,cure_fraction
from .test_fastening import setup,drive,tick,coat
from .design import ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    report={'scope':'prescribed-input constitutive checks; not robot dynamics or physical validation',
        'parameters':asdict(Parameters()),'measured_data_used':False,'mujoco_rollouts_executed':False,'cases':[]}
    for dt in (.001,.0005,.00025):
        j,a,b=setup('nails_only');drive(j,a,b,dt)
        # Prescribed 10 mm/s opening, with no rigid-body integration.
        reference=b.position.copy();failure=None
        for k in range(round(.6/dt)):
            b.position=reference+np.array([0,.01*(k+1)*dt,0]);tick(j,a,b,dt)
            if j.nail_failed:
                failure=.01*(k+1)*dt;break
        report['cases'].append({'dt_s':dt,'final_drive_depth_m':j.depth,'drive_input_work_J':j.drive_work,
            'drive_energy_balance_residual_J':j.drive_work-j.penetration_work-j.drive_viscous_work,
            'withdrawal_failure_displacement_m':failure,'metrics':j.evaluator_metrics()})
    ref=report['cases'][-1]
    for row in report['cases']:
        row['failure_displacement_relative_error']=abs(row['withdrawal_failure_displacement_m']-ref['withdrawal_failure_displacement_m'])/ref['withdrawal_failure_displacement_m']
        assert row['failure_displacement_relative_error']<=.01
        assert abs(row['drive_energy_balance_residual_J'])<=1e-10
    report['simpler_model_comparison']={
        'constant_full_strength_glue':{'normalized_strength_at_60s':1.,'normalized_strength_at_86400s':1.},
        'cure_dependent_glue':{'normalized_strength_at_60s':cure_fraction(60,86400),'normalized_strength_at_86400s':cure_fraction(86400,86400)},
        'interpretation':'Different predictions only. No measured data establishes which model is better.'}
    # File hashes bind evidence to this implementation and parameters.
    report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/box_assembly').glob('*.py')}
    report['status']='offline_constitutive_checks_passed_physical_validation_pending'
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(args.out/'report.json'),'cases':[{k:v for k,v in x.items() if k!='metrics'} for x in report['cases']]}))


if __name__=='__main__':main()
