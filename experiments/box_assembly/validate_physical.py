"""Compare frozen per-specimen predictions with held-out physical measurements.

The report cannot certify data authenticity or prospective registration; retain
raw instrument files and an externally timestamped prediction hash separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

THRESHOLDS={'force_curve_nrmse':.15,'peak_load_relative_error':.20,
            'failure_displacement_relative_error':.20,'failure_displacement_floor_m':.0005,
            'failure_mode_accuracy':.80,'release_classification_accuracy':.90,
            'false_safe_upper_95':.05,'complexity_minimum_relative_rmse_improvement':.10}


def compare(observed,predicted):
    if observed.get('source')!='physical_measurement':raise ValueError('Physical measurements required')
    records=observed.get('records',[])
    if not records:return {'status':'blocked_no_physical_measurements','physical_validation_passed':False}
    if not predicted.get('frozen_parameter_sha256') or not predicted.get('registration_reference'):
        raise ValueError('Need frozen parameters and prospective prediction registration reference')
    calibration=set(predicted.get('calibration_specimen_ids',[]));calibration_batches=set(predicted.get('calibration_stock_batches',[]))
    ids=[r['specimen_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('One independent destructive test per specimen record')
    if set(ids)&calibration or any(r['split']!='heldout' for r in records):raise ValueError('Calibration/held-out leakage')
    if not calibration_batches or any(r['stock_batch'] in calibration_batches for r in records):
        raise ValueError('Record and hold out stock batches, not only time samples')
    out={'status':'evaluated','thresholds':THRESHOLDS,'models':{},'physical_validation_passed':False,
         'independent_specimens':len(records),'data_authenticity_verified':False}
    for name,preds in predicted.get('models',{}).items():
        if len({x['specimen_id'] for x in preds})!=len(preds):raise ValueError('Duplicate prediction')
        by_id={r['specimen_id']:r for r in preds}
        if set(by_id)!=set(ids):raise ValueError('All models must predict the same held-out specimens')
        rows=[];false_safe=0;release_count=0
        for r in records:
            p=by_id[r['specimen_id']]
            x=np.asarray(r['displacement_m'],float);y=np.asarray(r['force_N'],float)
            xp=np.asarray(p['displacement_m'],float);yp=np.asarray(p['force_N'],float)
            if len(x)<3 or len(xp)<3 or x.shape!=y.shape or xp.shape!=yp.shape or not np.isfinite(np.r_[x,y,xp,yp]).all():
                raise ValueError('Invalid curve')
            if np.any(np.diff(x)<=0) or np.any(np.diff(xp)<=0) or x[0]<xp[0] or x[-1]>xp[-1]:
                raise ValueError('Curves must span the prescribed monotonic test ramp; do not extrapolate')
            scale=max(float(np.max(np.abs(y))),5.)
            rmse=float(np.sqrt(np.mean((y-np.interp(x,xp,yp))**2)))/scale
            peak_error=abs(float(np.max(yp)-np.max(y)))/scale
            failure_error=abs(float(p['failure_displacement_m'])-float(r['failure_displacement_m']))
            failure_limit=max(THRESHOLDS['failure_displacement_floor_m'],THRESHOLDS['failure_displacement_relative_error']*abs(float(r['failure_displacement_m'])))
            for entry,key in ((r,'release_stable'),(p,'recommend_release')):
                if type(entry[key]) is not bool:raise ValueError('Release outcomes must be Boolean measured/tested values')
            release_count+=int(p['recommend_release']);false_safe+=int(p['recommend_release'] and not r['release_stable'])
            rows.append({'specimen_id':r['specimen_id'],'force_curve_nrmse':rmse,'peak_load_relative_error':peak_error,
                'failure_displacement_pass':failure_error<=failure_limit,'failure_mode_correct':p['failure_mode']==r['failure_mode'],
                'release_correct':p['recommend_release']==r['release_stable']})
        # Exact one-sided binomial upper confidence bound (zero-event special case).
        # With any false-safe cases, conservatively fail this pilot's zero-event gate.
        upper=1-.05**(1/release_count) if release_count and false_safe==0 else 1.
        result={'specimens':rows,'median_curve_nrmse':float(np.median([x['force_curve_nrmse'] for x in rows])),
            'failure_mode_accuracy':float(np.mean([x['failure_mode_correct'] for x in rows])),
            'release_accuracy':float(np.mean([x['release_correct'] for x in rows])),
            'false_safe_count':false_safe,'recommended_release_count':release_count,'false_safe_upper_95':upper}
        result['passes_predeclared_thresholds']=bool(
            all(x['force_curve_nrmse']<=.15 and x['peak_load_relative_error']<=.20 and x['failure_displacement_pass'] for x in rows)
            and result['failure_mode_accuracy']>=.8 and result['release_accuracy']>=.9 and upper<=.05)
        out['models'][name]=result
    if not out['models']:raise ValueError('No predictions')
    if 'cure_dependent' in out['models'] and 'constant_joint' in out['models']:
        complex_errors=np.array([x['force_curve_nrmse'] for x in out['models']['cure_dependent']['specimens']])
        simple_errors=np.array([x['force_curve_nrmse'] for x in out['models']['constant_joint']['specimens']])
        gains=(simple_errors-complex_errors)/np.maximum(simple_errors,1e-9)
        rng=np.random.default_rng(31415)
        boot=np.median(gains[rng.integers(0,len(gains),size=(2000,len(gains)))],axis=1)
        out['complexity_comparison']={'median_relative_curve_error_improvement':float(np.median(gains)),
            'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'justified_in_this_dataset':bool(np.median(gains)>=.1 and np.quantile(boot,.025)>0
                and out['models']['cure_dependent']['release_accuracy']>=out['models']['constant_joint']['release_accuracy']
                and out['models']['cure_dependent']['false_safe_count']<=out['models']['constant_joint']['false_safe_count']
                and out['models']['cure_dependent']['failure_mode_accuracy']>=out['models']['constant_joint']['failure_mode_accuracy']),
            'limitation':'Specimen bootstrap; correlated stock batches require a larger batch-level confirmation study.'}
    required={(condition,mode,age) for condition in ('unfastened','nails_only','glue_only','combined') for mode in ('withdrawal','lateral') for age in (('handling','cured') if condition in ('glue_only','combined') else ('not_applicable',))}
    present={(r['condition'],r['mode'],r['age_group']) for r in records}
    out['missing_design_cells']=[list(x) for x in sorted(required-present)]
    out['heldout_curve_and_release_gate_passed']=bool(not out['missing_design_cells'] and out['models'].get('cure_dependent',{}).get('passes_predeclared_thresholds',False))
    out['physical_validation_passed']=False
    out['remaining_validation']=['sensor repeatability','seating metrology','recovery ranking','coupled robot dynamics','independent data/prospective-registration audit']
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--observed',type=Path,required=True);p.add_argument('--predicted',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=compare(json.loads(a.observed.read_text()),json.loads(a.predicted.read_text()))
    report['file_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.observed,a.predicted)}
    with a.out.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
