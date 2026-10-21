"""Fit R0 + k*depth + c*speed from slow, steady drive measurements only.

Impact nailer traces need an inertial model and are explicitly unsupported here.
Fits only split=calibration. Columns are SI; matrix conditioning checks identifiability.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def fit(rows):
    if not rows or any(r['split']!='calibration' for r in rows):
        raise ValueError('Supply calibration specimens only; never held-out rows')
    if any(r['tool_regime']!='quasistatic' for r in rows):
        raise ValueError('This feed model cannot identify impact nailer dynamics')
    X=np.array([[1.,float(r['depth_m']),float(r['speed_m_s'])] for r in rows])
    y=np.array([float(r['driver_force_N']) for r in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any(X[:,1:]<0):raise ValueError('Invalid measurement')
    scales=np.linalg.norm(X,axis=0)
    if np.any(scales==0):raise ValueError('Insufficient depth/speed excitation')
    normalized=X/scales
    if np.linalg.matrix_rank(normalized)!=3 or np.linalg.cond(normalized)>100:
        raise ValueError('Drive parameters not identifiable: vary depth AND controlled speed')
    coefficients=np.linalg.lstsq(normalized,y,rcond=None)[0]/scales
    if np.any(coefficients<=0):raise ValueError('Fit violates positive resistance; inspect data/model')
    residual=y-X@coefficients
    return {'parameters':dict(zip(('drive_intercept_N','drive_slope_N_m','drive_drag_N_s_m'),coefficients.tolist())),
        'training_RMSE_N':float(np.sqrt(np.mean(residual**2))),'scaled_condition_number':float(np.linalg.cond(normalized)),
        'calibration_specimens':sorted({r['specimen_id'] for r in rows}),
        'status':'calibration_fit_only_not_heldout_validation'}


def main():
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with a.csv.open() as f: result=fit(list(csv.DictReader(f)))
    result['input_sha256']=hashlib.sha256(a.csv.read_bytes()).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
