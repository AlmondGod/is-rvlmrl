"""Synthetic fixtures test the data pipeline, not material predictions."""
import copy
import unittest
import numpy as np
from .calibrate_drive import fit
from .validate_physical import compare

class ValidationTests(unittest.TestCase):
    def rows(self):
        return [{'split':'calibration','specimen_id':f'cal_{i}','tool_regime':'quasistatic',
                 'depth_m':d,'speed_m_s':v,'driver_force_N':35+1800*d+500*v}
                for i,(d,v) in enumerate(( (d,v) for d in (.001,.01,.02) for v in (.005,.02))) ]
    def test_fit_identifiable_synthetic_data(self):
        result=fit(self.rows());np.testing.assert_allclose(list(result['parameters'].values()),[35,1800,500],rtol=1e-12)
    def test_fit_rejects_leakage_and_impact(self):
        for key,value in [('split','heldout'),('tool_regime','impact')]:
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(ValueError):fit(rows)
    def test_fit_rejects_unidentifiable_speed(self):
        rows=self.rows()
        for row in rows:row['speed_m_s']=.01
        with self.assertRaises(ValueError):fit(rows)
    def fixtures(self):
        observed={'source':'physical_measurement','records':[{'specimen_id':'heldout_1','split':'heldout','stock_batch':'B',
            'condition':'combined','mode':'withdrawal','age_group':'handling',
            'displacement_m':[0,.001,.002],'force_N':[0,10,0],'failure_displacement_m':.001,
            'failure_mode':'withdrawal','release_stable':True}]}
        pred=copy.deepcopy(observed['records'][0]);pred['recommend_release']=True
        return observed,{'frozen_parameter_sha256':'synthetic_test_only','registration_reference':'synthetic_test_only',
             'calibration_specimen_ids':['cal_1'],'calibration_stock_batches':['A'],'models':{'cure_dependent':[pred],'constant_joint':[copy.deepcopy(pred)]}}
    def test_no_measurements_never_passes(self):
        self.assertFalse(compare({'source':'physical_measurement','records':[]},{})['physical_validation_passed'])
    def test_perfect_single_case_not_validation(self):
        a,b=self.fixtures();r=compare(a,b)
        self.assertFalse(r['physical_validation_passed']);self.assertAlmostEqual(r['models']['cure_dependent']['false_safe_upper_95'],.95)
    def test_batch_and_specimen_leakage(self):
        for field,value in [('stock_batch','A'),('specimen_id','cal_1')]:
            a,b=self.fixtures();a['records'][0][field]=value
            with self.assertRaises(ValueError):compare(a,b)
    def test_missing_prediction_or_extrapolation_rejected(self):
        a,b=self.fixtures();b['models']['cure_dependent'][0]['displacement_m']=[0,.0005,.001]
        with self.assertRaises(ValueError):compare(a,b)
