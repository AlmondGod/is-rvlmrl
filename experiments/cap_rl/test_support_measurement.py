import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.assembly_benchmark.evaluation import Evaluation
from .support_measurement import SupportEvaluation, supported_metrics


class SupportMeasurementTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.criteria = json.loads((root/'experiments/assembly_benchmark/protocol.json').read_text())['acceptance']
        self.metrics = dict(frame=dict(frame_geometry_ok=True, position_error_m=[0.]*4,
            yaw_error_deg=[0.]*4, tilt_deg=[0.]*4, finite=True, warnings=[0]*8),
            xy_error_m=0., height_error_m=0., tilt_deg=0.,
            rotation_evaluator_only=[[1,0,0],[0,1,0],[0,0,1]], panel_mass_kg=10.,
            forces=dict(frame=86.1, grip=[0.,0.], rack=0., other=0.))

    def stable(self, metrics, nail_force, evaluator=SupportEvaluation):
        instance = evaluator(self.criteria)
        env = SimpleNamespace(nails=SimpleNamespace(panel_support_z_N=nail_force))
        for _ in range(150):
            instance.update(env, metrics)
        return instance.stable

    def test_shared_support_counts_without_changing_contact_measurement(self):
        original = copy.deepcopy(self.metrics)
        self.assertAlmostEqual(self.stable(self.metrics, 12.), 3.)
        self.assertEqual(self.stable(self.metrics, 12., Evaluation), 0.)
        self.assertEqual(self.metrics, original)

    def test_downward_preload_is_signed(self):
        self.metrics['forces']['frame'] = 108.1
        self.assertAlmostEqual(self.stable(self.metrics, -10.), 3.)
        self.assertEqual(self.stable(self.metrics, 10.), 0.)

    def test_missing_support_fails(self):
        self.assertEqual(self.stable(self.metrics, 0.), 0.)

    def test_original_contact_only_case_unchanged(self):
        self.metrics['forces']['frame'] = 98.1
        self.assertAlmostEqual(self.stable(self.metrics, 0.), 3.)

    def test_external_support_or_grasp_still_fails(self):
        for key, value in [('rack', 1.), ('other', 1.), ('grip', [1.,0.])]:
            metrics = copy.deepcopy(self.metrics)
            metrics['forces'][key] = value
            self.assertEqual(self.stable(metrics, 12.), 0.)

    def test_bad_geometry_still_fails(self):
        self.metrics['xy_error_m'] = .007
        self.assertEqual(self.stable(self.metrics, 12.), 0.)

    def test_nonfinite_measurement_rejected(self):
        with self.assertRaises(ValueError):
            supported_metrics(self.metrics, float('nan'))

    def test_full_task_gates_are_retained(self):
        instance = SupportEvaluation(self.criteria)
        instance.preflip = self.metrics
        instance.preflip_stable = 3.
        instance.hold = 5.
        env = SimpleNamespace(peak=0., nails=SimpleNamespace(metrics=lambda: {'intact': 8}))
        self.assertTrue(instance.result(env, True, True, None)['passes'])
        env.nails.metrics = lambda: {'intact': 7}
        self.assertFalse(instance.result(env, True, True, None)['passes'])
        env.nails.metrics = lambda: {'intact': 8}
        for attribute, value in [('hold', 4.9), ('preflip_stable', 2.9),
                                 ('position_peak', .006), ('angle_peak', 3.1),
                                 ('nonfinite_or_warning', True)]:
            previous = getattr(instance, attribute)
            setattr(instance, attribute, value)
            self.assertFalse(instance.result(env, True, True, None)['passes'])
            setattr(instance, attribute, previous)
        env.peak = .00051
        self.assertFalse(instance.result(env, True, True, None)['passes'])


if __name__ == '__main__':
    unittest.main()
