import unittest
from dataclasses import replace
from .decisions import Evidence,DecisionLimits,decide

class DecisionTests(unittest.TestCase):
    def test_nonfinite_measurements_are_rejected(self):
        with self.assertRaises(ValueError):Evidence(alignment_error_m=float("nan"))
        with self.assertRaises(ValueError):DecisionLimits(minimum_handling_s=float("inf"))
    def test_unknown_is_not_success(self):
        self.assertEqual(decide(Evidence())['action'],'inspect')
    def test_latent_strength_cannot_authorize_release(self):
        e=Evidence(alignment_error_m=0,head_protrusion_m=0,visible_split=False,driver_stroke_m=.025,
            clamp_history_verified=True,elapsed_since_glue_s=86400,proof_test_passed=True,measurements_credible=True)
        self.assertEqual(decide(e)['action'],'hold_for_calibration')
        limits=DecisionLimits(validated_for_material_and_tool=True,minimum_handling_s=3600)
        self.assertEqual(decide(e,limits)['action'],'release_support')
        self.assertEqual(decide(replace(e,proof_test_passed=None),limits)['action'],'probe_with_support')
        self.assertEqual(decide(replace(e,proof_test_passed=False),limits)['action'],'reject_or_repair_offline')
    def test_cure_and_seating_distinct(self):
        e=Evidence(alignment_error_m=0,head_protrusion_m=.003,visible_split=False,driver_stroke_m=.02,measurements_credible=True)
        self.assertEqual(decide(e)['action'],'inspect_for_redrive')
        self.assertEqual(decide(replace(e,visible_split=True))['action'],'reject_or_repair_offline')
