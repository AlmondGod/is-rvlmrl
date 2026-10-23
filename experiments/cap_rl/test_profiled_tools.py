import unittest
import numpy as np
from experiments.torso_assembly.tool_skills import Tool
from .profiled_tools import SouthwestApproachTool
from .assembly import profile_options


class ProfiledToolTests(unittest.TestCase):
    def test_original_direction_is_bitwise_unchanged(self):
        axis = np.array([.1, np.sqrt(.99), 0.])
        for index in range(4):
            sign = -1 if index < 2 else 1
            np.testing.assert_array_equal(Tool.frame_direction(index, sign, axis),
                                          sign*.8*axis+np.array([0,0,-.6]))

    def test_candidate_matches_frozen_development_formula(self):
        axis = np.array([.1, np.sqrt(.99), 0.])
        for index in range(4):
            sign = -1 if index < 2 else 1
            horizontal = .9 if index == 0 else .8
            expected = sign*horizontal*axis+np.array([0,0,-np.sqrt(1-horizontal**2)])
            np.testing.assert_array_equal(SouthwestApproachTool.frame_direction(index, sign, axis), expected)

    def test_unknown_profiles_rejected(self):
        with self.assertRaises(ValueError):
            profile_options('relaxed', 'original')
        with self.assertRaises(ValueError):
            profile_options('contact_only', 'oracle')


if __name__ == '__main__':
    unittest.main()
