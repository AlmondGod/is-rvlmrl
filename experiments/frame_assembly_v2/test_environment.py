"""Physics and success-criterion regression tests; no rendering required."""
import json
import unittest

import mujoco
import numpy as np

from .environment import Config, FrameAssembly, NAMES, ROOT


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = FrameAssembly()

    def setUp(self):
        self.env.reset()

    def settle(self, seconds=3.):
        for _ in range(round(seconds / self.env.dt)):
            self.env.step()

    def test_separate_rails_do_not_succeed(self):
        self.settle(.5)
        self.assertFalse(self.env.metrics()["success"])

    def test_assembled_released_frame_is_stable(self):
        self.env.reset(assembled=True)
        self.settle()
        m = self.env.metrics()
        self.assertTrue(m["success"], m)
        self.assertEqual(len(m["lap_contact_pairs"]), 4)
        self.assertTrue(m["all_laps_loaded"])
        upper_weight = sum(self.env.model.body_mass[self.env.bids[n]] for n in NAMES[2:]) * 9.81
        self.assertAlmostEqual(sum(m["lap_normal_forces_N"].values()), upper_weight, delta=.02)
        self.assertLess(m["max_board_penetration_m"], .001)
        self.assertTrue(all(w.number == 0 for w in self.env.data.warning))

    def test_assistance_disabled_by_default(self):
        with self.assertRaises(RuntimeError):
            self.env.set_assistance("west", True)

    def test_held_frame_cannot_succeed(self):
        self.env.reset(assembled=True)
        self.settle()
        # Exercise the detector independently of the API permission gate.
        self.env.data.eq_active[self.env.carries["west"]] = 1
        self.assertFalse(self.env.metrics()["success"])

    def test_lifted_piece_invalidates_success(self):
        self.env.reset(assembled=True)
        self.settle()
        qa = self.env.model.joint("east_free").qposadr[0]
        self.env.data.qpos[qa + 2] += .05
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(self.env.metrics()["success"])
        self.env.step()
        self.assertEqual(self.env.stable_seconds, 0.)

    def test_misaligned_piece_cannot_succeed(self):
        self.env.reset(assembled=True)
        qa = self.env.model.joint("north_free").qposadr[0]
        self.env.data.qpos[qa] += .025
        self.settle()
        self.assertFalse(self.env.metrics()["success"])

    def test_reset_seed_is_repeatable(self):
        self.env.reset(seed=17)
        initial = self.env.data.qpos.copy()
        self.settle(.2)
        self.env.reset(seed=17)
        np.testing.assert_array_equal(initial, self.env.data.qpos)
        self.env.reset(seed=18)
        self.assertFalse(np.array_equal(initial, self.env.data.qpos))

    def test_wood_mass_matches_half_lap_volume(self):
        c = self.env.config
        for n in NAMES:
            length = c.outer_x if n in NAMES[:2] else c.outer_y
            expected = (length - c.rail_width - c.shoulder_clearance) * c.rail_width * c.thickness * c.wood_density
            self.assertAlmostEqual(self.env.model.body_mass[self.env.bids[n]], expected)

    def test_wrist_translation_is_rate_limited(self):
        old = self.env.data.mocap_pos[self.env.wrist_id].copy()
        self.env.step(wrist_position=old + [1, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(self.env.data.mocap_pos[self.env.wrist_id] - old), .25 * self.env.dt)

    def test_finger_actuator_moves_a_joint(self):
        action = self.env.data.ctrl.copy()
        i = self.env.model.actuator("r_FFJ2").id
        action[i] = .4
        for _ in range(50):
            self.env.step(finger_angles=action)
        self.assertGreater(self.env.data.qpos[self.env.finger_qadr[i]], .1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AssemblyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = ROOT / "outputs/frame_assembly/test_results.json"
    out.write_text(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "tests": unittest.defaultTestLoader.getTestCaseNames(AssemblyTests)}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
