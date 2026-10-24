"""Numerical equivalence checks for the opt-in execution backend."""
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from analysis.assembly_speed import audit_contacts, cross3, frame_velocity, install
from experiments.box_assembly.fastening import Frame, Connector
from experiments.torso_assembly.image_ops import mask_pixels


class SpeedEquivalence(unittest.TestCase):
    def test_physical_steps_match_exactly(self):
        from experiments.torso_assembly.environment import Env
        from experiments.arm_corner.environment import Env as CornerEnv
        from experiments.torso_assembly.fasteners import NailBank
        original_step, original_frame, original_velocity = CornerEnv.step, NailBank.frame, Frame.velocity
        reference = Env(dt=.0005, layout_seed=41002, physical_nails=True)
        states = []
        for _ in range(100):
            reference.step()
            states.append(np.r_[reference.d.qpos, reference.d.qvel, reference.d.ctrl])
        try:
            install()
            candidate = Env(dt=.0005, layout_seed=41002, physical_nails=True)
            for state in states:
                candidate.step()
                np.testing.assert_array_equal(state, np.r_[candidate.d.qpos, candidate.d.qvel, candidate.d.ctrl])
            self.assertEqual(reference.peak, candidate.peak)
            self.assertEqual(reference.worst, candidate.worst)
            self.assertEqual(reference.saturation, candidate.saturation)
        finally:
            CornerEnv.step, NailBank.frame, Frame.velocity = original_step, original_frame, original_velocity

    def test_mask_coordinates_exact(self):
        rng = np.random.default_rng(194)
        for shape in ((1, 1), (30, 50), (960, 1280)):
            for probability in (0., .01, .5, 1.):
                mask = (rng.random(shape) < probability).astype(np.uint8)*255
                for view in (mask, mask[::-1, ::2]):
                    for a, b in zip(np.where(view), mask_pixels(view)):
                        self.assertEqual(a.dtype, b.dtype)
                        np.testing.assert_array_equal(a, b)

    def test_mask_float_conversion_unnecessary_for_all_rgb_values(self):
        g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
        for r in range(256):
            for threshold, ratio in ((70, 1.02), (35, 1.05), (70, 1.05)):
                original = ((float(r)>threshold)&(float(r)>ratio*g.astype(float))&
                            (g.astype(float)>1.18*b.astype(float)))
                candidate = ((r>threshold)&(r>ratio*g)&(g>1.18*b))
                np.testing.assert_array_equal(original, candidate)

    def test_contact_audit_preserves_peak_and_first_tie(self):
        class Contacts(list):
            @property
            def dist(self):
                return np.array([c.dist for c in self])
        rng = np.random.default_rng(89)
        cases = [[], [0., 0.], [-.001, -.003, -.003], [np.nan, -.002, np.nan],
                 [-np.inf, -.003], *rng.normal(size=(100, 40))]
        for distances in cases:
            contacts = Contacts(SimpleNamespace(dist=v, geom1=i, geom2=i+1)
                                for i, v in enumerate(distances))
            env = SimpleNamespace(d=SimpleNamespace(time=3., contact=contacts),
                                  m=SimpleNamespace(geom=lambda i: SimpleNamespace(name=str(i))),
                                  peak=0., worst={})
            peak, worst = 0., {}
            for c in contacts:
                if -c.dist > peak:
                    peak = float(-c.dist)
                    worst = dict(time=3., pair=[str(c.geom1), str(c.geom2)], depth_m=peak)
            audit_contacts(env)
            self.assertEqual(env.peak, peak)
            self.assertEqual(env.worst, worst)

    def test_cross_matches_bit_for_bit(self):
        rng = np.random.default_rng(718)
        vectors = rng.normal(size=(2000, 2, 3))
        vectors[:4] = [[[0., -0., 1.], [-0., 1., 0.]],
                       [[1e100, 1e100, -1e100], [1e100, -1e100, 1e100]],
                       [[1e-100, -1e-100, 1e-100], [1e-100, 1e-100, -1e-100]],
                       [[1., 1., 1.], [1., 1., 1.]]]
        for a, b in vectors:
            self.assertEqual(np.cross(a, b).tobytes(), cross3(a, b).tobytes())

    def test_connector_force_failure_and_energy_match(self):
        rng = np.random.default_rng(97)
        original = Frame.velocity
        for scale in (0., .0001, .0039, .004, .0041, .02):
            for _ in range(100):
                a = Frame(rng.normal(size=3), np.eye(3), rng.normal(size=3), rng.normal(size=3))
                b = Frame(a.position+rng.normal(size=3)*scale, np.eye(3),
                          rng.normal(size=3), rng.normal(size=3))
                c = Connector(np.zeros(3), np.zeros(3))
                other = copy.deepcopy(c)
                axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
                args = (a, b, axis, 12000., 4., 45., 70., .004, .0005)
                with patch.object(Frame, 'velocity', original):
                    reference = c.evaluate(*args)
                with patch.object(Frame, 'velocity', frame_velocity):
                    accelerated = other.evaluate(*args)
                for x, y in zip(reference, accelerated):
                    self.assertEqual(np.asarray(x).tobytes(), np.asarray(y).tobytes())
                for key in ('failed', 'fracture_energy_J', 'peak_utilization',
                            'stiffness_work_J', 'previous_k'):
                    self.assertEqual(getattr(c, key), getattr(other, key))


if __name__ == '__main__':
    unittest.main()
