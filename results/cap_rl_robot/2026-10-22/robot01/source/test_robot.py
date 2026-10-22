"""Physical manipulation and reward counterexamples, not policy demonstrations."""
import unittest
import numpy as np
from . import task_robot as task
from .robot_lift import Robot, model


class RobotTests(unittest.TestCase):
    def test_no_workpiece_attachment_or_external_force(self):
        m = model()
        self.assertEqual(m.nmocap, 0)
        self.assertEqual(m.neq, 3)  # only native gripper linkage constraints
        self.assertTrue(all('wood' not in (m.body(int(i)).name or '')
                            for i in m.eq_obj1id[:2]))
        r = Robot()
        r.lower(); r.close(); r.lift(10)
        self.assertFalse(np.any(r.d.xfrc_applied))
        self.assertFalse(np.any(r.d.qfrc_applied))
        self.assertGreater(r.state()['wood'][2], .12)

    def test_valid_lifts_at_extremes(self):
        for l in [task.Layout((-3, 3), (4, -4), 6), task.Layout((3, -3), (-4, 4), 12)]:
            r = task.execute(l, task.oracle(l))
            self.assertTrue(r['success'], r)
            self.assertGreater(r['reward'], .99)

    def test_no_close_cannot_lift(self):
        r = task.execute(task.Layout((0, 0), (0, 0), 10), 'lower()\nlift(10)')
        self.assertFalse(r['success'])
        self.assertLess(r['final']['wood'][2], .04)

    def test_wrong_height_fails_even_when_grasped(self):
        r = task.execute(task.Layout((0, 0), (0, 0), 6), 'lower()\nclose()\nlift(12)')
        self.assertFalse(r['success'])
        self.assertGreater(r['final']['grip_force'], 5)

    def test_miss_cannot_receive_success_reward(self):
        r = task.execute(task.Layout((0, 0), (4, 0), 10), 'lower()\nclose()\nlift(10)')
        self.assertFalse(r['success'])
        self.assertLessEqual(r['reward'], .5)

    def test_parser_never_executes_python(self):
        for code in ['import os', 'close(x=1)', 'move(True, 0)', 'lift(99)', 'move(1+1, 0)', 'while True: close()']:
            with self.assertRaises((ValueError, SyntaxError)):
                task.parse_program(code)

    def test_splits_are_disjoint_and_reproducible(self):
        a, b = task.datasets(), task.datasets()
        self.assertEqual(a, b)
        ids = [l.key for rows in a.values() for l in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([len(a[k]) for k in ('train', 'validation', 'test')], [256, 64, 128])


if __name__ == '__main__':
    unittest.main()
