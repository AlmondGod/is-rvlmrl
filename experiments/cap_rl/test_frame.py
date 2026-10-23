"""Physical placement counterexamples: no grasp, no clearance, no release, wrong target."""
import unittest
from . import task_frame as t


class FrameTests(unittest.TestCase):
    layout = t.Layout((0, 0), (2, -1), (-3, 2), 8)

    def test_full_placement(self):
        r = t.execute(self.layout, t.oracle(self.layout))
        self.assertTrue(r['success'], r)
        self.assertTrue(r['lifted'] and r['arrived'])
        self.assertEqual(r['final']['grip_force'], 0.)
        self.assertGreater(r['reward'], .98)

    def test_no_grasp_fails(self):
        r=t.execute(self.layout,t.oracle(self.layout).replace('close()\n',''))
        self.assertFalse(r['success'])
        self.assertFalse(r['lifted'])

    def test_no_clearance_cannot_slide_to_success(self):
        r=t.execute(self.layout,t.oracle(self.layout).replace('lift(8)','lift(1)'))
        self.assertFalse(r['success'])
        self.assertFalse(r['lifted'])
        self.assertEqual(r['components']['placement'],0.)

    def test_no_release_fails(self):
        r=t.execute(self.layout,t.oracle(self.layout).replace('open()\n',''))
        self.assertFalse(r['success'])

    def test_wrong_target_fails(self):
        r=t.execute(self.layout,t.oracle(self.layout).replace('move(-5, 3)','move(-2, 3)'))
        self.assertFalse(r['success'])
        self.assertTrue(r['lifted'])
        self.assertFalse(r['arrived'])

    def test_no_retreat_fails(self):
        r=t.execute(self.layout,t.oracle(self.layout).replace('retreat()',''))
        self.assertFalse(r['success'])

    def test_parser_rejects_python(self):
        for s in ['import os','open("file")','move(True,0)','move(2-1,0)','lift(0)','move(9,0)','retreat(x=1)','open()\n'*11]:
            with self.assertRaises((ValueError,SyntaxError)):
                t.parse_program(s)

    def test_disjoint_splits(self):
        splits=t.datasets()
        self.assertEqual(splits,t.datasets())
        keys=[l.key for rows in splits.values() for l in rows]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual([len(splits[k]) for k in ('train','validation','test')],[512,64,128])


if __name__=='__main__':
    unittest.main()
