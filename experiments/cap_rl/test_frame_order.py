import unittest
from . import task_frame as t
from .frame_order import ordered_carry


class OrderedCarryTests(unittest.TestCase):
    def test_physical_slide_then_lift_is_rejected(self):
        l=t.datasets()['train'][0]
        calls=t.oracle(l).splitlines(); calls[3],calls[4]=calls[4],calls[3]
        r=t.execute(l,'\n'.join(calls))
        self.assertTrue(r['success'])  # Regression: the original reward loophole.
        self.assertFalse(ordered_carry(l,r['trace'])['passed'])

    def test_real_pick_carry_place_passes(self):
        l=t.datasets()['train'][0]
        r=t.execute(l,t.oracle(l))
        check=ordered_carry(l,r['trace'])
        self.assertTrue(r['success'] and check['passed'])
        self.assertLess(check['pickup_call_index'],check['arrival_call_index'])

    def test_arrival_at_same_boundary_is_insufficient(self):
        l=t.datasets()['train'][0]
        r=t.execute(l,t.oracle(l))
        self.assertFalse(ordered_carry(l,[r['trace'][4]])['passed'])


if __name__=='__main__':
    unittest.main()
