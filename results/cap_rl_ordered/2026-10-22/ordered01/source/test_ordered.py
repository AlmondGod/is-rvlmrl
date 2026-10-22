import unittest
from .task_ordered import Layout, datasets, execute, oracle, parse_program


class OrderedTests(unittest.TestCase):
    def setUp(self):
        self.layout = Layout((0, 0), (-2, 0), (2, 1), (0, -2), (-1, 2), 'red')

    def test_disjoint_feasible_both_orders(self):
        seen = set()
        for layouts in datasets().values():
            self.assertEqual({x.first for x in layouts}, {'red', 'blue'})
            for layout in layouts:
                self.assertNotIn(layout.key, seen)
                seen.add(layout.key)
                result = execute(layout, oracle(layout))
                self.assertTrue(result['success'])
                self.assertAlmostEqual(result['reward'], 1.)

    def test_one_delivery_is_only_partial(self):
        result = execute(self.layout, '\n'.join(oracle(self.layout).splitlines()[:4]))
        self.assertFalse(result['success'])
        self.assertAlmostEqual(result['reward'], .45)

    def test_order_is_enforced(self):
        result = execute(self.layout, 'move(2, 1)\npick()')
        self.assertEqual(result['error'], 'wrong_order')
        self.assertEqual(result['reward'], 0.)

    def test_state_carries_across_deliveries(self):
        # Fifth call must use first target as the origin, not the starting cell.
        code = oracle(self.layout).splitlines()
        self.assertEqual(code[4], 'move(2, 3)')
        code[4] = 'move(2, 1)'
        self.assertFalse(execute(self.layout, '\n'.join(code))['success'])

    def test_runtime_errors_override_partial_reward(self):
        prefix = '\n'.join(oracle(self.layout).splitlines()[:4])
        for tail, error in [('\nplace()', 'empty_place'), ('\npick()', 'already_delivered'),
                            ('\nmove(0, -1)', 'out_of_bounds')]:
            result = execute(self.layout, prefix+tail)
            self.assertEqual(result['reward'], 0.)
            self.assertEqual(result['error'], error)
        result = execute(self.layout, 'move(-2, 0)\npick()\nmove(4, 1)\nplace()')
        self.assertEqual(result['error'], 'occupied_place')

    def test_grammar_and_budget(self):
        for code in ['move(True, 0)', 'move(1+1, 0)', 'import os', 'move(0,0)\n'*13,
                     "__import__('os').system('id')", 'x=1', '# success', 'pick(*[])']:
            self.assertEqual(execute(self.layout, code)['reward'], 0.)
        self.assertEqual(len(parse_program(oracle(self.layout))), 8)
        self.assertTrue(execute(self.layout, '```python\n'+oracle(self.layout)+'\n```')['success'])


if __name__ == '__main__':
    unittest.main()
