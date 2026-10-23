"""Language boundary and reward regressions for full physical assembly."""
import ast
import unittest
from pathlib import Path
from .assembly import parse_program, score
from .task_assembly import datasets, prompt


class AssemblyTests(unittest.TestCase):
    def test_unsafe_python_rejected(self):
        for source in ('import os', '__import__("os")', 'position_member(True)',
                       'position_member(-1)', 'position_member(4)', 'position_member(i)',
                       'position_member(1+1)', 'robot.position_member(0)',
                       'position_member(index=0)', 'finish(0)',
                       'for i in range(4):\n position_member(i)',
                       'finish()\nposition_member(0)', 'stop()\nfinish()', '',
                       'park_grippers()\n' * 33):
            with self.subTest(source=source), self.assertRaises((ValueError, SyntaxError)):
                parse_program(source)

    def test_full_calls_preserved_without_repair(self):
        code = 'position_member(3)\nrelease_member(3)\npark_grippers()\nfinish()'
        self.assertEqual(parse_program(code), [('position_member', 3), ('release_member', 3),
                                               ('park_grippers', None), ('finish', None)])
        self.assertEqual(parse_program('```python\n' + code + '\n```'), parse_program(code))

    def test_safety_gate_cannot_be_outweighed_by_progress(self):
        self.assertEqual(score(range(4), 8, 30, 30, True, True), 0.)
        self.assertAlmostEqual(score(range(4), 8, 3, 5, True, False), 1.)
        self.assertAlmostEqual(score(range(4), 8, 3, 5, False, False), .8)
        self.assertEqual(score([], 0, 0, 0, False, False), 0.)
        self.assertEqual(score([0, 0], 0, 0, 0, False, False), .075)

    def test_no_seed_or_hidden_pose_in_model_prompt(self):
        data = datasets()
        self.assertEqual(prompt(data['train'][0]), prompt(data['test'][-1]))
        for rows in data.values():
            for layout in rows:
                self.assertNotIn(str(layout.seed), prompt(layout))

    def test_splits_disjoint_from_historical_seeds(self):
        data = datasets()
        seeds = [l.seed for rows in data.values() for l in rows]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertFalse(set(seeds) & {3, 22001, 22002, 22003, 23001, 23002, 23003})

    def test_adapter_has_no_workpiece_pose_writes_or_direct_code_execution(self):
        tree = ast.parse(Path(__file__).with_name('assembly.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assertNotIn('.qpos', ast.unparse(target))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ('exec', 'eval'))


if __name__ == '__main__':
    unittest.main()
