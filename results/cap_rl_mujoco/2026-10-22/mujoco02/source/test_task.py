import unittest
from .task import Layout, datasets, execute, oracle, parse_program


class TaskTests(unittest.TestCase):
    def test_disjoint_feasible_splits(self):
        seen = set()
        for split in datasets().values():
            for layout in split:
                self.assertNotIn(layout.key, seen)
                seen.add(layout.key)
                result = execute(layout, oracle(layout))
                self.assertTrue(result["success"])
                self.assertEqual(result["reward"], 1.)

    def test_relative_not_absolute(self):
        layout = Layout((-2, 1), (1, -1), (-1, 2))
        result = execute(layout, "move(3, -2)\npick()\nmove(-2, 3)\nplace()")
        self.assertTrue(result["success"])
        self.assertFalse(execute(layout, "move(1, -1)\npick()")['success'])

    def test_no_reward_for_textual_success_or_escape(self):
        layout = datasets()["train"][0]
        for code in ("success = True", "__import__('os').system('id')", "move(True, 0)",
                     "while True: pass", "pick(*[])", "move(2**999999, 0)",
                     "pick()\n" * 9, "place()", "# success = True"):
            self.assertEqual(execute(layout, code)["reward"], 0.)

    def test_failure_after_success_is_failure(self):
        layout = datasets()["train"][0]
        self.assertEqual(execute(layout, oracle(layout) + "\nplace()")["reward"], 0.)

    def test_holding_is_not_success(self):
        layout = datasets()["train"][0]
        result = execute(layout, oracle(layout).rsplit("\n", 1)[0])
        self.assertFalse(result["success"])
        self.assertLess(result["reward"], 1.)

    def test_fence_contract(self):
        self.assertEqual(parse_program("```python\nmove(-1, 2)\n```"), [("move", [-1, 2])])
        with self.assertRaises(ValueError):
            parse_program("```python\npick()\n```\nThis succeeds")


if __name__ == "__main__":
    unittest.main()
