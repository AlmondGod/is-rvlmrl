"""Language boundary and reward regressions for full physical assembly."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_cached_physics_is_bound_to_seed_and_rejects_tampering(self):
        from . import task_assembly as task
        def measured(seed, text, out, **profiles):
            out.mkdir(parents=True)
            value = dict(reward=.1, success=False, valid=True, seed=seed)
            (out / 'result.json').write_text(json.dumps(value))
            return value
        with tempfile.TemporaryDirectory() as root, \
             patch.dict('os.environ', CAP_RL_ASSEMBLY_CACHE=root), \
             patch.object(task, 'provenance', return_value={'test': 'fixed'}), \
             patch.object(task, 'EXPECTED_PROVENANCE', None), \
             patch.object(task, 'run', side_effect=measured) as runner:
            first = task.execute(task.Layout(41001), 'finish()')
            second = task.execute(task.Layout(41001), 'finish()')
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            self.assertEqual(runner.call_count, 1)
            task.execute(task.Layout(41002), 'finish()')
            self.assertEqual(runner.call_count, 2)
            path = Path(first['evidence']) / 'result.json'
            path.write_text(path.read_text().replace('0.1', '1.0'))
            with self.assertRaisesRegex(RuntimeError, 'evidence changed'):
                task.execute(task.Layout(41001), 'finish()')

    def test_source_mutation_cannot_change_training_task_mid_run(self):
        from . import task_assembly as task
        with patch.object(task, 'provenance', return_value={'source': 'new'}), \
             patch.object(task, 'EXPECTED_PROVENANCE', {'source': 'old'}):
            with self.assertRaisesRegex(RuntimeError, 'dependencies changed'):
                task.execute(task.Layout(41001), 'finish()')

    def test_concurrent_requests_share_one_execution(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from . import task_assembly as task
        entered, release = Event(), Event()
        def measured(seed, text, out, **profiles):
            out.mkdir(parents=True)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release measured execution')
            result = dict(reward=.1, success=False, valid=True)
            (out/'result.json').write_text(json.dumps(result))
            return result
        with tempfile.TemporaryDirectory() as root, \
             patch.dict('os.environ', CAP_RL_ASSEMBLY_CACHE=root), \
             patch.object(task, 'provenance', return_value={'test': 'concurrency'}), \
             patch.object(task, 'EXPECTED_PROVENANCE', None), \
             patch.object(task, 'run', side_effect=measured) as runner, \
             ThreadPoolExecutor(2) as pool:
            first = pool.submit(task.execute, task.Layout(41001), 'finish()')
            self.assertTrue(entered.wait(5))
            second = pool.submit(task.execute, task.Layout(41001), 'finish()')
            release.set()
            a, b = first.result(timeout=10), second.result(timeout=10)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(a['evidence_id'], b['evidence_id'])
            self.assertEqual({a['cache_hit'], b['cache_hit']}, {False, True})

    def test_interruption_and_mid_execution_mutation_are_not_certified(self):
        from . import task_assembly as task
        for interrupted in (True, False):
            manifest = {'source': 'original'}
            def measured(seed, text, out, **profiles):
                out.mkdir(parents=True)
                result = dict(reward=.1, success=False, valid=True,
                              error='interrupted' if interrupted else None)
                (out/'result.json').write_text(json.dumps(result))
                manifest['source'] = 'changed'
                return result
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as root, \
                 patch.dict('os.environ', CAP_RL_ASSEMBLY_CACHE=root), \
                 patch.object(task, 'provenance', side_effect=lambda: dict(manifest)), \
                 patch.object(task, 'EXPECTED_PROVENANCE', None), \
                 patch.object(task, 'run', side_effect=measured):
                with self.assertRaisesRegex(RuntimeError, 'Interrupted|dependencies changed'):
                    task.execute(task.Layout(41001), 'finish()')
                self.assertEqual(len(list(Path(root).glob('*/result.json'))), 1)
                self.assertEqual(list(Path(root).glob('*/certificate.json')), [])

    def test_profile_changes_cannot_reuse_another_profiles_reward(self):
        from . import task_assembly as task
        def measured(seed, text, out, **profiles):
            out.mkdir(parents=True)
            value = dict(reward=.1, success=False, valid=True, seed=seed)
            (out/'result.json').write_text(json.dumps(value))
            return value
        previous = task.SUPPORT_PROFILE, task.CONTROLLER_PROFILE
        try:
            with tempfile.TemporaryDirectory() as root, \
                 patch.dict('os.environ', CAP_RL_ASSEMBLY_CACHE=root), \
                 patch.object(task, '_POOL', None), \
                 patch.object(task, '_provenance', side_effect=lambda **profiles: profiles), \
                 patch.object(task, 'EXPECTED_PROVENANCE', None), \
                 patch.object(task, 'run', side_effect=measured) as runner:
                task.configure()
                original = task.execute(task.Layout(41001), 'finish()')
                task.configure('contact_and_nails', 'southwest_approach')
                revised = task.execute(task.Layout(41001), 'finish()')
                self.assertNotEqual(original['evidence_id'], revised['evidence_id'])
                self.assertEqual(runner.call_count, 2)
                self.assertTrue(task.execute(task.Layout(41001), 'finish()')['cache_hit'])
                with patch.object(task, '_POOL', object()), self.assertRaises(RuntimeError):
                    task.configure()
        finally:
            task.configure(*previous)

    def test_failure_reuse_requires_identical_executed_prefix(self):
        from .task_assembly import terminal_failure_prefix
        result = dict(error='ValueError:Arm already carries a member', history=[
            dict(skill='position_member', index=0, status='completed'),
            dict(skill='position_member', index=1, status='failed')])
        calls = [('position_member', 0), ('position_member', 1), ('finish', None)]
        self.assertEqual(terminal_failure_prefix(result, calls), calls[:2])
        with self.assertRaises(RuntimeError):
            terminal_failure_prefix(result, [('position_member', 2)] + calls[1:])
        for error in ('missing_finish', 'interrupted', 'simulation_deadline', None):
            self.assertIsNone(terminal_failure_prefix(dict(result, error=error), calls))

    def test_audit_rejects_changed_success_state_and_force(self):
        from .audit_assembly import compare
        value = dict(success=False, final_state=[.1], history=[{'force': 3.}], wall_s=1.)
        compare(value, dict(value, wall_s=200.))
        for change in (dict(success=True), dict(final_state=[.2]), dict(history=[{'force': 0.}])):
            with self.assertRaises(AssertionError):
                compare(value, dict(value, **change))


if __name__ == '__main__':
    unittest.main()
