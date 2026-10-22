import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from . import task_mujoco as task
from .audit import audit, equivalent


class MuJoCoTaskTests(unittest.TestCase):
    def test_all_splits_feasible_and_disjoint(self):
        splits = task.datasets()
        keys = [layout.key for rows in splits.values() for layout in rows]
        self.assertEqual(len(keys), len(set(keys)))
        for rows in splits.values():
            for layout in rows:
                result = task.execute(layout, task.oracle(layout))
                self.assertTrue(result['success'], (layout, result))
                self.assertAlmostEqual(result['reward'], 1.)
                self.assertLess(result['final_speed_m_s'], 1e-12)

    def test_momentum_persists_without_braking(self):
        layout = task.Layout((0, 0), (2, 0), (0, 2))
        result = task.execute(layout, 'thrust(2, 0)')
        self.assertTrue(np.allclose(result['trace'][0]['position'], [.1, 0]))
        self.assertTrue(np.allclose(result['trace'][0]['velocity'], [.4, 0]))
        # Final passive coast travels another 20 cm; no hidden target servo.
        self.assertTrue(np.allclose(result['final_state']['position'], [.3, 0]))
        self.assertFalse(result['success'])

    def test_position_alone_does_not_dock(self):
        layout = task.Layout((0, 0), (1, 0), (0, 2))
        result = task.execute(layout, 'thrust(2, 0)\ndock()')
        self.assertEqual(result['error'], 'dock_speed')
        self.assertEqual(result['components']['docked'], 0)

    def test_docking_order_and_prefix_credit(self):
        layout = task.Layout((0, 0), (1, 0), (0, 2))
        wrong = task.execute(layout, 'thrust(0, 2)\nthrust(0, -2)\ndock()')
        self.assertEqual(wrong['error'], 'dock_position')
        prefix = task.execute(layout, 'thrust(1, 0)\nthrust(-1, 0)\ndock()\ndock()')
        self.assertFalse(prefix['success'])
        self.assertAlmostEqual(prefix['reward'], .45)

    def test_bounds_checked_during_physics(self):
        layout = task.Layout((2, 2), (0, 0), (-2, -2))
        result = task.execute(layout, 'thrust(4, 4)\nthrust(4, 4)')
        self.assertEqual(result['error'], 'out_of_bounds')
        self.assertLess(result['final_state']['time'], 1.)

    def test_grammar_rejects_arbitrary_code(self):
        for source in ['import os', '__import__("os")', 'thrust(True, 0)',
                       'thrust(1+1, 0)', 'thrust(1., 0)', 'dock(x=1)',
                       'while True: dock()', 'thrust(5, 0)', 'x=dock()',
                       'dock()\n'*11]:
            result = task.execute(task.Layout((0, 0), (1, 0), (0, 1)), source)
            self.assertFalse(result['valid'], source)
            self.assertEqual(result['reward'], 0.)

    def test_replay_is_deterministic_and_frames_are_physics(self):
        layout = task.datasets()['train'][0]
        source = task.oracle(layout)
        a = task.execute(layout, source)
        b = task.execute(layout, source, record_frames=True)
        frames = b.pop('frames')
        self.assertEqual(a, b)
        self.assertGreater(len(frames), 100)
        self.assertEqual(frames[-1], b['final_state'])
        self.assertTrue(any(np.linalg.norm(f['velocity']) > .1 for f in frames))

    def test_numeric_comparison_keeps_flags_exact(self):
        self.assertTrue(equivalent({'p': [.1, 0.], 'ok': True}, {'p': [.1+1e-10, 0.], 'ok': True}, 1e-8))
        self.assertFalse(equivalent({'p': [.1], 'ok': True}, {'p': [.1], 'ok': False}, 1e-8))
        self.assertFalse(equivalent([.1], [.2], 1e-8))

    def test_audit_rejects_tampered_physical_state(self):
        splits = task.datasets()
        metadata = {'task': 'mujoco_docking',
                    'physics_runtime': {'mujoco': task.mujoco.__version__},
                    'splits': {k: [x.key for x in rows] for k, rows in splits.items()},
                    'source_sha256': {'task_mujoco.py': hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest()}}
        report = {'paired_wins': 128, 'paired_losses': 0, 'delta_success_rate': 1.}
        for label, key, solve in [('baseline', 'before', False), ('trained', 'after', True)]:
            rows = []
            for layout in splits['test']:
                code = task.oracle(layout) if solve else 'thrust(0, 0)'
                rows.append({'layout_id': layout.key, 'prompt': task.prompt(layout),
                             'completion': code, 'truncated': False, **task.execute(layout, code)})
            report[key] = rows
            report[label] = {'n': len(rows), 'successes': 128 if solve else 0,
                             'success_rate': float(solve), 'valid_rate': 1.,
                             'mean_reward': sum(r['reward'] for r in rows)/len(rows)}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'metadata.json').write_text(json.dumps(metadata))
            (p/'test_comparison.json').write_text(json.dumps(report))
            self.assertEqual(audit(p)['replayed_programs'], 256)
            report['after'][0]['final_state']['velocity'][0] += .1
            (p/'test_comparison.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'final_state'):
                audit(p)


if __name__ == '__main__':
    unittest.main()
