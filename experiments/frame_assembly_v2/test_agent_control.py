import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .agent_control import apply, load, snapshot, validate
from .environment import FrameAssembly


class BridgeTests(unittest.TestCase):
    def test_rejects_object_control(self):
        for field in ("qpos", "board_poses", "eq_active", "force", "assist"):
            with self.assertRaises(ValueError):
                validate({field: [0, 0, 0]})

    def test_rejects_malformed_actions(self):
        for action in ({"seconds": 100}, {"seconds": float("nan")}, {"wrist_position": [0, 0, 100]}, {"finger_angles": [0] * 19}, {"wrist_quaternion": [0] * 4}):
            with self.assertRaises(ValueError):
                validate(action)

    def test_checkpoint_restores_deterministic_control(self):
        env = FrameAssembly()
        a = validate({"seconds": .3, "wrist_position": [.35, .28, .4], "finger_angles": [.1] * 20})
        apply(env, a)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / "session.json").write_text(json.dumps({"scene_sha256": hashlib.sha256(env.xml.encode()).hexdigest()}))
            snapshot(env, p)
            restored, _ = load(p)
            b = validate({"seconds": .4, "wrist_position": [.32, .27, .39]})
            apply(env, b)
            apply(restored, b)
            np.testing.assert_allclose(env.data.qpos, restored.data.qpos, atol=1e-10)
            self.assertFalse(any(restored.data.eq_active[e] for e in restored.carries.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
