import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .interface import validate_action, public_robot, public_events, response_ready, SYSTEM
from .provider import request_payload, extract_text
from .freeze import content_manifest, digest, verify

class BoundaryTests(unittest.TestCase):
    def test_no_privileged_fields_pass_robot_whitelist(self):
        keys = ("time", "tcp", "rotation", "joints", "joint_velocity",
                "gripper_command", "shuttle_x", "rack_slide0", "rack_slide1")
        robot = dict.fromkeys(keys, 0)
        robot.update(object_state="SECRET", reward="SECRET", contacts="SECRET",
                     seed="SECRET", nail_strength="SECRET")
        self.assertNotIn("SECRET", json.dumps(public_robot(robot)))

    def test_hidden_joint_status_is_not_a_sensor(self):
        value = public_events([{"kind": "frame", "index": 0, "triggered": True,
                                "physical_bond": True, "intact": False, "embed_m": .03}])
        self.assertEqual(value, [{"kind": "frame", "index": 0, "triggered": True}])

    def test_action_cannot_include_executable_code_or_extra_fields(self):
        good = {"skill": "position_member", "index": 0, "reason": "Visible rail."}
        self.assertEqual(validate_action(json.dumps(good)), good)
        for wrong in (dict(good, code="open('evaluator')"), dict(good, index=True),
                      dict(good, index=4), dict(good, skill="exec"),
                      dict(good, reason=""), [good], dict(good, skill="finish")):
            with self.assertRaises(ValueError):
                validate_action(json.dumps(wrong))
        with self.assertRaises(ValueError):
            validate_action("```json\n" + json.dumps(good) + "\n```")

    def test_slow_simulation_does_not_discount_inference_delay(self):
        self.assertFalse(response_ready(10., 5., 12.))
        self.assertTrue(response_ready(10., 5., 15.))

    def test_equivalent_public_content_for_transports(self):
        packet = {"robot": {"time": 4}, "phase": "idle"}
        images = {"camera0": b"same image"}
        a = request_payload("gpt-6-astra", "responses", packet, images, {}, 4096)
        q = request_payload("Qwen/Qwen3.5-4B", "chat", packet, images,
                            {"temperature": .6, "top_p": .95, "enable_thinking": True}, 4096)
        self.assertEqual(a["instructions"], q["messages"][0]["content"])
        ac, qc = a["input"][0]["content"], q["messages"][1]["content"]
        self.assertEqual(ac[0]["text"], qc[0]["text"])
        self.assertEqual(ac[2]["image_url"], qc[2]["image_url"]["url"])
        self.assertNotIn("tools", a)
        self.assertNotIn("tools", q)

    def test_truncated_response_is_not_accepted(self):
        with self.assertRaises(ValueError):
            extract_text({"choices": [{"finish_reason": "length",
                                      "message": {"content": "{}"}}]}, "chat")
        with self.assertRaises(ValueError):
            extract_text({"status": "incomplete"}, "responses")

    def test_codex_rejects_extra_tools_failures_and_overbudget_output(self):
        from .codex_provider import parse_events
        completed = {"type": "turn.completed", "usage": {"output_tokens": 20}}
        self.assertEqual(parse_events([completed], 4096)[0]["output_tokens"], 20)
        for item in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            with self.assertRaises(ValueError):
                parse_events([{"type": "item.started", "item": {"type": item}}, completed], 4096)
        with self.assertRaises(ValueError):
            parse_events([{"type": "turn.failed"}], 4096)
        with self.assertRaises(ValueError):
            parse_events([completed], 10)
        with self.assertRaises(ValueError):
            parse_events([], 4096)

    def test_feeding_preserves_observed_frame_before_sheet_occlusion(self):
        import numpy as np
        from .executor import Executor, ExistingExecutor, update_frame_observations
        from experiments.torso_assembly.environment import GOALS, YAWS
        estimates = [{"center": p.copy(), "yaw": y} for p, y in zip(GOALS, YAWS)]
        executor = Executor.__new__(Executor)
        executor.frame_observation_locked = False
        executor.panel_fed = False
        with patch.object(ExistingExecutor, "start"):
            executor.start({"skill": "feed_panel", "index": None}, {}, estimates, None)
        # Critical regression: feed has STARTED but panel_fed is still false.
        self.assertFalse(executor.panel_fed)
        self.assertFalse(update_frame_observations(executor))
        executor.frame_observation_locked = False
        estimates[0]["center"] = estimates[0]["center"] + np.array([.1, 0, 0])
        with patch.object(ExistingExecutor, "start"):
            with self.assertRaises(ValueError):
                executor.start({"skill": "feed_panel", "index": None}, {}, estimates, None)
        self.assertTrue(update_frame_observations(executor))

    def test_bundle_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "controller.py").write_text("original")
            (root / "manifest.json").write_text(json.dumps(content_manifest(root)))
            (root / "freeze.json").write_text(json.dumps(
                {"manifest_sha256": digest(root / "manifest.json")}))
            verify(root)
            (root / "controller.py").write_text("changed")
            with self.assertRaises(RuntimeError):
                verify(root)

    def test_placement_handoff_preserves_settled_target_during_model_wait(self):
        import numpy as np
        from experiments.torso_assembly.frame_skills import Skills
        from experiments.torso_assembly.environment import GOALS, YAWS
        robot = {"tcp": np.zeros((3, 3)), "rotation": np.tile(np.eye(3), (3, 1, 1)),
                 "time": 1.0}
        controller = Skills(robot)
        controller.action = {"skill": "position"}
        controller.state = "settle"
        controller.settled = 1.0
        controller.visual_integral = np.array([.003, -.002])
        controller.targets[0] = [.203, .098, .4]
        previous = controller.targets.copy()
        estimate = {"center": GOALS[0].copy(), "yaw": YAWS[0], "time": 1.0,
                    "full_multiview_silhouette": True,
                    "tcp_at_measurement": np.array([.2, .1, .4])}
        targets = controller.command(robot, [estimate] * 4)[0]
        self.assertEqual(controller.state, "done")
        np.testing.assert_array_equal(targets, previous)
        for _ in range(300):  # six seconds of supervisor waiting
            np.testing.assert_array_equal(controller.command(robot, [estimate] * 4)[0], previous)

if __name__ == "__main__":
    unittest.main()
