import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .audit import audit
from .task import datasets, execute, oracle, prompt


class AuditTests(unittest.TestCase):
    def fixture(self, directory):
        splits = datasets()
        metadata = {"splits": {k: [x.key for x in v] for k,v in splits.items()},
                    "source_sha256": {"task.py": hashlib.sha256(
                        Path(__file__).with_name("task.py").read_bytes()).hexdigest()}}
        report = {"paired_wins": 128, "paired_losses": 0, "delta_success_rate": 1.}
        for label, key, solve in (("baseline", "before", False), ("trained", "after", True)):
            rows = []
            for layout in splits["test"]:
                code = oracle(layout) if solve else "move(0, 0)"
                rows.append({"layout_id": layout.key, "prompt": prompt(layout),
                             "completion": code, "truncated": False, **execute(layout, code)})
            report[key] = rows
            report[label] = {"n": 128, "successes": 128 if solve else 0,
                             "success_rate": float(solve), "mean_reward": float(solve), "valid_rate": 1.}
        Path(directory, "metadata.json").write_text(json.dumps(metadata))
        Path(directory, "test_comparison.json").write_text(json.dumps(report))
        return report

    def test_replays_all_programs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            self.assertEqual(audit(directory)["replayed_programs"], 256)

    def test_rejects_fabricated_success(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.fixture(directory)
            report["after"][0]["completion"] = "move(0, 0)"
            Path(directory, "test_comparison.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                audit(directory)

    def test_rejects_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.fixture(directory)
            report["after"][0] = report["after"][1]
            Path(directory, "test_comparison.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                audit(directory)


if __name__ == "__main__":
    unittest.main()
