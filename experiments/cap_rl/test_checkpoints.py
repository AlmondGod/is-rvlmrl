import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from .checkpoints import save_step_adapter


class CheckpointTests(unittest.TestCase):
    def test_completed_snapshot_is_certified(self):
        class Model:
            def save_pretrained(self, path):
                self.path = path
                (path/'weights.bin').write_bytes(b'complete example weights')
        with tempfile.TemporaryDirectory() as root:
            model = Model()
            final = save_step_adapter(model, root, dict(step=3, reward=.5))
            self.assertFalse(model.path.exists())
            record = json.loads((final/'checkpoint.json').read_text())
            self.assertEqual(record['step'], 3)
            self.assertEqual(record['files_sha256']['weights.bin'],
                             hashlib.sha256((final/'weights.bin').read_bytes()).hexdigest())

    def test_interrupted_save_is_not_published_or_silently_overwritten(self):
        class Model:
            def save_pretrained(self, path):
                (path/'weights.bin').write_bytes(b'partial')
                raise RuntimeError('interrupted save')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                save_step_adapter(Model(), root, dict(step=3))
            folders = list((Path(root)/'checkpoints').iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue(folders[0].name.endswith('.pending'))
            self.assertFalse((folders[0]/'checkpoint.json').exists())
            with self.assertRaises(FileExistsError):
                save_step_adapter(Model(), root, dict(step=3))


if __name__ == '__main__':
    unittest.main()
