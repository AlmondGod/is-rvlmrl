from pathlib import Path
import tempfile
import unittest

from analysis.assembly_prefix_evidence import verify_origin,digest,SnapshotFailure


class PrefixOriginTests(unittest.TestCase):
    def test_exact_origin_required_and_mutation_or_deletion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=('program.py','provenance.json','result.json','states.npz','prefix_receipt.json')
            for name in names:(root/name).write_bytes(name.encode())
            entry=dict(origin=directory,origin_files_sha256={name:digest(root/name) for name in names})
            verify_origin(entry)
            (root/'program.py').write_bytes(b'changed')
            with self.assertRaisesRegex(SnapshotFailure,'changed'):verify_origin(entry)
            (root/'program.py').write_bytes(b'program.py')
            (root/'states.npz').unlink()
            with self.assertRaisesRegex(SnapshotFailure,'changed'):verify_origin(entry)

    def test_incomplete_manifest_is_not_an_origin_certificate(self):
        with self.assertRaisesRegex(SnapshotFailure,'complete'):
            verify_origin(dict(origin='unused',origin_files_sha256={'program.py':'x'}))


if __name__=='__main__':unittest.main()
