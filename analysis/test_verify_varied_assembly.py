"""Check that automatic held-out use cannot start from a weaker/baseline selection."""
import json
from pathlib import Path
import tempfile
import unittest
from analysis.verify_varied_assembly import eligible


class VerificationGateTests(unittest.TestCase):
    def test_requires_perfect_learned_validation_policy(self):
        metadata={'assembly_split':'varied_v2','policy_decoding':'assembly_permutations_v1'}
        selection={'selection_split':'validation','validation_success_rate':1.,'best_step':2}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def write(m,s):
                (root/'metadata.json').write_text(json.dumps(m))
                (root/'selection.json').write_text(json.dumps(s))
            write(metadata,selection)
            self.assertEqual(eligible(root),metadata)
            for key,value in [('best_step',0),('validation_success_rate',.875),('selection_split','test')]:
                write(metadata,dict(selection,**{key:value}))
                with self.subTest(key=key),self.assertRaises(ValueError):eligible(root)
            write(dict(metadata,assembly_split='pilot_v1'),selection)
            with self.assertRaises(ValueError):eligible(root)


if __name__=='__main__':unittest.main()
