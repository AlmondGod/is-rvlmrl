"""Native arithmetic, irreversible state and provenance checks; physics separately."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments.box_assembly.fastening import Connector, Frame
from experiments.torso_assembly import native_connectors


class NativeConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('cc')
        if compiler is None: raise unittest.SkipTest('Optional native backend requires a C compiler')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([compiler, '-x', 'c', '-', '-o', str(Path(directory)/'probe')],
                input='int main(void) { return 0; }', text=True, capture_output=True)
            if result.returncode:
                raise unittest.SkipTest('System C toolchain cannot link a trivial program: '+result.stderr.strip())
        cls.reference = staticmethod(Connector.evaluate)
        native_connectors.enable()

    @classmethod
    def tearDownClass(cls):
        Connector.evaluate = cls.reference
        native_connectors._LIBRARY = None
        native_connectors._RECEIPT = None

    def test_twenty_thousand_reference_cases(self):
        result = native_connectors.compatibility_check(native_connectors._LIBRARY, 20000, self.reference)
        self.assertTrue(result['exact'])

    def test_irreversible_failure_and_work_history(self):
        a = Frame(np.zeros(3), np.eye(3))
        b = Frame(np.zeros(3), np.eye(3))
        original = Connector.at(a, b, np.zeros(3)); native = copy.deepcopy(original)
        for displacement, stiffness in [(0., 0.), (.0001, 100.), (.0002, 12000.),
                                        (.0003, 6000.), (.01, 12000.), (0., 12000.), (0., 0.)]:
            b.position[:] = [displacement, -.3*displacement, 0.]
            args = (a, b, np.array([1., 0., 0.]), stiffness, 4., 45., 70., .004, .0005)
            expected = self.reference(original, *args)
            actual = native_connectors.evaluate(native, *args)
            for x, y in zip(expected, actual): np.testing.assert_array_equal(x, y)
            for field in ('failed', 'fracture_energy_J', 'peak_utilization', 'stiffness_work_J', 'previous_k'):
                self.assertEqual(getattr(original, field), getattr(native, field))
        self.assertTrue(native.failed)
        self.assertGreater(native.fracture_energy_J, 0.)

    def test_cannot_compare_replacement_against_itself(self):
        with self.assertRaisesRegex(RuntimeError, 'original reference'):
            native_connectors.compatibility_check(native_connectors._LIBRARY)

    def test_changed_runtime_method_is_rejected(self):
        with patch.object(Connector, 'evaluate', self.reference):
            with self.assertRaisesRegex(RuntimeError, 'was replaced'):
                native_connectors.receipt()

    def test_backend_receipts_and_no_silent_disable(self):
        code = '''from experiments.cap_rl import task_assembly as task
task.configure(assembly_backend='vectorized_native')
import json
p=task.provenance()
assert 'native_pixels' in p and 'native_connectors' in p
print(json.dumps({'backend':p['execution_backend'],'native':p['native_connectors']}))
for backend in ('original','vectorized','vectorized_pixels'):
 try:task.configure(assembly_backend=backend)
 except RuntimeError:pass
 else:raise AssertionError('Native backend silently disabled')
'''
        result = json.loads(subprocess.check_output([sys.executable, '-c', code], text=True))
        self.assertEqual(result['backend'], 'vectorized_native')
        self.assertTrue(result['native']['compatibility']['exact'])
        self.assertEqual(len(result['native']['binary_sha256']), 64)
        self.assertIn('-ffp-contract=off', result['native']['flags'])
        self.assertNotIn('-ffast-math', result['native']['flags'])


if __name__ == '__main__': unittest.main()
