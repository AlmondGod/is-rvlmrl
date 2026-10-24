import json
import platform
import subprocess
import sys
import unittest
from unittest.mock import patch

from experiments.torso_assembly import native_substeps
from experiments.torso_assembly.fasteners import NailBank


class NativeSubstepGuards(unittest.TestCase):
    def test_reference_guard_binds_force_and_audit_implementations(self):
        self.assertEqual(len(native_substeps.verify_reference()),4)
        with patch.object(NailBank,'apply',lambda self:None):
            with self.assertRaisesRegex(RuntimeError,'revalidation'):
                native_substeps.verify_reference()

    def test_infrastructure_exception_bypasses_physical_failure_catch(self):
        self.assertTrue(issubclass(native_substeps.NativeBackendError,BaseException))
        self.assertFalse(issubclass(native_substeps.NativeBackendError,Exception))


@unittest.skipUnless(platform.system()=='Linux','Fused backend currently requires Linux')
class NativeSubstepRuntime(unittest.TestCase):
    def test_two_force_profiles_all_states_failures_and_joint_overrides(self):
        result=subprocess.run([sys.executable,'-m','analysis.probe_native_substeps'],
            text=True,capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(report['passes']);self.assertEqual(len(report['rows']),2)
        self.assertTrue(all(r['exact'] and r['failures']==3 and r['internal_steps']==4000 for r in report['rows']))


if __name__=='__main__':unittest.main()
