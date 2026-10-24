"""Scene assets must resolve independently of the launch working directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AssetPathsTests(unittest.TestCase):
    def test_same_scenes_outside_repository(self):
        root=Path(__file__).resolve().parents[1]
        script='''import hashlib,json
from experiments.arm_corner.environment import scene as corner
from experiments.torso_assembly.environment import scene as torso
print(json.dumps({n:hashlib.sha256(f().encode()).hexdigest() for n,f in [('corner',corner),('torso',lambda:torso(layout_seed=51002))]}))
'''
        env=dict(os.environ,PYTHONPATH=str(root))
        baseline=json.loads(subprocess.check_output([sys.executable,'-c',script],cwd=root,env=env,text=True))
        with tempfile.TemporaryDirectory() as outside:
            actual=json.loads(subprocess.check_output([sys.executable,'-c',script],cwd=outside,env=env,text=True))
        self.assertEqual(actual,baseline)


if __name__=='__main__':unittest.main()
