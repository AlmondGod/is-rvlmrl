"""Exact pixel-level tests; full simulator equivalence is measured separately."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.torso_assembly import image_ops, native_pixels


class NativePixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('cc')
        if compiler is None: raise unittest.SkipTest('Optional native backend requires a C compiler')
        with tempfile.TemporaryDirectory() as directory:
            probe=subprocess.run([compiler,'-x','c','-','-o',str(Path(directory)/'probe')],
                input='int main(void) { return 0; }',text=True,capture_output=True)
            if probe.returncode:
                raise unittest.SkipTest('System C toolchain cannot link even a trivial program: '+probe.stderr.strip())
        native_pixels.enable()

    def test_every_uint8_rgb_value_for_each_rule(self):
        g,b=np.meshgrid(np.arange(256,dtype=np.uint8),np.arange(256,dtype=np.uint8))
        im=np.zeros((256,256,3),np.uint8);im[:,:,1]=g;im[:,:,2]=b
        for threshold,ratio in ((70,1.02),(35,1.05),(70,1.05),(35,1.02)):
            for r in range(256):
                im[:,:,0]=r
                expected=((r>threshold)&(r>ratio*g)&(g>1.18*b)).astype(np.uint8)
                np.testing.assert_array_equal(expected,native_pixels.mask(im,threshold,ratio))

    def test_strides_and_non_uint8_fallback(self):
        im=np.random.default_rng(1).integers(0,256,(50,70,3),dtype=np.uint8)
        for view in (im,im[::-1,::2],im.astype(float)+.3):
            with patch.object(image_ops,'_NATIVE',False):expected=image_ops.wood_mask(view,70,1.02)
            with patch.object(image_ops,'_NATIVE',True):actual=image_ops.wood_mask(view,70,1.02)
            np.testing.assert_array_equal(expected,actual)

    def test_sparse_scan_preserves_every_coordinate_and_order(self):
        rng=np.random.default_rng(2)
        for shape in ((0,0),(0,5),(1,1),(30,50),(960,1280)):
            for probability in (0.,.01,.5,1.):
                mask=(rng.random(shape)<probability).astype(np.uint8)*255
                for view in (mask,mask[::-1,::2]):
                    with patch.object(image_ops,'_NATIVE',True):actual=image_ops.mask_pixels(view)
                    for expected,value in zip(np.where(view),actual):
                        self.assertEqual(expected.dtype,value.dtype)
                        np.testing.assert_array_equal(expected,value)

    def test_explicit_backend_and_build_are_recorded(self):
        code='''from experiments.cap_rl import task_assembly as t
t.configure(assembly_backend="vectorized_pixels")
import json
p=t.provenance()
print(json.dumps({"backend":p["execution_backend"],"native":p["native_pixels"]}))
try:t.configure(assembly_backend="vectorized")
except RuntimeError:pass
else:raise AssertionError("Backend was silently disabled")
'''
        result=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
        self.assertEqual(result['backend'],'vectorized_pixels')
        self.assertEqual(len(result['native']['binary_sha256']),64)
        self.assertNotIn('-ffast-math',result['native']['flags'])


if __name__=='__main__':unittest.main()
