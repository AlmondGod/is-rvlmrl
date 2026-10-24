"""Isolated native connector candidate; never imported by the training backend."""
import ctypes,hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/opt/cap-speed')
FLAGS=['-O3','-std=c99','-shared','-fPIC','-ffp-contract=off','-fno-builtin-pow']
def install():
 from experiments.box_assembly.fastening import Connector
 source=ROOT/'connector_probe02.c';library=ROOT/'connector_candidate.so'
 subprocess.run(['cc',*FLAGS,str(source),'-lm','-o',str(library)],check=True)
 lib=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 lib.connectors.argtypes=[array,array,ctypes.c_size_t];lib.connectors.restype=None
 original=Connector.evaluate
 def evaluate(self,a,b,axis,k,damping,normal_limit,shear_limit,displacement_limit,dt):
  x=np.empty(56)
  for offset,frame in ((0,a),(18,b)):
   x[offset:offset+3]=frame.position;x[offset+3:offset+12]=frame.rotation.ravel();x[offset+12:offset+15]=frame.linear;x[offset+15:offset+18]=frame.angular
  x[36:39]=self.anchor_a;x[39:42]=self.anchor_b;x[42:45]=axis
  x[45:51]=k,damping,normal_limit,shear_limit,displacement_limit,dt
  x[51:]=self.failed,self.fracture_energy_J,self.peak_utilization,self.stiffness_work_J,self.previous_k
  out=np.empty(13);lib.connectors(x,out,1)
  self.failed=bool(out[8]);self.fracture_energy_J=float(out[9]);self.peak_utilization=float(out[10]);self.stiffness_work_J=float(out[11]);self.previous_k=float(out[12])
  return out[:3],out[3:6],float(out[6]),float(out[7])
 Connector.evaluate=evaluate
 receipt=dict(scope='Isolated prototype only, not the certified training backend',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),flags=FLAGS,numpy=np.__version__,compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0])
 return original,evaluate,receipt
