"""Optional native connector arithmetic with runtime compatibility checks.

The finite-strength force law, state transitions and paired-force application
remain unchanged. The loader refuses a changed reference implementation or a
NumPy/runtime whose arithmetic fails the reference comparison. This is CPU code.
"""
import ctypes
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

_LIBRARY = None
_RECEIPT = None
_ORIGINAL_SHA256 = 'f0808a76699335d70cfbf43489f5d21fb8b112f5f45e51c9ecc990d3e881aa59'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(self, a, b, axis, k, damping, normal_limit, shear_limit, displacement_limit, dt):
    if _LIBRARY is None: raise RuntimeError('Native connector backend is not enabled')
    x = np.empty(56)
    for offset, frame in ((0, a), (18, b)):
        x[offset:offset+3] = frame.position
        x[offset+3:offset+12] = frame.rotation.ravel()
        x[offset+12:offset+15] = frame.linear
        x[offset+15:offset+18] = frame.angular
    x[36:39], x[39:42], x[42:45] = self.anchor_a, self.anchor_b, axis
    x[45:51] = k, damping, normal_limit, shear_limit, displacement_limit, dt
    x[51:] = self.failed, self.fracture_energy_J, self.peak_utilization, self.stiffness_work_J, self.previous_k
    out = np.empty(13)
    _LIBRARY.connectors(x, out, 1)
    self.failed = bool(out[8])
    self.fracture_energy_J, self.peak_utilization = float(out[9]), float(out[10])
    self.stiffness_work_J, self.previous_k = float(out[11]), float(out[12])
    return out[:3], out[3:6], float(out[6]), float(out[7])


def compatibility_check(library, cases=512, reference=None):
    """Compare arithmetic and state transitions against the pinned reference.

    Local random state cannot perturb scene generation. Rotations, anchor
    offsets, force limits, nonzero histories and failed/zero-stiffness cases vary.
    Float and NumPy scalar parameters are both covered.
    """
    from scipy.spatial.transform import Rotation
    from experiments.box_assembly.fastening import Frame, Connector
    reference = Connector.evaluate if reference is None else reference
    if hashlib.sha256(inspect.getsource(reference).encode()).hexdigest() != _ORIGINAL_SHA256:
        raise RuntimeError('Compatibility check requires the original reference, not an installed native replacement')
    rng = np.random.default_rng(6841)
    inputs = np.empty((cases, 56)); expected = np.empty((cases, 13))
    for i in range(cases):
        rotations = Rotation.random(2, random_state=rng).as_matrix()
        position = rng.normal(size=3)
        a = Frame(position, rotations[0], rng.normal(size=3)*.1, rng.normal(size=3))
        b = Frame(position+rng.normal(size=3)*.001, rotations[1], rng.normal(size=3)*.1, rng.normal(size=3))
        joint = Connector.at(a, b, position+rng.normal(size=3)*.02)
        b.position += rng.normal(size=3)*rng.choice([1e-4, 1e-3, 1e-2])
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        fraction = rng.uniform(.05, 1.)
        params = [rng.choice([0., 100., 12000.])*fraction, 4.*fraction,
                  45.*fraction, 70.*fraction, .004, .0005]
        params = [float(x) if i%2 else np.float64(x) for x in params]
        joint.failed = bool(rng.random()<.1)
        joint.fracture_energy_J, joint.peak_utilization, joint.stiffness_work_J = rng.random(3).tolist()
        joint.previous_k = float(rng.choice([0., params[0]]))
        state = lambda: [joint.failed, joint.fracture_energy_J, joint.peak_utilization,
                         joint.stiffness_work_J, joint.previous_k]
        inputs[i] = np.r_[a.position, a.rotation.ravel(), a.linear, a.angular,
                         b.position, b.rotation.ravel(), b.linear, b.angular,
                         joint.anchor_a, joint.anchor_b, axis, params, state()]
        force, point, energy, loss = reference(joint, a, b, axis, *params)
        expected[i] = np.r_[force, point, energy, loss, state()]
    actual = np.empty_like(expected)
    library.connectors(inputs, actual, cases)
    if not np.array_equal(actual, expected):
        raise RuntimeError('Native connector arithmetic differs from this NumPy/runtime; use vectorized_pixels or vectorized')
    return dict(cases=cases, exact=True, scope='Finite arithmetic compatibility sample; full-episode evidence is separate')


def enable():
    global _LIBRARY, _RECEIPT
    from experiments.box_assembly.fastening import Connector
    if _LIBRARY is not None:
        if Connector.evaluate is not evaluate:
            raise RuntimeError('Installed native connector method was replaced')
        return
    if hashlib.sha256(inspect.getsource(Connector.evaluate).encode()).hexdigest() != _ORIGINAL_SHA256:
        raise RuntimeError('Native connector backend needs revalidation against changed reference')
    compiler = shutil.which('cc')
    if compiler is None: raise RuntimeError('vectorized_native requires a C compiler (cc)')
    source = Path(__file__).with_name('connector_kernel.c')
    flags = ['-O3', '-std=c99', '-shared', '-fPIC', '-ffp-contract=off', '-fno-builtin-pow']
    if platform.system() == 'Darwin': flags[2] = '-dynamiclib'
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    specification = dict(source_sha256=sha(source), compiler=version, flags=flags,
                         system=platform.system(), architecture=platform.machine(), numpy=np.__version__)
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('CAP_RL_NATIVE_CONNECTOR_CACHE', str(Path(tempfile.gettempdir())/'cap-rl-native-connectors')))
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/(key+'.so')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            pending = cache/(key+f'.{os.getpid()}.pending.so')
            try:
                try:
                    subprocess.run([compiler, *flags, str(source), '-lm', '-o', str(pending)],
                                   capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError('Native connector compilation failed: '+error.stderr.strip()) from error
                pending.replace(library)
            finally:
                pending.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.connectors.argtypes = [array, array, ctypes.c_size_t]
    lib.connectors.restype = None
    checked = compatibility_check(lib)
    _LIBRARY = lib
    _RECEIPT = dict(specification, binary_sha256=sha(library), compatibility=checked,
                    reference_sha256=_ORIGINAL_SHA256, implementation='native_connector_arithmetic_v1')
    Connector.evaluate = evaluate


def receipt():
    if _LIBRARY is not None: enable()
    return dict(_RECEIPT) if _RECEIPT is not None else None
