"""Opt-in exact integer RGB predicates, built once outside the source tree.

A C compiler is required only for the vectorized_pixels backend. Compiler,
source and binary digests are part of physical runtime provenance. No fast-math
or machine-specific instruction flags are used.
"""
import ctypes
import fcntl
import hashlib
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


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enable():
    global _LIBRARY, _RECEIPT
    if _LIBRARY is not None: return
    compiler = shutil.which('cc')
    if compiler is None: raise RuntimeError('vectorized_pixels requires a C compiler (cc)')
    source = Path(__file__).with_name('rgb_mask.c')
    flags = ['-O3', '-std=c99', '-shared', '-fPIC']
    if platform.system() == 'Darwin':
        flags = ['-O3', '-std=c99', '-dynamiclib', '-fPIC']
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    specification = dict(source_sha256=sha(source), compiler=version,
                         flags=flags, system=platform.system(), architecture=platform.machine())
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('CAP_RL_NATIVE_CACHE', str(Path(tempfile.gettempdir())/'cap-rl-native-pixels')))
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/(key+'.so')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            pending = cache/(key+f'.{os.getpid()}.pending.so')
            try:
                try:
                    subprocess.run([compiler, *flags, str(source), '-o', str(pending)],
                                   check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError('Native pixel compilation failed: '+error.stderr.strip()) from error
                pending.replace(library)
            finally:
                pending.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    lib.wood_mask.argtypes = [array, array, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
    lib.wood_mask.restype = None
    _LIBRARY = lib
    _RECEIPT = dict(specification, binary_sha256=sha(library),
                    implementation='exact_uint8_rgb_predicates_v1')


def receipt():
    return dict(_RECEIPT) if _RECEIPT is not None else None


def mask(image, threshold, ratio):
    if _LIBRARY is None: raise RuntimeError('Native pixel backend is not enabled')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image')
    if threshold not in (35, 70) or ratio not in (1.02, 1.05):
        raise ValueError('Unverified color predicate')
    image = np.ascontiguousarray(image)
    out = np.empty(image.shape[:2], np.uint8)
    _LIBRARY.wood_mask(image, out, out.size, threshold, 102 if ratio == 1.02 else 105)
    return out
