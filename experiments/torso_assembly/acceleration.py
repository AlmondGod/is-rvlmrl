"""Opt-in execution optimizations; no controller, timestep, or force-law changes.

Install only in a fresh process, before creating environments. This experimental
backend remains separate from the frozen historical simulator and its caches.
"""
import numpy as np
import mujoco
import hashlib
import inspect
from scipy.spatial.transform import Rotation

_INSTALLED = False
_PIXELS_INSTALLED = False
_CONNECTORS_INSTALLED = False
BACKENDS = ('original', 'vectorized', 'vectorized_pixels', 'vectorized_native')


def backend():
    if _INSTALLED:
        install()  # Verify that the selected methods were not replaced mid-run.
    if _CONNECTORS_INSTALLED:
        from .native_connectors import enable
        enable()
        return 'vectorized_native'
    return 'vectorized_pixels' if _PIXELS_INSTALLED else 'vectorized' if _INSTALLED else 'original'


def configure(name):
    global _PIXELS_INSTALLED, _CONNECTORS_INSTALLED
    if name not in BACKENDS:
        raise ValueError('Unknown assembly execution backend: '+str(name))
    if name in ('vectorized', 'vectorized_pixels', 'vectorized_native'):
        if _CONNECTORS_INSTALLED and name != 'vectorized_native':
            raise RuntimeError('Start a fresh process to disable native connectors')
        if _PIXELS_INSTALLED and name not in ('vectorized_pixels', 'vectorized_native'):
            raise RuntimeError('Start a fresh process to disable native pixels')
        install()
        if name in ('vectorized_pixels', 'vectorized_native'):
            from .image_ops import enable_native
            enable_native()
            _PIXELS_INSTALLED = True
        if name == 'vectorized_native':
            from .native_connectors import enable
            enable()
            _CONNECTORS_INSTALLED = True
    elif _INSTALLED:
        raise RuntimeError('Start a fresh process to switch back to the original execution backend')


def cross3(a, b):
    """Same operation order as numpy.cross for two float64 vectors of length 3."""
    return np.array((a[1]*b[2]-a[2]*b[1],
                     a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]))


def frame_velocity(self, point):
    return self.linear + cross3(self.angular, point-self.position)


def body_frame(self, body):
    from experiments.box_assembly.fastening import Frame
    m, d = self.env.m, self.env.d
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    origin = d.xpos[body]
    linear = velocity[3:] + cross3(velocity[:3], origin-d.xipos[body])
    return Frame(origin.copy(), d.xmat[body].reshape(3, 3).copy(), linear, velocity[:3].copy())


def audit_contacts(self):
    # Argmax returns the first tied maximum, matching the ordered original loop.
    # Masking NaNs matches the original `depth > peak` comparison semantics.
    d, m = self.d, self.m
    depths = -d.contact.dist
    if len(depths):
        eligible = depths > self.peak
        if eligible.any():
            k = int(np.argmax(np.where(eligible, depths, -np.inf)))
            c = d.contact[k]
            self.peak = float(depths[k])
            self.worst = {'time': float(d.time),
                          'pair': [m.geom(g).name or str(g) for g in (c.geom1, c.geom2)],
                          'depth_m': self.peak}


def step(self, targets=None, rotations=None, gripper=None):
    """Original controller/physics order, with array-based read-only audits."""
    m, d = self.m, self.d
    if targets is not None:
        for i in range(3):
            delta = np.asarray(targets[i])-self.targets[i]
            self.targets[i] += delta*min(1., .07*.02/max(np.linalg.norm(delta), 1e-9))
    if rotations is not None:
        for i in range(3):
            rv = Rotation.from_matrix(rotations[i]@self.rotations[i].T).as_rotvec()
            self.rotations[i] = Rotation.from_rotvec(rv*min(1., .3*.02/max(np.linalg.norm(rv), 1e-9))).as_matrix()@self.rotations[i]
    if gripper is not None:
        self.gripper += np.clip(np.asarray(gripper)-self.gripper, -2.5, 2.5)
    errors = []
    for i in range(3):
        q, err = self.ik(i, self.targets[i], self.rotations[i])
        self.qtarget[i] += np.clip(q-self.qtarget[i], -.7*.02, .7*.02)
        self.integral[i] = np.clip(self.integral[i]+200*.02*(self.qtarget[i]-d.qpos[self.qids[i]]), -8, 8)
        d.ctrl[self.aids[i]] = self.qtarget[i]+(d.qfrc_bias[self.vids[i]]+self.integral[i])/m.actuator_gainprm[self.aids[i], 0]
        errors.append(err)
    d.ctrl[self.grips] = self.gripper
    assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    internal = getattr(self, '_joint_internal_forces', None)
    ids = np.concatenate(self.aids)
    limits = m.actuator_forcerange[ids, 1]-.01
    for _ in range(round(.02/m.opt.timestep)):
        assert not np.any(d.xfrc_applied)
        if internal is not None:
            d.qfrc_applied[:] = 0
            internal()
        else:
            assert not np.any(d.qfrc_applied)
        mujoco.mj_step(m, d)
        audit_contacts(self)
        self.saturation += int(np.any(np.abs(d.actuator_force[ids]) >= limits))
    mujoco.mj_forward(m, d)
    audit_contacts(self)
    if internal is not None:
        d.qfrc_applied[:] = 0
    return errors


def install():
    global _INSTALLED
    from experiments.box_assembly.fastening import Frame
    from experiments.torso_assembly.fasteners import NailBank
    from experiments.arm_corner.environment import Env
    if _INSTALLED:
        if (Env.step, NailBank.frame, Frame.velocity) != (step, body_frame, frame_velocity):
            raise RuntimeError('Installed assembly execution backend was modified')
        return
    # Refuse to silently replace a newer controller or force implementation
    # with this explicitly audited copy of the historical implementation.
    expected = [(Env.step, 'cd369686ef75724f6c3048e35051ff9719e600ceffc706d8090024a47bd7f238'),
                (NailBank.frame, '32f15448461b29b4723aa2395ff8e50ae8ee9333a879c92a27c857f653b4b394'),
                (Frame.velocity, '70a1e488712fb38187e4639eb2f7367a77a759e8f27503fc12d68e05ac2b439f')]
    for function, digest in expected:
        if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != digest:
            raise RuntimeError('Execution backend needs revalidation: '+function.__qualname__)
    Frame.velocity = frame_velocity
    NailBank.frame = body_frame
    Env.step = step
    _INSTALLED = True
