"""One native xArm7 and a free wooden frame rail; actuator-only manipulation.

The native gripper's linkage equalities are retained. There is no equality,
external force, mocap body, or pose write attached to the wood after reset.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / 'third_party/3-PoinTr/simulation_assets/ufactory_xarm7'
if not ASSETS.exists():
    ASSETS = Path.cwd() / 'third_party/3-PoinTr/simulation_assets/ufactory_xarm7'


def asset_hashes():
    return {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ASSETS.rglob('*')) if p.is_file()}


ORIGIN = np.array([.38, -.21])
HEIGHT = .064
DOWN = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])


def xml():
    root = ET.parse(ASSETS / 'xarm7.xml').getroot()
    root.set('model', 'frame_rail_lift')
    root.find('compiler').set('meshdir', str(ASSETS / 'assets'))
    option = root.find('option')
    option.set('timestep', '.002')
    option.set('cone', 'elliptic')
    option.set('iterations', '100')
    option.set('tolerance', '1e-9')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.6 .6 .6')
    world = root.find('worldbody')
    world.find('body').set('pos', '.80 -.28 .02')
    ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='bench', type='box', size='1.1 1 .045',
                  pos='0 0 -.045', rgba='.12 .17 .20 1', friction='.5 .001 .0001')
    body = ET.SubElement(world, 'body', name='wood', pos='.38 -.21 .032')
    ET.SubElement(body, 'freejoint', name='wood_free')
    ET.SubElement(body, 'geom', name='wood_geom', type='box', size='.019 .4572 .032',
                  density='600', rgba='.72 .49 .28 1', friction='.5 .001 .0001',
                  solref='.004 1', solimp='.99 .999 .001 .5 2')
    contact = root.find('contact')
    ET.SubElement(contact, 'exclude', body1='link_base', body2='world')
    for side in ('left', 'right'):
        for n in (1, 2):
            ET.SubElement(contact, 'pair', geom1='wood_geom', geom2=f'{side}_finger_pad_{n}',
                          condim='3', friction='.65 .65 .001 .0001 .0001',
                          solref='.004 1', solimp='.99 .999 .001 .5 2')
    return ET.tostring(root, encoding='unicode')


@lru_cache(maxsize=1)
def model():
    return mujoco.MjModel.from_xml_string(xml())


class Robot:
    def __init__(self, x=0, y=0, start=(0, 0), record=False):
        self.m = model()
        self.d = mujoco.MjData(self.m)
        self.ikd = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.jids = np.array([m.joint(f'joint{i}').id for i in range(1, 8)])
        self.qa, self.va = m.jnt_qposadr[self.jids], m.jnt_dofadr[self.jids]
        self.aids = np.array([m.actuator(f'act{i}').id for i in range(1, 8)])
        self.grip = m.actuator('gripper').id
        self.sid = m.site('link_tcp').id
        self.bid = m.body('wood').id
        self.woodqa = int(m.joint('wood_free').qposadr[0])
        d.qpos[self.woodqa:self.woodqa+2] = ORIGIN + np.array([x, y])*.01
        d.qpos[self.qa] = [0, -.4, 0, 1.1, 0, .6, 0]
        self.target = np.r_[ORIGIN+np.array(start)*.01, .20]
        mujoco.mj_forward(m, d)
        q, error = self.ik(self.target, 250, True)
        if error > .001:
            raise RuntimeError(f'initial_ik:{error}')
        d.qpos[self.qa] = q  # initial robot reset only
        mujoco.mj_forward(m, d)
        self.qtarget = q.copy()
        self.integral = np.zeros(7)
        self.closed = 0.
        self.record = record
        self.frames = []
        self.peak = 0.
        self.call_index = -1
        self.ticks = 0
        self.sample()

    def ik(self, target, iterations=12, multistart=False):
        m, d = self.m, self.ikd
        d.qpos[:] = self.d.qpos
        initial = d.qpos[self.qa].copy()
        best = initial.copy(), 1e9
        jp = np.zeros((3, m.nv)); jr = jp.copy()
        quat = np.zeros(4); rv = np.zeros(3)
        rng = np.random.default_rng(123)
        for attempt in range(12 if multistart else 1):
            d.qpos[self.qa] = initial if attempt == 0 else rng.uniform(
                [-2, -1, -2, .3, -2, -1, -2], [2, 1, 2, 2.5, 2, 2, 2])
            for _ in range(iterations):
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                matrix = DOWN @ d.site_xmat[self.sid].reshape(3, 3).T
                mujoco.mju_mat2Quat(quat, matrix.ravel())
                mujoco.mju_quat2Vel(rv, quat, 1.)
                e = np.r_[target-d.site_xpos[self.sid], .3*rv]
                error = float(np.linalg.norm(e))
                if error < best[1]:
                    best = d.qpos[self.qa].copy(), error
                if error < 1e-5:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, self.sid)
                J = np.vstack((jp[:, self.va], .3*jr[:, self.va]))
                dq = J.T @ np.linalg.solve(J @ J.T + .0001*np.eye(6), e)
                d.qpos[self.qa] = np.clip(d.qpos[self.qa]+np.clip(dq, -.15, .15),
                    m.jnt_range[self.jids, 0]+.01, m.jnt_range[self.jids, 1]-.01)
            if best[1] < 1e-5:
                break
        return best

    def sample(self):
        if self.record:
            self.frames.append({'time': float(self.d.time), 'qpos': self.d.qpos.tolist(),
                                'call_index': self.call_index, **self.state()})

    def state(self):
        d, m = self.d, self.m
        forces = np.zeros(6)
        grip_force = 0.
        for i, c in enumerate(d.contact):
            names = [m.geom(g).name or '' for g in (c.geom1, c.geom2)]
            if 'wood_geom' in names and any('finger_pad' in n for n in names):
                mujoco.mj_contactForce(m, d, i, forces)
                grip_force += max(0., forces[0])
        return {'wood': d.xpos[self.bid].tolist(), 'tcp': d.site_xpos[self.sid].tolist(),
                'wood_velocity': d.qvel[int(m.joint('wood_free').dofadr[0]):][:6].tolist(),
                'grip_force': float(grip_force), 'peak_penetration': self.peak,
                'tilt': float(np.arccos(np.clip(d.xmat[self.bid].reshape(3, 3)[2, 2], -1, 1)))}

    def tick(self, destination):
        m, d = self.m, self.d
        delta = np.asarray(destination)-self.target
        self.target += delta*min(1., .05*.02/max(np.linalg.norm(delta), 1e-9))
        q, _ = self.ik(self.target)
        self.qtarget += np.clip(q-self.qtarget, -.7*.02, .7*.02)
        self.integral = np.clip(self.integral+200*.02*(self.qtarget-d.qpos[self.qa]), -8, 8)
        d.ctrl[self.aids] = self.qtarget+(d.qfrc_bias[self.va]+self.integral)/m.actuator_gainprm[self.aids, 0]
        d.ctrl[self.grip] += np.clip(self.closed-d.ctrl[self.grip], -2.5, 2.5)
        assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        for _ in range(10):
            mujoco.mj_step(m, d)
            if d.ncon:
                self.peak = max(self.peak, float(-min(d.contact.dist)))
        mujoco.mj_forward(m, d)
        self.ticks += 1
        if self.ticks % 5 == 0:
            self.sample()
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
            raise ValueError('physics_instability')

    def move(self, destination, timeout=8.):
        for _ in range(round(timeout/.02)):
            self.tick(destination)
            if np.linalg.norm(self.d.site_xpos[self.sid]-destination) < .002:
                return
        raise ValueError('motion_timeout')

    def hold(self, seconds):
        target = self.target.copy()
        for _ in range(round(seconds/.02)):
            self.tick(target)

    def reach(self, x, y):
        self.move(np.r_[ORIGIN+np.array([x, y])*.01, .20])
        self.move(np.r_[ORIGIN+np.array([x, y])*.01, .018])
        self.hold(.3)

    def shift(self, dx, dy):
        goal = self.target.copy()
        goal[:2] += np.array([dx, dy])*.01
        if np.any(np.abs(goal[:2]-ORIGIN) > .12):
            raise ValueError('workspace_bounds')
        self.move(goal)

    def lower(self):
        goal = self.target.copy()
        goal[2] = .018
        self.move(goal)
        self.hold(.3)

    def close(self):
        self.closed = 180.
        self.hold(2.)

    def lift(self, height_cm):
        goal = self.target.copy()
        goal[2] += .01*height_cm
        if goal[2] > .35:
            raise ValueError('workspace_bounds')
        self.move(goal)
        self.hold(1.)
