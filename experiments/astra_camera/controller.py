"""Astra image/proprio feedback controller; PREPARED, NOT CLOUD-TESTED.

Contract: Controller(config), reset(), command(observation) -> (p3, qwxyz4, f20).
No simulator/model handle, ground-truth object state, contact data, depth,
segmentation IDs, episode seed, or rollout-file reads are accepted. The sole
object estimate comes from the paired RGB images in RailPerception. Public
robot joint/motor constants and a previously learned grasp pose are explicit
config priors. This controller does not modify physics or object state.

Required config: pose_prior={offset, quaternion, open, close},
 target_search_bounds=[[xmin,ymin,zmin],[xmax,ymax,zmax]]. The latter is a fixed
 task staging-region prior, not an episode-specific object pose. Optional:
 rail_size=[.04,.30,.02], tabletop_height=0, workspace_bounds,
 motor_kp[20], actuator_force_limits[20,2], wood_color thresholds.

Motor spring tracking error is only a proprioceptive load surrogate: it includes
inertia, gravity and friction, and is NOT a measured contact force. A small
visually monitored probe lift checks that the object follows the hand before
continuing. Camera occlusion/staleness freezes ascent, then triggers a retreat
and reacquisition. The evaluator alone determines actual task success.
"""
from collections import deque
import numpy as np
from .perception import RailPerception


ALLOWED_OBSERVATION = frozenset({
    'time', 'rgb', 'cameras', 'image_time', 'wrist_position', 'wrist_quaternion',
    'finger_q', 'finger_qvel', 'wrist_target', 'wrist_target_quaternion',
    'finger_targets', 'joint_limits',
})


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def rotation(q):
    w, x, y, z = np.asarray(q) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def limited_delta(target, current, max_norm):
    delta = np.asarray(target) - np.asarray(current)
    return current + delta * min(1., max_norm / max(np.linalg.norm(delta), 1e-12))


class Controller:
    def __init__(self, config):
        self.config = dict(config)
        prior = self.config['pose_prior']
        self.open = np.asarray(prior['open'], dtype=float).copy()
        self.close = np.asarray(prior['close'], dtype=float).copy()
        self.offset = np.asarray(prior['offset'], dtype=float).copy()
        self.prior_quat = np.asarray(prior['quaternion'], dtype=float).copy()
        if self.open.shape != (20,) or self.close.shape != (20,):
            raise ValueError('Pose prior must specify twenty finger targets')
        if self.offset.shape != (3,) or self.prior_quat.shape != (4,):
            raise ValueError('Pose prior must specify an offset and wxyz quaternion')
        if not all(np.isfinite(x).all() for x in (self.open, self.close, self.offset, self.prior_quat)):
            raise ValueError('Pose prior must be finite')
        if np.linalg.norm(self.prior_quat) < 1e-8:
            raise ValueError('Pose prior quaternion must be nonzero')
        self.prior_quat /= np.linalg.norm(self.prior_quat)
        self.workspace = np.asarray(self.config.get('workspace_bounds',
                                    [[-.6, -.47, .04], [.6, .47, .75]]), dtype=float)
        self.control_dt = float(self.config.get('control_dt', .02))
        self.kp = np.asarray(self.config.get('motor_kp', np.ones(20)), dtype=float)
        self.force_limits = np.asarray(self.config.get('actuator_force_limits',
                                       np.tile([-np.inf, np.inf], (20, 1))), dtype=float)
        if self.workspace.shape != (2, 3) or np.any(self.workspace[1] <= self.workspace[0]):
            raise ValueError('workspace_bounds must be two ordered xyz corners')
        if self.kp.shape != (20,) or self.force_limits.shape != (20, 2):
            raise ValueError('Robot motor constants have incompatible shapes')
        self.directions = (self.close - self.open).reshape(5, 4)
        self.direction_norms = np.maximum(np.linalg.norm(self.directions, axis=1), 1e-8)
        self.unit_directions = self.directions / self.direction_norms[:, None]
        self.perception = RailPerception(self.config)
        self.reset()

    def reset(self):
        self.perception.reset()
        self.state = 'acquire'
        self.state_time = None
        self.previous_time = None
        self.target = None
        self.quaternion = None
        self.anchor = None
        self.fingers = self.open.copy()
        self.alpha = np.zeros(5)
        self.grip_correction = np.zeros(5)
        self.baseline_error = np.zeros((5, 4))
        self.baseline_count = 0
        self.proprio_history = deque(maxlen=150)
        self.acquisition = deque(maxlen=5)
        self.last_image_time = -np.inf
        self.last_good_image_time = -np.inf
        self.estimate = None
        self.previous_visual = None
        self.visual_slip = 0.
        self.visual_spin = 0.
        self.clearance = None
        self.ready_seconds = 0.
        self.retries = 0
        self.probe_start_center = None
        self.probe_start_wrist = None
        self.hold_seconds = 0.
        self.status = {'state': self.state, 'tested': False}

    def _transition(self, state, now):
        self.state = state
        self.state_time = now
        self.ready_seconds = 0.

    def _wrist_at(self, image_time):
        history = list(self.proprio_history)
        before = history[0]
        for after in history[1:]:
            if after[0] >= image_time:
                u = np.clip((image_time - before[0]) / max(after[0] - before[0], 1e-8), 0., 1.)
                q1 = after[2] if np.dot(before[2], after[2]) >= 0 else -after[2]
                q = before[2] * (1-u) + q1 * u
                q /= max(np.linalg.norm(q), 1e-8)
                return before[1] * (1-u) + after[1] * u, q
            before = after
        return before[1], before[2]

    def _vision(self, obs, now):
        image_time = float(obs['image_time'])
        if not np.isfinite(image_time) or image_time > now + 1e-6:
            raise ValueError('image_time must be finite and no later than control time')
        if image_time <= self.last_image_time:
            return False
        self.last_image_time = image_time
        # Delayed RGB is not retroactively evidence that the grasp is current.
        if now - image_time > .35:
            return False
        estimate = self.perception.update(obs['rgb'], obs['cameras'], image_time)
        if estimate is None or estimate.confidence < self.config.get('minimum_visual_confidence', .30):
            return False
        wp, wq = self._wrist_at(image_time)
        R = rotation(wq)
        relative = R.T @ (estimate.center - wp)
        relative_axis = R.T @ estimate.axis
        if self.previous_visual is not None:
            old_time, old_relative, old_axis = self.previous_visual
            image_dt = max(image_time - old_time, .001)
            if image_dt <= .4:
                instantaneous_slip = np.linalg.norm(relative - old_relative) / image_dt
                instantaneous_spin = np.arccos(np.clip(abs(np.dot(relative_axis, old_axis)), 0., 1.)) / image_dt
                self.visual_slip = .6 * self.visual_slip + .4 * instantaneous_slip
                self.visual_spin = .6 * self.visual_spin + .4 * instantaneous_spin
        self.previous_visual = (image_time, relative.copy(), relative_axis.copy())
        self.estimate = estimate
        self.last_good_image_time = image_time
        self.clearance = estimate.lower_bound(self.perception.size, self.perception.margin)
        return True

    def _load_surrogate(self, obs):
        error = (np.asarray(obs['finger_targets']) - np.asarray(obs['finger_q'])).reshape(5, 4)
        corrected = error - self.baseline_error
        projected_error = np.maximum(0., np.sum(corrected * self.unit_directions, axis=1))
        torque = np.clip(self.kp * corrected.ravel(), self.force_limits[:, 0], self.force_limits[:, 1])
        projected_effort = np.maximum(0., np.sum(torque.reshape(5, 4) * self.unit_directions, axis=1))
        speed = np.linalg.norm(np.asarray(obs['finger_qvel']).reshape(5, 4), axis=1)
        return projected_error, projected_effort, speed

    def _set_anchor_from_images(self):
        centers = np.asarray([e.center for e in self.acquisition])
        axes = np.asarray([e.axis for e in self.acquisition])
        center = np.median(centers, axis=0)
        # Acquisition explicitly assumes the object is resting on the known
        # tabletop. This height prior is not used once the lift begins.
        center[2] = self.perception.table_z + self.perception.size[2] / 2.
        axis = np.median(axes, axis=0)
        yaw = np.arctan2(-axis[0], axis[1])
        qyaw = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
        self.quaternion = quaternion_multiply(qyaw, self.prior_quat)
        self.anchor = center + rotation(qyaw) @ self.offset
        self.target = self.anchor + [0., 0., .12]

    def _recover(self, now):
        if self.retries >= int(self.config.get('max_retries', 2)):
            self._transition('stopped', now)
            return
        self.retries += 1
        self.target[2] = min(self.workspace[1, 2], self.target[2] + .10)
        self._transition('retreat', now)

    def command(self, observation):
        obs = observation
        extra = set(obs) - ALLOWED_OBSERVATION
        missing = ALLOWED_OBSERVATION - set(obs)
        if extra or missing:
            raise ValueError(f'Camera controller observation contract mismatch: extra={sorted(extra)}, missing={sorted(missing)}')
        for key, shape in [('wrist_position', (3,)), ('wrist_quaternion', (4,)),
                           ('finger_q', (20,)), ('finger_qvel', (20,)),
                           ('wrist_target', (3,)), ('wrist_target_quaternion', (4,)),
                           ('finger_targets', (20,)), ('joint_limits', (20, 2))]:
            value = np.asarray(obs[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid proprioception field: {key}')
        now = float(obs['time'])
        if not np.isfinite(now):
            raise ValueError('Nonfinite control time')
        dt = self.control_dt if self.previous_time is None else now - self.previous_time
        if dt < 0:
            raise ValueError('Call reset() before starting a new episode')
        dt = float(np.clip(dt, 0., .04))
        self.previous_time = now
        wp = np.asarray(obs['wrist_position'], dtype=float)
        wq = np.asarray(obs['wrist_quaternion'], dtype=float)
        self.proprio_history.append((now, wp.copy(), wq.copy()))
        if self.target is None:
            self.target = np.asarray(obs['wrist_target'], dtype=float).copy()
            self.quaternion = np.asarray(obs['wrist_target_quaternion'], dtype=float).copy()
            self.state_time = now
        fresh = self._vision(obs, now)
        vision_live = self.estimate is not None and now - self.last_good_image_time <= .35
        age = now - self.state_time
        error, effort, speed = self._load_surrogate(obs)
        opposed_load = error[0] > .012 and np.max(error[1:]) > .012

        if self.state == 'acquire':
            self.fingers = self.open.copy()
            if fresh:
                self.acquisition.append(self.estimate)
            if len(self.acquisition) >= 4:
                centers = np.asarray([e.center for e in self.acquisition])
                axes = np.asarray([e.axis for e in self.acquisition])
                stable = np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)) < .010
                on_table = abs(float(np.median(centers[:, 2])) - self.perception.table_z) < .045
                horizontal = np.max(abs(axes[:, 2])) < .20
                if stable and on_table and horizontal:
                    self._set_anchor_from_images()
                    self._transition('approach', now)
            if age > 8.:
                self._transition('stopped', now)
        elif self.state == 'approach':
            self.fingers = self.open.copy()
            arrived = np.linalg.norm(wp - self.target) < .003 and abs(np.dot(wq, self.quaternion)) > .999
            if arrived:
                # Free-space, settled-joint offset is a load-surrogate baseline.
                if np.max(speed) < .15:
                    joint_error = (np.asarray(obs['finger_targets']) - np.asarray(obs['finger_q'])).reshape(5, 4)
                    self.baseline_error = .8*self.baseline_error + .2*joint_error
                    self.baseline_count += 1
                self.ready_seconds += dt
                if self.ready_seconds >= .20 and vision_live:
                    self._transition('descend', now)
            else:
                self.ready_seconds = 0.
        elif self.state == 'descend':
            if vision_live:
                self.target[2] = max(self.anchor[2], self.target[2] - .08 * dt)
                if abs(wp[2] - self.anchor[2]) < .0045:
                    self._transition('close', now)
        elif self.state == 'close':
            # Static joint tracking error bounds closure without claiming any
            # contact-force measurement. A timeout leads only to a small probe.
            for j in range(5):
                overloaded = error[j] > .30 and speed[j] < .15
                if not overloaded:
                    self.alpha[j] = min(1., self.alpha[j] + 1.1 * dt)
            self.fingers = self.open + np.repeat(self.alpha, 4) * (self.close-self.open)
            nearly_closed = np.min(self.alpha) > .98
            settled = np.max(speed) < .5
            self.ready_seconds = self.ready_seconds + dt if nearly_closed and settled else 0.
            if vision_live and ((self.ready_seconds > .7 and opposed_load) or
                                (self.ready_seconds > 1.1) or age > 5.):
                self.probe_start_center = self.estimate.center.copy()
                self.probe_start_wrist = wp.copy()
                self._transition('probe', now)
        elif self.state in ('probe', 'lift', 'hold'):
            # Increase closure only a small amount when image motion indicates
            # slip and the measured joints do not already show a large load.
            slipping = self.visual_slip > .025 or self.visual_spin > .35
            for j in range(5):
                correction_speed = .025 if slipping and error[j] < .15 else 0.
                if error[j] > .25:
                    correction_speed = -.05
                self.grip_correction[j] = np.clip(self.grip_correction[j] + correction_speed*dt, -.04, .10)
            self.fingers = self.open + np.repeat(self.alpha+self.grip_correction, 4) * (self.close-self.open)
            if vision_live:
                if self.state == 'probe':
                    self.target[2] = min(self.probe_start_wrist[2] + .025,
                                         self.target[2] + .025*dt)
                    wrist_rise = wp[2] - self.probe_start_wrist[2]
                    board_rise = self.estimate.center[2] - self.probe_start_center[2]
                    follows = board_rise > max(.004, .25*wrist_rise)
                    if fresh and wrist_rise > .012 and follows and self.visual_slip < .08:
                        self._transition('lift', now)
                    elif age > 2.0:
                        self._recover(now)
                else:
                    if self.clearance < .075:
                        ascent = .020 if slipping else .045
                        self.target[2] = min(.48, self.target[2] + ascent*dt)
                    if self.clearance >= .075 and self.state == 'lift':
                        self._transition('hold', now)
                    stable = self.clearance >= .060 and self.visual_slip < .025 and self.visual_spin < .25
                    self.hold_seconds = self.hold_seconds + dt if stable else 0.
                    fallen = self.clearance < .002 and self.estimate.center[2] < self.probe_start_center[2] + .004
                    if fresh and fallen and age > 1.:
                        self._recover(now)
            # With stale images, leave the wrist target at its last commanded
            # position. Never continue an unobserved upward trajectory.
        elif self.state == 'retreat':
            self.fingers = self.open.copy()
            if np.linalg.norm(wp - self.target) < .006 and age > 1.:
                self.alpha[:] = 0.
                self.grip_correction[:] = 0.
                self.acquisition.clear()
                self.perception.reset()
                self.estimate = None
                self.previous_visual = None
                self.last_good_image_time = -np.inf
                self._transition('acquire', now)
        elif self.state == 'stopped':
            # A stopped controller holds its last bounded motor commands;
            # caller decides whether to terminate/reinitialize the episode.
            pass

        if self.state not in ('acquire', 'retreat', 'stopped') and now - self.last_good_image_time > 1.2:
            self._recover(now)
        self.target = np.clip(self.target, self.workspace[0], self.workspace[1])
        limits = np.asarray(obs['joint_limits'], dtype=float)
        motor_previous = np.asarray(obs['finger_targets'], dtype=float)
        finger_step = 1.5 * dt
        fingers = np.clip(self.fingers, motor_previous-finger_step, motor_previous+finger_step)
        fingers = np.clip(fingers, limits[:, 0], limits[:, 1])
        # Interface also bounds translation, independently of the cloud harness.
        position = limited_delta(self.target, np.asarray(obs['wrist_target']), .25*dt)
        quaternion = self.quaternion / max(np.linalg.norm(self.quaternion), 1e-8)
        self.status = {
            'state': self.state, 'retries': self.retries,
            'vision_live': bool(vision_live), 'fresh_visual_estimate': bool(fresh),
            'visual_confidence': None if self.estimate is None else self.estimate.confidence,
            'visual_center': None if self.estimate is None else self.estimate.center.tolist(),
            'visual_axis': None if self.estimate is None else self.estimate.axis.tolist(),
            'visual_clearance_lower_bound': self.clearance,
            'visual_slip': float(self.visual_slip), 'visual_spin': float(self.visual_spin),
            'joint_error_load_surrogate': error.tolist(),
            'motor_effort_surrogate': effort.tolist(),
            'alpha': self.alpha.tolist(), 'grip_correction': self.grip_correction.tolist(),
            'visual_hold_seconds': float(self.hold_seconds),
            'perception': dict(self.perception.diagnostics),
            'validation_status': 'prepared_only_not_cloud_tested',
        }
        return position.copy(), quaternion.copy(), fingers.copy()
