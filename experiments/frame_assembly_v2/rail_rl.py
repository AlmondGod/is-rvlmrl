"""One-rail contact manipulation pilot. No board assistance or state edits in step.

Identical observations, rewards, motor limits and physics for residual/direct RL.
Only the additive nominal command differs. Initial-state edits occur in reset only.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .environment import Config, FrameAssembly, ROOT

OUT = ROOT / "outputs/frame_assembly/rl"
POSES = json.loads((ROOT / "outputs/frame_assembly/astra/grasp_targets.json").read_text())
DEFAULT = {"offset": [.125, .025, .067], "close": POSES["curl"], "open": POSES["open"], "quaternion": [.5, -.5, .5, -.5]}


def multiply(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result


def rotvec(q, delta):
    result = np.asarray(q, dtype=float).copy()
    mujoco.mju_quatIntegrate(result, np.asarray(delta, dtype=float), 1.)
    return result


class RailEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode="residual", params=None, randomize=True, ood=False, max_steps=800):
        super().__init__()
        self.mode, self.params = mode, params or DEFAULT
        self.randomize, self.ood, self.max_steps = randomize, ood, max_steps
        self.sim = FrameAssembly(allow_assistance=False)
        self.nominal_mass = self.sim.model.body_mass.copy()
        self.nominal_inertia = self.sim.model.body_inertia.copy()
        self.nominal_friction = self.sim.model.geom_friction.copy()
        self.corners = np.array(list(itertools.product((-1, 1), repeat=3)))
        self.action_space = gym.spaces.Box(-1., 1., (26,), dtype=np.float32)
        self.reset(seed=0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.obs().shape, dtype=np.float32)

    @property
    def dt(self):
        return .02

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        scale = float(rng.uniform(.9, 1.1)) if self.ood else 1.
        if self.ood:
            self.sim = FrameAssembly(Config(outer_x=.40*scale, outer_y=.30*scale, rail_width=.04*scale, thickness=.02*scale), allow_assistance=False)
            self.nominal_mass = self.sim.model.body_mass.copy()
            self.nominal_inertia = self.sim.model.body_inertia.copy()
            self.nominal_friction = self.sim.model.geom_friction.copy()
        self.sim.reset(seed=0, jitter=0.)
        s, m, d = self.sim, self.sim.model, self.sim.data
        self.bid = s.bids["east"]
        self.wid = m.body("r_mount").id
        self.qa = int(m.joint("east_free").qposadr[0])
        self.va = int(m.joint("east_free").dofadr[0])
        self.geom_ids = [m.geom("east_" + p).id for p in ("slab", "shoulder")]
        self.tip_ids = [m.site("r_" + n + "_tip").id for n in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
        spread = .03 if self.ood else .015
        jitter = rng.uniform(-spread, spread, 2) if self.randomize else np.zeros(2)
        yaw = float(rng.uniform(-.2 if self.ood else -.1, .2 if self.ood else .1)) if self.randomize else 0.
        self.initial = np.array([.4+jitter[0], jitter[1], .01*scale])
        d.qpos[self.qa:self.qa+3] = self.initial
        self.initial_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        d.qpos[self.qa+3:self.qa+7] = self.initial_quat
        self.rotation = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        mass_factor = float(rng.uniform(.6 if self.ood else .8, 1.4 if self.ood else 1.2)) if self.randomize else 1.
        friction = float(rng.uniform(.3 if self.ood else .4, .9 if self.ood else .8)) if self.randomize else .55
        m.body_mass[:] = self.nominal_mass
        m.body_inertia[:] = self.nominal_inertia
        m.body_mass[self.bid] *= mass_factor
        m.body_inertia[self.bid] *= mass_factor
        m.geom_friction[:] = self.nominal_friction
        m.geom_friction[self.geom_ids, 0] = friction
        # MuJoCo combines equal-priority geom sliding friction by taking the
        # maximum. Vary the hand as well, otherwise its nominal 1.0 masks wood
        # coefficients below 1.0. A factor of 1 preserves the original hand.
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in s.hand_body_ids:
                m.geom_friction[g,0]=self.nominal_friction[g,0]*(friction/.55)
        # mj_setConst temporarily resets qpos to the model reference pose.
        # Restore the randomized episode state after recomputing inertial constants.
        initial_qpos = d.qpos.copy()
        mujoco.mj_setConst(m, d)
        d.qpos[:] = initial_qpos
        mujoco.mj_forward(m, d)
        self.goal = self.initial + [-.12, 0, 0]
        self.grasp_position = self.initial + self.rotation @ np.asarray(self.params["offset"])
        self.grasp_quat = multiply(self.initial_quat, self.params["quaternion"])
        self.steps = 0
        self.hold = self.max_hold = self.place_hold = 0.
        self.lift_achieved = self.success = False
        self.max_height = self.max_penetration = 0.
        self.last_action = np.zeros(26)
        self.last_relative = None
        self.mass_factor, self.friction, self.scale = mass_factor, friction, scale
        self.contacts = self.measure()
        self.previous_potential = self.potential()
        return self.obs(), {}

    def reference(self):
        t = self.steps*self.dt
        p = self.grasp_position.copy()
        opened, closed = np.array(self.params["open"]), np.array(self.params["close"])
        if t < 2.:
            p[2] += .12; q = opened
        elif t < 3.5:
            p[2] += .12 * max(0, (3.5-t)/1.5); q = opened
        elif t < 5.:
            a = min(1., (t-3.5)/.9); q = (1-a)*opened+a*closed
        elif t < 9.:
            p[2] += .10*min(1., (t-5.)/2.); q = closed
        elif t < 11.:
            p += [-.12*min(1., (t-9.)/2.),0,.10]; q=closed
        elif t < 13.:
            p += [-.12,0,.10*max(0., (13.-t)/2.)]; q=closed
        elif t < 14.:
            p += [-.12,0,0]; a=min(1.,(t-13.)/.6); q=(1-a)*closed+a*opened
        else:
            p += [-.12,0,.15]; q=opened
        # Feedback: retain a slightly tighter closure if fingertips begin losing
        # load during lift. It changes actual motor targets, never object forces.
        if 5 <= t < 13 and self.contacts["hand_force"] < 1.5*self.sim.model.body_mass[self.bid]*9.81:
            q = np.clip(q+.06*(closed-opened), *self.sim.model.actuator_ctrlrange.T)
        return p, self.grasp_quat, q

    def base_action(self):
        p, q, fingers = self.reference()
        d, s = self.sim.data, self.sim
        inv = d.mocap_quat[s.wrist_id].copy(); inv[1:] *= -1
        relative = multiply(inv, q)
        if relative[0] < 0: relative *= -1
        angle = 2*np.arccos(np.clip(relative[0],-1,1))
        r = relative[1:]*angle/max(np.linalg.norm(relative[1:]),1e-8)
        # At half the total rate limit, a residual can oppose the nominal command.
        return np.clip(np.r_[(p-d.mocap_pos[s.wrist_id])/.005, r/.03, (fingers-d.ctrl)/.06], -.5, .5)

    def measure(self):
        s,m,d=self.sim,self.sim.model,self.sim.data
        heights=[]
        for g in self.geom_ids:
            vertices=(self.corners*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            heights.append(float(vertices[:,2].min()))
        forces=np.zeros(5); other=0.; penetration=0.
        f=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i]
            a,b=(int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2))
            if self.bid not in (a,b): continue
            penetration=max(penetration,-float(c.dist))
            mujoco.mj_contactForce(m,d,i,f)
            partner=b if a==self.bid else a
            if partner in s.hand_body_ids:
                name=m.body(partner).name
                for j,fragment in enumerate(("thumb","index_finger","middle_finger","ring_finger","pinky")):
                    if fragment in name: forces[j]+=max(0.,float(f[0]));break
            else: other+=max(0.,float(f[0]))
        R=d.xmat[self.wid].reshape(3,3)
        relative=np.r_[R.T@(d.xpos[self.bid]-d.xpos[self.wid]), (R.T@d.xmat[self.bid].reshape(3,3)).flatten()]
        slip=0. if self.last_relative is None else float(np.linalg.norm(relative[:3]-self.last_relative[:3])/self.dt)
        spin=0. if self.last_relative is None else float(np.arccos(np.clip((np.trace(self.last_relative[3:].reshape(3,3).T@relative[3:].reshape(3,3))-1)/2,-1,1))/self.dt)
        self.last_relative=relative
        return {"lowest":min(heights),"hand_force":float(forces.sum()),"forces":forces,"other_force":other,"penetration":penetration,"slip":slip,"spin":spin}

    def potential(self):
        d=self.sim.data
        center=d.xpos[self.bid]
        tips=d.site_xpos[self.tip_ids]
        distance=float(np.mean(np.linalg.norm(tips-center,axis=1)))
        height=np.clip(self.contacts["lowest"]/.05,0,1.5)
        contact=min(self.contacts["hand_force"],3.)/3.
        location=np.linalg.norm(center[:2]-self.goal[:2])
        return float(-3*distance + .6*contact + 2*height + (2*np.exp(-location/.04) if self.lift_achieved else 0))

    def obs(self):
        s,m,d=self.sim,self.sim.model,self.sim.data
        c=self.contacts
        return np.r_[d.xpos[self.wid],d.xquat[self.wid],d.mocap_pos[s.wrist_id],d.qpos[s.finger_qadr],d.qvel[m.jnt_dofadr[s.joint_ids]],d.ctrl,d.xpos[self.bid],d.xquat[self.bid],d.qvel[self.va:self.va+6],(d.site_xpos[self.tip_ids]-d.xpos[self.bid]).flatten(),np.clip(c["forces"],0,20)/5,self.goal,self.initial,self.steps/self.max_steps,self.hold/2.,float(self.lift_achieved),self.base_action(),self.last_action].astype(np.float32)

    def step(self, action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(26,) or not np.isfinite(a).all():raise ValueError("Invalid action")
        s,m,d=self.sim,self.sim.model,self.sim.data
        assert not any(d.eq_active[e] for e in s.carries.values())
        base=self.base_action()
        u=np.clip((base if self.mode in ("residual","base") else 0)+(0 if self.mode=="base" else a),-1,1)
        dp=u[:3]*.005; dp*=min(1.,.005/max(np.linalg.norm(dp),1e-9))
        d.mocap_pos[s.wrist_id]=np.clip(d.mocap_pos[s.wrist_id]+dp,[-.6,-.47,.04],[.6,.47,.75])
        dr=u[3:6]*.03;dr*=min(1.,.03/max(np.linalg.norm(dr),1e-9))
        d.mocap_quat[s.wrist_id]=rotvec(d.mocap_quat[s.wrist_id],dr)
        d.ctrl[:]=np.clip(d.ctrl+u[6:]*.06,*m.actuator_ctrlrange.T)
        mujoco.mj_step(m,d,nstep=20)
        mujoco.mj_forward(m,d)
        self.steps+=1;self.contacts=self.measure();c=self.contacts
        good=c["lowest"]>=.05 and c["hand_force"]>.05 and c["other_force"]<.01 and c["penetration"]<.001 and c["slip"]<.03 and c["spin"]<.25
        self.hold=self.hold+self.dt if good else 0.
        self.max_hold=max(self.max_hold,self.hold)
        newly_lifted=not self.lift_achieved and self.hold>=2.-1e-8
        self.lift_achieved |= newly_lifted
        self.max_height=max(self.max_height,c["lowest"])
        self.max_penetration=max(self.max_penetration,c["penetration"])
        quat=d.xquat[self.bid]; angle=2*np.arccos(np.clip(abs(np.dot(quat,self.initial_quat)),0,1))
        placed=self.lift_achieved and np.linalg.norm(d.xpos[self.bid][:2]-self.goal[:2])<.015 and abs(d.xpos[self.bid][2]-self.goal[2])<.003 and angle<np.deg2rad(10) and np.linalg.norm(d.qvel[self.va:self.va+3])<.02 and np.linalg.norm(d.qvel[self.va+3:self.va+6])<.2 and c["hand_force"]<.01
        self.place_hold=self.place_hold+self.dt if placed else 0.
        self.success=self.place_hold>=1.-1e-8
        potential=self.potential()
        reward=potential-self.previous_potential + .03*min(c["hand_force"],3)/3 + .06*float(good) + .2*float(placed) + 10*float(newly_lifted) + 20*float(self.success) - .03*max(0.,(c["penetration"]-.001)/.001) - .001*np.mean(a*a) - .002*np.mean((a-self.last_action)**2)
        self.previous_potential=potential;self.last_action=np.clip(a,-1,1)
        bad=not np.isfinite(d.qpos).all() or d.xpos[self.bid][2]<-.05 or np.linalg.norm(d.xpos[self.bid][:2])>.7 or any(w.number for w in d.warning)
        if bad:reward-=10
        info=self.summary()
        return self.obs(),float(reward),bool(self.success or bad),bool(self.steps>=self.max_steps),info

    def summary(self):
        return {"success":bool(self.success),"lift_success":bool(self.lift_achieved),"max_hold_seconds":float(self.max_hold),"max_lowest_point_m":float(self.max_height),"max_penetration_m":float(self.max_penetration),"steps":self.steps,"mass_factor":self.mass_factor,"friction":self.friction,"size_scale":self.scale,"assistance":False}


def rollout(params=None, seed=0, randomize=False, max_steps=800):
    env=RailEnv("base",params,randomize=randomize,max_steps=max_steps)
    env.reset(seed=seed)
    total=0.
    for _ in range(max_steps):
        _,r,done,truncated,info=env.step(np.zeros(26));total+=r
        if done or truncated:break
    return {**info,"return":total}
