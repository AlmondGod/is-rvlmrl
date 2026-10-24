"""Opt-in support measurement; original benchmark behavior is unchanged.

The sheet receives frame support through contact AND finite nail connections.
Capture the latter before Env.step clears its internal-force scratch buffer.
This observer never changes forces, controller commands, or acceptance limits.
"""
import math

import mujoco

from experiments.assembly_benchmark.evaluation import Evaluation
from experiments.torso_assembly.fasteners import NailBank


class ObservedNailBank(NailBank):
    def __init__(self, env, parameters=None, enabled=True):
        super().__init__(env, parameters, enabled)
        joint = env.m.body_jntadr[env.panel]
        if env.m.body_jntnum[env.panel] != 1 or env.m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Support observer requires the original free plywood body')
        self.panel_z_dof = int(env.m.jnt_dofadr[joint]) + 2
        self.panel_support_z_N = 0.

    def apply(self):
        super().apply()
        # Free-joint translation is in world coordinates. The existing hook
        # supplies only paired internal nail loads; gravity is not in this buffer.
        self.panel_support_z_N = float(self.env.d.qfrc_applied[self.panel_z_dof])


def install_observer(env):
    original = env.nails
    if original.connections or original.events or original.keys:
        raise ValueError('Install support observer before any fastener is fired')
    env.nails = ObservedNailBank(env, original.p, original.enabled)
    env._joint_internal_forces = env.nails.apply


def supported_metrics(metrics, vertical_nail_force):
    if not math.isfinite(vertical_nail_force):
        raise ValueError('Nonfinite measured nail support')
    # Signed force is essential: a downward nail preload increases the contact
    # reaction but must not be counted as additional upward support.
    return dict(metrics, forces=dict(metrics['forces'],
        frame=metrics['forces']['frame'] + vertical_nail_force))


class SupportEvaluation(Evaluation):
    def __init__(self, criteria):
        super().__init__(criteria)
        self.preflip_support = None
        self.contact_only_evaluation = Evaluation(criteria)

    def update(self, env, metrics):
        self.contact_only_evaluation.update(env, metrics)
        super().update(env, supported_metrics(metrics, env.nails.panel_support_z_N))

    def begin_flip(self, env):
        metrics = env.panel_metrics()
        self.preflip_support = dict(contact_N=metrics['forces']['frame'],
            nail_vertical_N=env.nails.panel_support_z_N,
            weight_N=metrics['panel_mass_kg'] * 9.81)
        self.contact_only_evaluation.begin_flip(env)
        super().begin_flip(env)

    def result(self, env, finished, flipped, error):
        result = super().result(env, finished, flipped, error)
        original = self.contact_only_evaluation.result(env, finished, flipped, error)
        result.update(support_measurement='frame_contact_plus_signed_internal_nail_force',
                      preflip_support_forces=self.preflip_support,
                      contact_only_acceptance={k:v for k,v in original.items() if k != 'last_flip'})
        return result
