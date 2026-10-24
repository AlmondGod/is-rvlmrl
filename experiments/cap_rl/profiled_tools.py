"""Explicit development controller profile; original Tool remains the default."""
import numpy as np
from experiments.torso_assembly.tool_skills import Tool


class SouthwestApproachTool(Tool):
    @staticmethod
    def frame_direction(index, sign, axis):
        # Fixed from training-layout development, never selected by seed,
        # model identity, evaluator state, or a failed runtime shot.
        horizontal = .9 if index == 0 else .8
        return sign * horizontal * axis + np.array([0, 0, -np.sqrt(1-horizontal**2)])
