"""Frozen skill adapter. No model-specific behavior or automatic full sequence."""
import numpy as np
from experiments.torso_assembly.live import Executor as ExistingExecutor
from experiments.torso_assembly.environment import HOME
from experiments.torso_assembly.frame_skills import visual_frame_ok

def update_frame_observations(executor):
    """The verified frame becomes occluded by the feeder/sheet.

    Preserve its last RGB observation explicitly after feed starts. This is a
    stationary-fastened-frame planning prior, not a new visual measurement.
    Do not use this to fill in failed detections during pickup or placement.
    """
    return not getattr(executor, "frame_observation_locked", False)

class Executor(ExistingExecutor):
    """Expose the explicit clearance step used by the successful run83 baseline."""
    def __init__(self, env, tool_factory=None):
        if tool_factory is None:
            super().__init__(env)
        else:
            super().__init__(env, tool_factory=tool_factory)
        self.parked = False
        self.frame_observation_locked = False

    def start(self, command, robot, estimates, sheet):
        if command["skill"] == "park_grippers":
            if self.phase not in ("idle", "done") or len(self.released) != 4:
                raise ValueError("park_requires_four_released_members")
            self.queue = [("park", 0), ("park", 1)]
            self.advance(robot, estimates, sheet)
        else:
            if command["skill"] == "nail_corner" and not self.parked:
                raise ValueError("park_grippers_before_nailing")
            if command["skill"] == "feed_panel":
                if not visual_frame_ok(estimates):
                    raise ValueError("feed_requires_last_verified_frame_observation")
            super().start(command, robot, estimates, sheet)
            if command["skill"] == "feed_panel":
                self.frame_observation_locked = True

    def advance(self, robot, estimates, sheet):
        if self.queue and self.queue[0][0] == "park":
            import time
            self.kind, self.i = self.queue.pop(0)
            self.timer = 0.
            started = time.monotonic()
            self.route = np.array(self.planner.plan(self.i, robot, estimates,
                                  HOME[self.i], self.frame.rotations[self.i]))
            self.route_index = 1
            self.planning.append({"kind": "park", "wall_s": time.monotonic() - started})
            self.phase = "park"
        else:
            super().advance(robot, estimates, sheet)

    def command(self, robot, estimates, sheet):
        if self.kind != "park":
            return super().command(robot, estimates, sheet)
        self.timer += .02
        self.cmd = (robot["tcp"].copy(), robot["rotation"].copy(), np.zeros(2),
                    {self.i: self.route[self.route_index]})
        if np.max(abs(robot["joints"][self.i] - self.route[self.route_index])) < .01:
            self.route_index += 1
        if self.route_index >= len(self.route):
            if self.i == 1:
                self.parked = True
            self.advance(robot, estimates, sheet)
        if self.timer > 45:
            raise RuntimeError("park_timeout")
        return self.cmd
