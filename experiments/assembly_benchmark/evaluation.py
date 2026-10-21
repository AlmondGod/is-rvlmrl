"""Independent measurement gates. This module never constructs model input."""
import numpy as np
from experiments.torso_assembly.flip_evaluator import reference, measure

def assembly_ok(metrics, criteria):
    frame = metrics["frame"]
    return bool(frame["frame_geometry_ok"] and
        max(frame["position_error_m"]) < criteria["member_xy_error_m"] and
        max(frame["yaw_error_deg"]) < criteria["member_yaw_error_deg"] and
        max(frame["tilt_deg"]) < criteria["member_tilt_deg"] and
        metrics["xy_error_m"] < criteria["panel_xy_error_m"] and
        abs(metrics["height_error_m"]) < criteria["panel_z_error_m"] and
        metrics["tilt_deg"] < criteria["panel_tilt_deg"] and
        abs(np.degrees(np.arctan2(metrics["rotation_evaluator_only"][1][0],
                                  metrics["rotation_evaluator_only"][0][0]))) <
        criteria["panel_yaw_error_deg"])

class Evaluation:
    def __init__(self, criteria):
        self.c = criteria
        self.stable = 0.
        self.preflip_stable = 0.
        self.preflip = None
        self.ref = None
        self.hold = 0.
        self.position_peak = self.angle_peak = 0.
        self.unexpected = set()
        self.last_flip = None
        self.nonfinite_or_warning = False

    def begin_flip(self, env):
        self.ref = reference(env)
        self.preflip = env.panel_metrics()
        self.preflip_stable = self.stable

    def update(self, env, metrics):
        c = self.c
        f = metrics["frame"]
        self.nonfinite_or_warning |= not f["finite"] or any(f["warnings"])
        released = max(metrics["forces"]["grip"]) < .1
        support_ok = (metrics["forces"]["rack"] < .01 and
                      metrics["forces"]["other"] < .1 and
                      abs(metrics["forces"]["frame"] - metrics["panel_mass_kg"] * 9.81)
                      < .05 * metrics["panel_mass_kg"] * 9.81)
        self.stable = self.stable + .02 if assembly_ok(metrics, c) and released and support_ok else 0.
        if self.ref is None:
            return
        v = measure(env, self.ref)
        self.last_flip = v
        if v["minimum_height_above_fixture_m"] > c["fixture_clearance_m"]:
            self.position_peak = max(self.position_peak, max(v["position_errors_m"]))
            self.angle_peak = max(self.angle_peak, np.degrees(max(v["angle_errors_rad"])))
        other_support = False
        for contact in env.d.contact:
            names = [env.m.geom(g).name or "" for g in (contact.geom1, contact.geom2)]
            work = lambda n: n.startswith("wood") or n == "panel_geom"
            if contact.dist <= 0 and any(work(n) for n in names):
                if not all(work(n) for n in names) and not any("finger_pad" in n for n in names):
                    other_support = True
                if contact.dist < -.000005 and any(n.startswith(("a0_", "a1_", "a2_")) and
                                                   "finger_pad" not in n for n in names):
                    self.unexpected.add(" / ".join(names))
        good = (v["panel_up_dot"] < c["panel_up_dot_max"] and
                max(v["position_errors_m"]) < c["airborne_relative_position_m"] and
                np.degrees(max(v["angle_errors_rad"])) < c["airborne_relative_angle_deg"] and
                v["minimum_height_above_fixture_m"] > c["fixture_clearance_m"] and
                env.nails.metrics()["intact"] == c["intact_fasteners"] and not other_support)
        self.hold = self.hold + .02 if good else 0.

    def result(self, env, finished, flipped, error):
        c = self.c
        good = bool(finished and flipped and error is None and self.preflip is not None and
                    assembly_ok(self.preflip, c) and self.preflip_stable + 1e-8 >= c["preflip_released_stable_s"] and
                    self.hold + 1e-8 >= c["inverted_hold_s"] and
                    env.peak <= c["max_penetration_m"] and not self.unexpected and
                    not self.nonfinite_or_warning and
                    self.position_peak < c["airborne_relative_position_m"] and
                    self.angle_peak < c["airborne_relative_angle_deg"] and
                    env.nails.metrics()["intact"] == c["intact_fasteners"])
        return {"passes": good, "preflip_stable_s": self.preflip_stable,
                "inverted_hold_s": self.hold, "airborne_relative_position_peak_m": self.position_peak,
                "airborne_relative_angle_peak_deg": self.angle_peak,
                "unexpected_robot_contacts": sorted(self.unexpected),
                "nonfinite_or_warning": bool(self.nonfinite_or_warning),
                "peak_penetration_m": env.peak, "last_flip": self.last_flip}
