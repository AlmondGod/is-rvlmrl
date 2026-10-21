"""RGB-only known-rail perception. Prepared by Astra; NOT cloud-tested.

Narrow assumptions: a brown/orange wooden rail, matte non-wood surroundings,
two calibrated cameras with both rail ends visible, known rail dimensions and
an explicitly configured staging-region prior. Color masks below are computed
from RGB; they are NOT renderer segmentation/object IDs. No depth input, model,
simulator, filesystem, ground-truth pose, or episode seed is used.

The two visible rail-end centers are triangulated and then fitted with a fixed
known length. This estimates center and long axis; roll around that axis is NOT
observable reliably here. Clearance therefore uses a conservative circumscribed
cross-section bound, plus a configurable perception margin. Image-derived
confidence is a heuristic, not a certified pose-error bound.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class RailEstimate:
    center: np.ndarray
    axis: np.ndarray
    image_time: float
    confidence: float
    reprojection_px: float
    apparent_length_m: float
    masks: list
    endpoints_px: list

    def lower_bound(self, size, margin):
        width, length, thickness = np.asarray(size, dtype=float)
        # Valid for any roll about the estimated long axis, before pose error.
        return float(self.center[2] - .5 * length * abs(self.axis[2])
                     - .5 * np.hypot(width, thickness) - margin)


def project(points, camera):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    T = np.asarray(camera['world_to_camera'], dtype=float)
    K = np.asarray(camera['K'], dtype=float)
    pc = points @ T[:3, :3].T + T[:3, 3]
    uvw = pc @ K.T
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8), pc[:, 2]


def plane_points(pixels, camera, z):
    T = np.asarray(camera['world_to_camera'], dtype=float)
    K = np.asarray(camera['K'], dtype=float)
    rays_camera = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    rays_world = rays_camera @ T[:3, :3]
    origin = -T[:3, :3].T @ T[:3, 3]
    denom = rays_world[:, 2]
    distance = (z - origin[2]) / np.where(abs(denom) > 1e-8, denom, np.nan)
    return origin + distance[:, None] * rays_world


def triangulate(pixels, cameras):
    rows = []
    for uv, camera in zip(pixels, cameras):
        P = np.asarray(camera['K']) @ np.asarray(camera['world_to_camera'])[:3]
        rows.extend([uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    if abs(vt[-1, 3]) < 1e-9:
        return None
    return vt[-1, :3] / vt[-1, 3]


class RailPerception:
    def __init__(self, config):
        self.size = np.asarray(config.get('rail_size', [.04, .30, .02]), dtype=float)
        self.bounds = np.asarray(config['target_search_bounds'], dtype=float)
        if self.size.shape != (3,) or np.any(self.size <= 0):
            raise ValueError('rail_size must contain positive width, length, thickness')
        if self.bounds.shape != (2, 3) or np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError('target_search_bounds must be two ordered xyz corners')
        self.table_z = float(config.get('tabletop_height', 0.))
        self.color = dict(config.get('wood_color', {}))
        self.margin = float(config.get('clearance_margin', .008))
        self.previous = None
        self.diagnostics = {}

    def reset(self):
        self.previous = None
        self.diagnostics = {}

    def wood_mask(self, rgb):
        """Broad RGB chromaticity gate, deliberately exposed in config."""
        x = np.asarray(rgb, dtype=float) / 255.
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        maxc, minc = x.max(axis=-1), x.min(axis=-1)
        sat = (maxc - minc) / np.maximum(maxc, .001)
        c = self.color
        mask = ((r > c.get('min_r', .12)) & (sat > c.get('min_saturation', .22))
                & (r > g * c.get('min_r_over_g', 1.12))
                & (r < np.maximum(g, .001) * c.get('max_r_over_g', 2.8))
                & (g > b * c.get('min_g_over_b', 1.12)))
        # Opening removes isolated antialiasing/specular pixels; no fill across
        # a hand occlusion. Components are associated from RGB geometry below.
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))

    def _select(self, rgb, camera, previous):
        mask = self.wood_mask(rgb)
        h, w = mask.shape
        if previous is not None:
            ends = previous.center + np.array([[-.5], [.5]]) * self.size[1] * previous.axis
            uv, depth = project(ends, camera)
            if np.any(depth <= 0):
                return None
            # Wide enough for the known cross-section, frame-to-frame motion
            # and hand-split pieces; only RGB pixels inside this corridor count.
            focal = max(float(camera['K'][0][0]), float(camera['K'][1][1]))
            radius = max(12., focal * (.5 * np.hypot(self.size[0], self.size[2]) + .025)
                         / max(float(depth.min()), .05))
            corridor = np.zeros((h, w), np.uint8)
            cv2.line(corridor, tuple(np.rint(uv[0]).astype(int)),
                     tuple(np.rint(uv[1]).astype(int)), 1, int(2 * radius))
            mask &= corridor
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            keep = np.zeros_like(mask)
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] >= 10:
                    keep[labels == i] = 1
            mask = keep
        else:
            # At acquisition only, the rail rests on the known tabletop. A ray
            # intersection is a scene assumption, never a measured depth map.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            candidates = []
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] < 50:
                    continue
                yy, xx = np.where(labels == i)
                pixels = np.column_stack((xx, yy))
                stride = max(1, len(pixels) // 500)
                world = plane_points(pixels[::stride], camera,
                                     self.table_z + self.size[2])
                world = world[np.isfinite(world).all(axis=1)]
                if len(world) < 12:
                    continue
                center = np.median(world, axis=0)
                if not np.all((center[:2] >= self.bounds[0, :2]) &
                              (center[:2] <= self.bounds[1, :2])):
                    continue
                cov = np.cov(world[:, :2].T)
                eig = np.linalg.eigvalsh(cov)
                extent = np.sqrt(max(eig[-1], 0.) * 12.)
                if .45 * self.size[1] <= extent <= 1.7 * self.size[1]:
                    candidates.append((stats[i, cv2.CC_STAT_AREA], i))
            if not candidates:
                return None
            mask = (labels == max(candidates)[1]).astype(np.uint8)
        yy, xx = np.where(mask)
        if len(xx) < 60:
            return None
        pixels = np.column_stack((xx, yy)).astype(float)
        mean = pixels.mean(axis=0)
        _, _, vt = np.linalg.svd(pixels - mean, full_matrices=False)
        axis = vt[0]
        perpendicular = np.array([-axis[1], axis[0]])
        coordinate = (pixels - mean) @ axis
        lo, hi = np.quantile(coordinate, [.005, .995])
        if hi - lo < 20:
            return None
        endpoints = []
        for end in [lo, hi]:
            strip = abs(coordinate - end) <= max(2., .035 * (hi - lo))
            transverse = np.median((pixels[strip] - mean) @ perpendicular)
            endpoints.append(mean + end * axis + transverse * perpendicular)
        return np.asarray(endpoints), mask

    def update(self, rgb, cameras, image_time):
        """Return a fresh estimate or None; failure never returns truth/prediction."""
        if len(cameras) != 2 or len(rgb) != 2:
            raise ValueError('This estimator requires exactly two calibrated RGB views')
        for image, camera in zip(rgb, cameras):
            if np.asarray(image).ndim != 3 or np.asarray(image).shape[-1] != 3:
                raise ValueError('RGB images must have shape H,W,3')
            if np.asarray(camera['K']).shape != (3, 3) or np.asarray(camera['world_to_camera']).shape != (4, 4):
                raise ValueError('Camera calibration must contain K3x3 and world_to_camera4x4')
        previous = self.previous
        if previous is not None and image_time - previous.image_time > .6:
            previous = None
        selections = [self._select(im, cam, previous) for im, cam in zip(rgb, cameras)]
        if any(x is None for x in selections):
            self.diagnostics = {'reason': 'missing_or_occluded_rgb_component'}
            return None
        endpoints = [x[0] for x in selections]
        choices = []
        for reverse in [False, True]:
            paired = endpoints[1][::-1] if reverse else endpoints[1]
            a = triangulate([endpoints[0][0], paired[0]], cameras)
            b = triangulate([endpoints[0][1], paired[1]], cameras)
            if a is None or b is None:
                continue
            length = float(np.linalg.norm(b - a))
            if not .65 * self.size[1] <= length <= 1.35 * self.size[1]:
                continue
            center = (a + b) * .5
            axis = (b - a) / max(length, 1e-9)
            if previous is not None and np.dot(axis, previous.axis) < 0:
                axis = -axis
                paired = paired[::-1]
                target0 = endpoints[0][::-1]
            else:
                target0 = endpoints[0]
            if previous is None and axis[1] < 0:
                axis = -axis
                paired = paired[::-1]
                target0 = target0[::-1]
            targets = [target0, paired]
            # A fixed-length 3-D line fit reduces disparity noise while keeping
            # image residuals visible for rejection. It does not fit GT poses.
            def residual(x):
                direction = x[3:] / max(np.linalg.norm(x[3:]), 1e-9)
                p = x[:3] + np.array([[-.5], [.5]]) * self.size[1] * direction
                reprojection = []
                for cam, target in zip(cameras, targets):
                    projected, _ = project(p, cam)
                    reprojection.extend((projected - target).ravel())
                return np.r_[reprojection, (np.linalg.norm(x[3:]) - 1.) * 10.]
            fit = least_squares(residual, np.r_[center, axis], max_nfev=35,
                                loss='soft_l1', f_scale=2.)
            center = fit.x[:3]
            axis = fit.x[3:] / max(np.linalg.norm(fit.x[3:]), 1e-9)
            reprojection = float(np.sqrt(np.mean(residual(fit.x)[:8] ** 2)))
            if not np.isfinite(center).all() or reprojection > 6.:
                continue
            if not np.all((center >= self.bounds[0] - .06) &
                          (center <= self.bounds[1] + .06)):
                continue
            if any(np.any(project(center[None], cam)[1] <= .05) for cam in cameras):
                continue
            if previous is not None:
                dt = max(image_time - previous.image_time, .001)
                if np.linalg.norm(center - previous.center) > .018 + .30 * dt:
                    continue
                if abs(np.dot(axis, previous.axis)) < .85:
                    continue
            score = reprojection + 25. * abs(length - self.size[1])
            choices.append((score, center, axis, reprojection, length, targets))
        if not choices:
            self.diagnostics = {'reason': 'inconsistent_multiview_endpoints'}
            return None
        _, center, axis, error, length, targets = min(choices, key=lambda x: x[0])
        confidence = float(np.exp(-error / 3.) * np.exp(-abs(length - self.size[1]) / .055))
        estimate = RailEstimate(center, axis, float(image_time), confidence, error,
                                length, [x[1] for x in selections], targets)
        self.previous = estimate
        self.diagnostics = {'reason': 'rgb_endpoint_fit', 'confidence': confidence,
                            'reprojection_px': error, 'apparent_length_m': length}
        return estimate
