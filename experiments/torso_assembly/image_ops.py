"""Exact pixel operations for the original full-resolution RGB observations."""
import cv2
import numpy as np

_NATIVE = False


def enable_native():
    global _NATIVE
    from .native_pixels import enable
    enable()
    _NATIVE = True


def wood_mask(image, threshold, ratio):
    if (_NATIVE and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
            and threshold in (35, 70) and ratio in (1.02, 1.05)):
        from .native_pixels import mask
        return mask(image, threshold, ratio)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return ((r > threshold) & (r > ratio*g) & (g > 1.18*b)).astype(np.uint8)


def mask_pixels(mask):
    """Return np.where(mask)'s row-major intp coordinates using OpenCV's scan."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        return np.where(mask)
    x = y = 0
    if _NATIVE:
        x, y, w, h = cv2.boundingRect(mask)
        if not w or not h:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        mask = mask[y:y+h, x:x+w]
    points = cv2.findNonZero(mask)
    if points is None:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    xy = points.reshape(-1, 2).astype(np.intp)
    if _NATIVE:
        xy[:, 0] += x
        xy[:, 1] += y
    return xy[:, 1], xy[:, 0]
