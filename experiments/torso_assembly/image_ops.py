"""Exact pixel operations for the original full-resolution RGB observations."""
import cv2
import numpy as np


def mask_pixels(mask):
    """Return np.where(mask)'s row-major intp coordinates using OpenCV's scan."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        return np.where(mask)
    points = cv2.findNonZero(mask)
    if points is None:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    xy = points.reshape(-1, 2).astype(np.intp)
    return xy[:, 1], xy[:, 0]
