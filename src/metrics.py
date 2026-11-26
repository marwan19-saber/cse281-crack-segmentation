"""Evaluation metrics used everywhere in the project (MEM2).

Write once, test on hand-worked arrays, reuse in every experiment:
    MSE  = mean((a - b)^2)
    PSNR = 10 * log10(255^2 / MSE)            (infinite when MSE = 0)
    IoU  = |P AND G| / |P OR G|               (1.0 when both masks are empty)
    Dice = 2 |P AND G| / (|P| + |G|)          (1.0 when both masks are empty)
"""
import math

import numpy as np


def _check_same_shape(a, b):
    """Raise ValueError when two arrays do not have the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def mse(a, b):
    """Mean squared error between two images.

    Both inputs are cast to float64 first, because uint8 subtraction wraps around
    (10 - 12 gives 254, not -2).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    _check_same_shape(a, b)
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a, b, data_range=255.0):
    """Peak signal-to-noise ratio in dB; returns math.inf for identical images."""
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    error = mse(a, b)
    if error == 0:
        return math.inf
    return float(10.0 * np.log10(data_range ** 2 / error))


def _as_bool_pair(pred, gt):
    """Return both masks as bool arrays (any non-zero value counts as foreground)."""
    pred = np.asarray(pred) != 0
    gt = np.asarray(gt) != 0
    _check_same_shape(pred, gt)
    return pred, gt


def iou(pred, gt):
    """Intersection over union of the foreground (crack) class.

    Two empty masks score 1.0: there is nothing to find and nothing was found.
    """
    pred, gt = _as_bool_pair(pred, gt)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def dice(pred, gt):
    """Dice coefficient (F1 score) of the foreground class; 1.0 for two empty masks."""
    pred, gt = _as_bool_pair(pred, gt)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / total)


def count_abs_error(pred_count, gt_count):
    """Absolute difference between a predicted and a reference count."""
    return abs(int(pred_count) - int(gt_count))
