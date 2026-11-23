"""Noise-removal (restoration) filters: mean-type and order-statistic filters (MEM4).

All filters take a uint8 gray image and an odd window size k, use reflect-101 borders
(np.pad "reflect" = cv2.BORDER_REFLECT_101 = scipy.ndimage mode "mirror") and return uint8.
Noise models themselves live in src.data.add_noise (single source of degradation).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(img, k):
    """All k x k neighbourhoods as a float64 array of shape (H, W, k*k)."""
    if k % 2 == 0:
        raise ValueError("window size k must be odd")
    padded = np.pad(img.astype(np.float64), k // 2, mode="reflect")
    return sliding_window_view(padded, (k, k)).reshape(*img.shape, k * k)


def _to_uint8(x):
    """Round to the nearest integer and clip to 0-255."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- mean-type filters
def arithmetic_mean(img, k=3):
    """Average of the window: reduces Gaussian noise, blurs edges."""
    return _to_uint8(_windows(img, k).mean(axis=-1))


def geometric_mean(img, k=3):
    """(prod w)^(1/k^2); computed in log space on w + 1 so zero pixels are allowed."""
    return _to_uint8(np.exp(np.log(_windows(img, k) + 1).mean(axis=-1)) - 1)


def harmonic_mean(img, k=3):
    """k^2 / sum(1 / w); good for salt noise. Shifted by +1 so zero pixels are allowed."""
    w = _windows(img, k) + 1
    return _to_uint8(w.shape[-1] / (1.0 / w).sum(axis=-1) - 1)


def contraharmonic_mean(img, k=3, q=1.5):
    """sum(w^(q+1)) / sum(w^q): q > 0 removes pepper, q < 0 removes salt, q = 0 is the mean.

    Shifted by +1 so zero pixels do not divide by zero when q < 0.
    """
    w = _windows(img, k) + 1
    return _to_uint8((w ** (q + 1)).sum(axis=-1) / (w ** q).sum(axis=-1) - 1)


# ---------------------------------------------------------------- order-statistic filters
def median_filter(img, k=3):
    """Median of the window: removes salt-and-pepper impulses (outliers do not move it)."""
    return _to_uint8(np.median(_windows(img, k), axis=-1))


def max_filter(img, k=3):
    """Maximum of the window: removes pepper (dark) noise, grows bright regions."""
    return _to_uint8(_windows(img, k).max(axis=-1))


def min_filter(img, k=3):
    """Minimum of the window: removes salt (bright) noise, grows dark regions."""
    return _to_uint8(_windows(img, k).min(axis=-1))


def midpoint_filter(img, k=3):
    """(max + min) / 2: suits uniform or Gaussian noise without impulses."""
    w = _windows(img, k)
    return _to_uint8((w.max(axis=-1) + w.min(axis=-1)) / 2)


def alpha_trimmed_mean(img, k=3, d=2):
    """Average after dropping the d/2 lowest and d/2 highest values.

    d = 0 gives the arithmetic mean; d = k*k - 1 gives the median.
    """
    if d % 2 or not 0 <= d < k * k:
        raise ValueError("d must be even and smaller than k*k")
    w = np.sort(_windows(img, k), axis=-1)
    return _to_uint8(w[..., d // 2: k * k - d // 2].mean(axis=-1))
