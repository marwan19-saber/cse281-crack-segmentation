"""Point transforms, histograms, histogram equalization, smoothing and sharpening (MEM3).

All functions take and return uint8 gray images (H, W) unless stated otherwise.
Neighbourhood operations use reflect-101 borders (np.pad "reflect" = cv2.BORDER_REFLECT_101)
and correlation, like cv2.filter2D. For symmetric kernels correlation equals convolution.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

LAPLACIAN_KERNELS = {
    "4n": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64),
    "8n": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float64),
}
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)


def _to_uint8(x):
    """Round to the nearest integer and clip to 0-255."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- histogram, point transforms
def histogram(img, bins=256):
    """Count of each gray level 0..bins-1."""
    return np.bincount(np.asarray(img).ravel(), minlength=bins)


def negative(img):
    """s = 255 - r."""
    return (255 - img.astype(np.int16)).astype(np.uint8)


def log_transform(img, c=None):
    """s = c * ln(1 + r); default c = 255 / ln(256) maps 255 to 255."""
    c = 255.0 / np.log(256.0) if c is None else c
    return _to_uint8(c * np.log1p(img.astype(np.float64)))


def gamma_transform(img, gamma, c=1.0):
    """Power law s = 255 * c * (r / 255) ** gamma; gamma < 1 brightens, > 1 darkens."""
    return _to_uint8(255.0 * c * (img.astype(np.float64) / 255.0) ** gamma)


def contrast_stretch(img, low_pct=2, high_pct=98):
    """Linearly map the [low_pct, high_pct] percentile range to [0, 255]."""
    lo, hi = np.percentile(img, (low_pct, high_pct))
    if hi <= lo:
        return img.copy()
    return _to_uint8((img.astype(np.float64) - lo) * 255.0 / (hi - lo))


def hist_equalize(img, variant="gw", levels=256):
    """Histogram equalization by mapping each level through the cumulative histogram.

    variant "gw": s_k = round((L - 1) * cdf_k)  (Gonzalez & Woods)
    variant "cv": OpenCV's form, round((cum_k - cum_min) * 255 / (N - cum_min)),
                  which sends the darkest occupied level to 0 (matches cv2.equalizeHist).
    """
    counts = histogram(img, levels).astype(np.int64)
    cum = np.cumsum(counts)
    if variant == "gw":
        lut = np.rint((levels - 1) * cum / cum[-1])
    elif variant == "cv":
        first = int(np.flatnonzero(counts)[0])
        if counts[first] == cum[-1]:
            return np.full_like(img, first)
        scale = np.float32(levels - 1) / np.float32(cum[-1] - counts[first])
        lut = np.rint((cum - counts[first]).astype(np.float32) * scale)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return np.clip(lut, 0, levels - 1).astype(np.uint8)[img]


# ---------------------------------------------------------------- convolution and smoothing
def convolve2d(img, kernel):
    """Correlate img with an odd-sized kernel, reflect-101 borders; returns float64 (H, W)."""
    kernel = np.asarray(kernel, np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sides must be odd")
    padded = np.pad(img.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
                    mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def mean_kernel(k):
    """k x k box kernel that sums to 1."""
    return np.full((k, k), 1.0 / (k * k))


SMALL_GAUSSIAN = {  # OpenCV's fixed (binomial) 1-D kernels used when sigma <= 0 and k <= 7
    1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}


def gaussian_kernel(k, sigma=0):
    """k x k Gaussian kernel that sums to 1.

    sigma <= 0 follows OpenCV: binomial table for k <= 7, else sigma = 0.3((k-1)/2 - 1) + 0.8.
    """
    if sigma <= 0 and k in SMALL_GAUSSIAN:
        g = np.array(SMALL_GAUSSIAN[k])
    else:
        if sigma <= 0:
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        x = np.arange(k) - (k - 1) / 2
        g = np.exp(-x ** 2 / (2 * sigma ** 2))
        g /= g.sum()
    return np.outer(g, g)


def mean_filter(img, k=3):
    """Box (arithmetic mean) smoothing."""
    return _to_uint8(convolve2d(img, mean_kernel(k)))


def gaussian_filter(img, k=5, sigma=0):
    """Gaussian smoothing."""
    return _to_uint8(convolve2d(img, gaussian_kernel(k, sigma)))

# غقف
# ---------------------------------------------------------------- sharpening
def laplacian_sharpen(img, kernel="8n"):
    """g = f - laplacian(f), using a centre-negative Laplacian kernel."""
    f = img.astype(np.float64)
    return _to_uint8(f - convolve2d(f, LAPLACIAN_KERNELS[kernel]))


def gradient_sharpen(img, mask_k=5, kernel="8n"):
    """Gonzalez & Woods combined method: Laplacian detail weighted by a smoothed gradient mask.

    M = Sobel magnitude, smoothed with a mask_k x mask_k mean and scaled to [0, 1];
    g = f + M * (laplacian_sharpen(f) - f). Flat noisy areas (small M) are not amplified.
    """
    f = img.astype(np.float64)
    mag = np.hypot(convolve2d(f, SOBEL_X), convolve2d(f, SOBEL_X.T))
    mask = convolve2d(mag, mean_kernel(mask_k))
    peak = mask.max()
    mask = mask / peak if peak > 0 else mask
    return _to_uint8(f - mask * convolve2d(f, LAPLACIAN_KERNELS[kernel]))
