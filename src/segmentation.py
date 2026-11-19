"""Edge, point and line detection, thresholding, edge linking and Hough transform (MEM4).

Cracks are DARK on bright concrete, so thresholds return True for pixels at or below T and
line detectors respond positively to dark lines by default. Neighbourhood operations reuse
MEM2's convolve2d (reflect-101 borders, correlation).
"""
import numpy as np

from src.enhancement import convolve2d, gaussian_kernel

ROBERTS_X = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], np.float64)   # 2x2 kernel,
ROBERTS_Y = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)   # anchored at centre
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
LAPLACIAN = {
    "4n": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64),
    "8n": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float64),
}
LINE_MASKS = {  # Gonzalez & Woods line masks; respond positively to BRIGHT 1-pixel lines
    0: np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float64),
    45: np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], np.float64),
    90: np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], np.float64),
    -45: np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], np.float64),
}


# ---------------------------------------------------------------- gradient operators
def _gradient(img, kx, ky):
    """Return (gx, gy, magnitude) as float64 for a pair of derivative kernels."""
    f = img.astype(np.float64)
    gx, gy = convolve2d(f, kx), convolve2d(f, ky)
    return gx, gy, np.hypot(gx, gy)


def roberts(img):
    """Roberts cross-gradient (2x2 diagonal differences)."""
    return _gradient(img, ROBERTS_X, ROBERTS_Y)


def prewitt(img):
    """Prewitt gradient (equal weights)."""
    return _gradient(img, PREWITT_X, PREWITT_X.T)


def sobel(img):
    """Sobel gradient (centre row/column weighted by 2: slight smoothing)."""
    return _gradient(img, SOBEL_X, SOBEL_X.T)


# ---------------------------------------------------------------- second derivatives
def laplacian(img, kernel="8n"):
    """Laplacian response (float64)."""
    return convolve2d(img.astype(np.float64), LAPLACIAN[kernel])


def log_filter(img, sigma=2.0):
    """Laplacian of Gaussian: Gaussian smoothing (size 2*ceil(3 sigma)+1), then Laplacian."""
    size = 2 * int(np.ceil(3 * sigma)) + 1
    smooth = convolve2d(img.astype(np.float64), gaussian_kernel(size, sigma))
    return convolve2d(smooth, LAPLACIAN["4n"])


def zero_crossings(response, thr=0.0):
    """True where opposite neighbours have strictly opposite signs and differ by more than thr.

    A zero next to a positive value is not a crossing (a * b must be negative).
    """
    p = np.pad(response, 1, mode="edge")
    h, w = response.shape
    out = np.zeros((h, w), bool)
    for (dy1, dx1), (dy2, dx2) in (((0, 0), (2, 2)), ((0, 1), (2, 1)),
                                   ((0, 2), (2, 0)), ((1, 0), (1, 2))):
        a, b = p[dy1:dy1 + h, dx1:dx1 + w], p[dy2:dy2 + h, dx2:dx2 + w]
        out |= (a * b < 0) & (np.abs(a - b) > thr)
    return out


def point_detect(img, t):
    """Isolated points: |Laplacian (8-neighbour)| >= t."""
    return np.abs(laplacian(img, "8n")) >= t


def line_response(img, direction="max", polarity="dark"):
    """Line-mask response; direction 0, 45, 90, -45 or "max" over all four.

    polarity "dark" negates the masks so dark lines (cracks) give positive values.
    """
    sign = -1.0 if polarity == "dark" else 1.0
    f = img.astype(np.float64)
    if direction == "max":
        return np.max([sign * convolve2d(f, m) for m in LINE_MASKS.values()], axis=0)
    return sign * convolve2d(f, LINE_MASKS[direction])


def line_detect(img, t, direction="max", polarity="dark"):
    """Pixels whose line response is at least t."""
    return line_response(img, direction, polarity) >= t


# ---------------------------------------------------------------- thresholds
def threshold_dark(img, t):
    """Crack mask: pixels at or below t."""
    return img <= t


def global_threshold(img, eps=0.5):
    """Basic iterative global threshold: T = (mean below + mean above) / 2 until stable."""
    f = img.astype(np.float64)
    t = f.mean()
    while True:
        low, high = f[f <= t], f[f > t]
        if low.size == 0 or high.size == 0:
            return t
        new = (low.mean() + high.mean()) / 2
        if abs(new - t) < eps:
            return new
        t = new


def otsu_threshold(img):
    """Otsu: the level k that maximises between-class variance; class 0 is <= k.

    sigma_B^2(k) = (mu_T * omega(k) - mu(k))^2 / (omega(k) * (1 - omega(k)))
    """
    p = np.bincount(img.ravel(), minlength=256) / img.size
    omega = np.cumsum(p)
    mu = np.cumsum(np.arange(256) * p)
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu[-1] * omega - mu) ** 2 / denom, -1.0)
    if sigma_b.max() < 0:          # single gray level: nothing to separate
        return int(img.flat[0])
    return int(np.flatnonzero(sigma_b == sigma_b.max()).mean())


def adaptive_threshold(img, block=31, c=5, method="mean"):
    """Crack = pixel <= local mean (or Gaussian-weighted mean) - c over a block x block window.

    The local mean uses an integral image on a replicate-padded image (as OpenCV does).
    """
    if block % 2 == 0:
        raise ValueError("block must be odd")
    f = img.astype(np.float64)
    if method == "mean":
        r = block // 2
        s = np.pad(np.pad(f, r, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        h, w = f.shape
        local = (s[block:block + h, block:block + w] - s[:h, block:block + w]
                 - s[block:block + h, :w] + s[:h, :w]) / block ** 2
    elif method == "gaussian":
        local = convolve2d(f, gaussian_kernel(block))
    else:
        raise ValueError(f"unknown method {method!r}")
    return f <= local - c


# ---------------------------------------------------------------- linking and Hough
def link_gaps(edges, max_gap=3):
    """Fill runs of at most max_gap background pixels between edge pixels, along rows and
    along columns (simple local edge linking)."""
    out = edges.copy()
    for grid in (out, out.T):            # out.T is a view: filling it fills out
        for row in grid:
            idx = np.flatnonzero(row)
            gaps = np.flatnonzero((np.diff(idx) > 1) & (np.diff(idx) <= max_gap + 1))
            for g in gaps:
                row[idx[g] + 1: idx[g + 1]] = True
    return out


def hough_lines(edges, thetas=None):
    """Line Hough transform: each edge pixel votes rho = x cos(theta) + y sin(theta).

    Same layout as skimage.transform.hough_line: rows are rho from -D to D (D = ceil of the
    image diagonal), columns are thetas. Returns (accumulator, thetas, rhos).
    """
    if thetas is None:
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    offset = int(np.ceil(np.hypot(*edges.shape)))
    ys, xs = np.nonzero(edges)
    rho = xs[:, None] * np.cos(thetas) + ys[:, None] * np.sin(thetas)
    rho_idx = (np.sign(rho) * np.floor(np.abs(rho) + 0.5)).astype(int) + offset  # half away
    acc = np.zeros((2 * offset + 1, len(thetas)), np.uint64)
    np.add.at(acc, (rho_idx, np.broadcast_to(np.arange(len(thetas)), rho_idx.shape)), 1)
    return acc, thetas, np.arange(-offset, offset + 1)
