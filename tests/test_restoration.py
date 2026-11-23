"""Tests for src/restoration.py (MEM4): hand window, SciPy/OpenCV comparisons, edge cases."""
import cv2
import numpy as np
import pytest
from scipy import ndimage

from src import restoration as rest

RNG = np.random.default_rng(1)
IMG = RNG.integers(0, 256, (40, 56), dtype=np.uint8)
# Hand window (plan Section 19.3): sorted 0,10,10,11,11,12,12,13,255
WINDOW = np.array([[10, 12, 11], [13, 255, 12], [11, 10, 0]], np.uint8)


def centre(fn, **kw):
    """Value the filter gives at the centre of the 3x3 hand window."""
    return fn(WINDOW, 3, **kw)[1, 1]


def test_hand_window_values():
    assert centre(rest.median_filter) == 11
    assert centre(rest.max_filter) == 255
    assert centre(rest.min_filter) == 0
    assert centre(rest.midpoint_filter) == 128            # (0 + 255) / 2 = 127.5 -> 128
    assert centre(rest.arithmetic_mean) == 37             # 334 / 9 = 37.1
    assert centre(rest.alpha_trimmed_mean, d=2) == 11     # drop 0 and 255: 79 / 7 = 11.29


@pytest.mark.parametrize("fn, ref", [
    (rest.median_filter, ndimage.median_filter),
    (rest.max_filter, ndimage.maximum_filter),
    (rest.min_filter, ndimage.minimum_filter),
])
def test_order_statistics_equal_scipy_mirror(fn, ref):
    assert np.array_equal(fn(IMG, 5), ref(IMG, size=5, mode="mirror"))


def test_median_equals_opencv_away_from_border():
    # cv2.medianBlur always uses replicate borders (no option); ours uses reflect-101 like
    # SciPy's "mirror" (exact match above). Away from the 1-pixel border they must agree.
    ours, ref = rest.median_filter(IMG, 3), cv2.medianBlur(IMG, 3)
    assert np.array_equal(ours[1:-1, 1:-1], ref[1:-1, 1:-1])


def test_arithmetic_mean_matches_cv2_blur():
    ref = cv2.blur(IMG, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(rest.arithmetic_mean(IMG, 3).astype(int) - ref.astype(int)).max() <= 1


def test_alpha_trimmed_limits():
    # d = 0 is the arithmetic mean; d = k*k - 1 is the median
    assert np.array_equal(rest.alpha_trimmed_mean(IMG, 3, d=0), rest.arithmetic_mean(IMG, 3))
    assert np.array_equal(rest.alpha_trimmed_mean(IMG, 3, d=8), rest.median_filter(IMG, 3))


def test_contraharmonic_q0_is_arithmetic_mean():
    assert np.array_equal(rest.contraharmonic_mean(IMG, 3, q=0), rest.arithmetic_mean(IMG, 3))


def test_contraharmonic_positive_q_removes_pepper():
    img = np.full((9, 9), 100, np.uint8); img[4, 4] = 0
    assert rest.contraharmonic_mean(img, 3, q=1.5)[4, 4] > 95


def test_geometric_and_harmonic_keep_constant_image():
    flat = np.full((7, 7), 60, np.uint8)
    for fn in (rest.geometric_mean, rest.harmonic_mean, rest.midpoint_filter):
        assert np.array_equal(fn(flat, 3), flat)


def test_harmonic_mean_handles_zero_pixels():
    img = np.zeros((5, 5), np.uint8)
    assert np.array_equal(rest.harmonic_mean(img, 3), img)


def test_even_window_raises():
    with pytest.raises(ValueError):
        rest.median_filter(IMG, 4)
