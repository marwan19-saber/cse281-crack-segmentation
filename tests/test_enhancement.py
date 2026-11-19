"""Tests for src/enhancement.py (MEM2): hand examples, library comparisons, edge cases."""
import cv2
import numpy as np
import pytest
from skimage.exposure import adjust_gamma

from src import enhancement as enh

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 256, (48, 64), dtype=np.uint8)


# ---------------------------------------------------------------- histogram, point operations
def test_histogram_counts_every_pixel():
    h = enh.histogram(IMG)
    assert h.shape == (256,) and h.sum() == IMG.size
    assert h[7] == (IMG == 7).sum()


def test_negative_is_its_own_inverse():
    assert np.array_equal(enh.negative(enh.negative(IMG)), IMG)
    assert enh.negative(np.array([[0, 255]], np.uint8)).tolist() == [[255, 0]]


def test_log_transform_hand_example():
    # c = 255 / ln(256); r = 15 -> 255 * ln(16) / ln(256) = 127.5 -> rounds to 128
    out = enh.log_transform(np.array([[0, 15, 255]], np.uint8))
    assert out.tolist() == [[0, 128, 255]]


def test_gamma_hand_example_and_skimage_match():
    # 255 * (64/255)^0.5 = 127.75 -> 128
    assert enh.gamma_transform(np.array([[64]], np.uint8), 0.5)[0, 0] == 128
    diff = enh.gamma_transform(IMG, 2.2).astype(int) - adjust_gamma(IMG, 2.2).astype(int)
    assert np.abs(diff).max() <= 1


def test_contrast_stretch_maps_percentiles_to_full_range():
    img = np.linspace(100, 150, 101).astype(np.uint8).reshape(1, -1)
    out = enh.contrast_stretch(img, 0, 100)
    assert out.min() == 0 and out.max() == 255


def test_contrast_stretch_constant_image_is_unchanged():
    flat = np.full((4, 4), 90, np.uint8)
    assert np.array_equal(enh.contrast_stretch(flat), flat)


# ---------------------------------------------------------------- histogram equalization
def test_hist_equalize_hand_example_3bit():
    # plan Section 19.2, L = 8: counts r0=2 r1=3 r2=2 r3=1 r7=1 -> 0->2, 1->4, 2->5, 3->6, 7->7
    img = np.array([[0, 0, 1], [1, 1, 2], [2, 3, 7]], np.uint8)
    out = enh.hist_equalize(img, levels=8)
    assert out.tolist() == [[2, 2, 4], [4, 4, 5], [5, 6, 7]]


def test_hist_equalize_cv_variant_matches_opencv():
    assert np.array_equal(enh.hist_equalize(IMG, variant="cv"), cv2.equalizeHist(IMG))


def test_hist_equalize_constant_image_does_not_crash():
    flat = np.full((5, 5), 77, np.uint8)
    assert enh.hist_equalize(flat, variant="cv").shape == flat.shape


# ---------------------------------------------------------------- convolution and smoothing
def test_convolve2d_matches_filter2d_reflect101():
    k = RNG.normal(size=(5, 5))
    ours = enh.convolve2d(IMG, k)
    ref = cv2.filter2D(IMG.astype(np.float64), -1, k, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - ref).max() < 1e-9


def test_convolve2d_identity_kernel_returns_input():
    k = np.zeros((3, 3)); k[1, 1] = 1
    assert np.array_equal(enh.convolve2d(IMG, k), IMG.astype(np.float64))


def test_convolve2d_hand_example_mean_centre():
    # mean of [[1..9]] = 45 / 9 = 5 at the centre; output size (5 - 3 + 2)/1 + 1 = 5 stays N
    x = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    out = enh.convolve2d(x, np.full((3, 3), 1 / 9))
    assert out.shape == (3, 3) and out[1, 1] == pytest.approx(5.0)


def test_convolve2d_even_kernel_raises():
    with pytest.raises(ValueError):
        enh.convolve2d(IMG, np.ones((2, 2)))


def test_mean_filter_matches_cv2_blur():
    ref = cv2.blur(IMG, (5, 5), borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(enh.mean_filter(IMG, 5).astype(int) - ref.astype(int)).max() <= 1


def test_gaussian_kernel_matches_opencv():
    g = cv2.getGaussianKernel(5, 1.2)
    assert np.allclose(enh.gaussian_kernel(5, 1.2), g @ g.T)


def test_gaussian_filter_matches_cv2():
    ref = cv2.GaussianBlur(IMG, (5, 5), 0, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(enh.gaussian_filter(IMG, 5).astype(int) - ref.astype(int)).max() <= 1


def test_smoothing_keeps_constant_image():
    flat = np.full((9, 9), 120, np.uint8)
    assert np.array_equal(enh.mean_filter(flat, 3), flat)
    assert np.array_equal(enh.gaussian_filter(flat, 5), flat)


# ---------------------------------------------------------------- sharpening
def test_laplacian_sharpen_constant_unchanged_and_edge_overshoots():
    flat = np.full((7, 7), 100, np.uint8)
    assert np.array_equal(enh.laplacian_sharpen(flat), flat)
    step = np.zeros((7, 8), np.uint8); step[:, 4:] = 100
    out = enh.laplacian_sharpen(step).astype(int)
    assert out[3, 4] > 100 and out[3, 3] == 0      # bright side overshoots, dark side clips


def test_gradient_sharpen_constant_unchanged():
    flat = np.full((9, 9), 80, np.uint8)
    assert np.array_equal(enh.gradient_sharpen(flat), flat)
