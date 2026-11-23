"""Tests for src/segmentation.py (MEM5): hand examples, library comparisons, edge cases."""
import cv2
import numpy as np
import pytest
from skimage.filters import threshold_otsu
from skimage.transform import hough_line

from src import segmentation as seg

RNG = np.random.default_rng(3)
IMG = RNG.integers(0, 256, (40, 56), dtype=np.uint8)
STEP = np.array([[10, 10, 10], [10, 10, 10], [200, 200, 200]], np.uint8)


# ---------------------------------------------------------------- gradients
def test_sobel_hand_example():
    # plan Section 19.4: Gy = -(10+20+10) + (200+400+200) = 760, Gx = 0 at the centre
    gx, gy, mag = seg.sobel(STEP)
    assert gx[1, 1] == 0 and gy[1, 1] == 760 and mag[1, 1] == 760


def test_sobel_matches_opencv():
    _, _, mag = seg.sobel(IMG)
    f = IMG.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(mag - np.hypot(gx, gy)).max() < 1e-9


@pytest.mark.parametrize("fn", ["roberts", "prewitt", "sobel"])
def test_gradient_of_constant_image_is_zero(fn):
    _, _, mag = getattr(seg, fn)(np.full((6, 6), 90, np.uint8))
    assert not mag.any()


def test_prewitt_hand_example():
    # Gy = -(10+10+10) + (200+200+200) = 570
    assert seg.prewitt(STEP)[1][1, 1] == 570


def test_roberts_detects_diagonal_difference():
    img = np.array([[0, 0], [0, 100]], np.uint8)
    _, _, mag = seg.roberts(np.pad(img, 2, mode="edge"))
    assert mag.max() > 0


# ---------------------------------------------------------------- Laplacian, LoG, points
def test_laplacian_matches_opencv_ksize1():
    ref = cv2.Laplacian(IMG.astype(np.float64), cv2.CV_64F, ksize=1,
                        borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(seg.laplacian(IMG, kernel="4n") - ref).max() < 1e-9


def test_log_zero_crossings_mark_a_step_edge():
    img = np.zeros((20, 20), np.uint8); img[:, 10:] = 200
    # threshold relative to the strongest response: the truncated Gaussian leaves tiny
    # side-lobe sign changes at its edge (5 px away) that an absolute thr of 1 would keep
    resp = seg.log_filter(img, sigma=1.5)
    zc = seg.zero_crossings(resp, thr=0.05 * np.abs(resp).max())
    cols = np.flatnonzero(zc[10])
    assert cols.size and cols.min() >= 8 and cols.max() <= 11


def test_point_detect_finds_isolated_pixel():
    img = np.full((9, 9), 100, np.uint8); img[4, 4] = 200
    found = seg.point_detect(img, t=500)
    assert found[4, 4] and found.sum() == 1


def test_line_response_prefers_matching_direction():
    img = np.full((9, 9), 200, np.uint8); img[4, :] = 50        # dark horizontal line
    r0 = seg.line_response(img, direction=0)
    r90 = seg.line_response(img, direction=90)
    assert r0[4, 4] > 0 and r0[4, 4] > r90[4, 4]


def test_line_detect_max_over_directions_finds_dark_diagonal():
    img = np.full((9, 9), 200, np.uint8); np.fill_diagonal(img, 50)
    assert seg.line_detect(img, t=300)[4, 4]


# ---------------------------------------------------------------- thresholds
def test_otsu_matches_skimage_and_opencv():
    t = seg.otsu_threshold(IMG)
    assert abs(t - threshold_otsu(IMG)) <= 1
    t_cv, _ = cv2.threshold(IMG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(t - t_cv) <= 1


def test_otsu_bimodal_hand_example():
    img = np.array([[10, 10, 12, 12], [10, 10, 12, 12], [200, 200, 205, 205],
                    [200, 200, 205, 205]], np.uint8)
    assert 12 <= seg.otsu_threshold(img) < 200


def test_otsu_constant_image_does_not_crash():
    assert seg.otsu_threshold(np.full((4, 4), 9, np.uint8)) == 9


def test_iterative_global_threshold_bimodal():
    img = np.array([[20] * 8 + [220] * 8], np.uint8)
    assert seg.global_threshold(img) == pytest.approx(120, abs=0.5)


def test_threshold_dark_marks_pixels_at_or_below_t():
    img = np.array([[5, 6, 7]], np.uint8)
    assert seg.threshold_dark(img, 6).tolist() == [[True, True, False]]


def test_adaptive_mean_agrees_with_opencv():
    ours = seg.adaptive_threshold(IMG, block=11, c=5, method="mean")
    ref = cv2.adaptiveThreshold(IMG, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                11, 5) > 0
    assert (ours != ref).mean() < 0.01


def test_adaptive_finds_dark_line_under_a_gradient():
    ramp = np.tile(np.linspace(80, 240, 60), (30, 1)).astype(np.uint8)
    ramp[15, :] -= 40                                   # crack darker than its surroundings
    mask = seg.adaptive_threshold(ramp, block=9, c=10)
    assert mask[15, 5:55].all() and not mask[5].any()


def test_adaptive_even_block_raises():
    with pytest.raises(ValueError):
        seg.adaptive_threshold(IMG, block=10, c=2)


# ---------------------------------------------------------------- edge linking and Hough
def test_link_gaps_bridges_short_gaps_only():
    edges = np.zeros((3, 12), bool)
    edges[1, [0, 1, 4, 5, 10, 11]] = True               # gaps of 2 and 4
    linked = seg.link_gaps(edges, max_gap=2)
    assert linked[1, 2:4].all() and not linked[1, 6:10].any()


def test_hough_matches_skimage_accumulator():
    edges = np.zeros((30, 40), bool); edges[10, 5:35] = True
    thetas = np.deg2rad(np.arange(-90, 90))
    acc, _, _ = seg.hough_lines(edges, thetas)
    ref, _, _ = hough_line(edges, theta=thetas)
    assert np.array_equal(acc, ref)


def test_hough_peak_is_the_drawn_line():
    # line y = 10: rho = x cos(theta) + y sin(theta) peaks at theta = -90 deg, rho = -10
    edges = np.zeros((30, 40), bool); edges[10, 5:35] = True
    acc, thetas, rhos = seg.hough_lines(edges, np.deg2rad(np.arange(-90, 90)))
    r, t = np.unravel_index(np.argmax(acc), acc.shape)
    assert rhos[r] == -10 and round(np.rad2deg(thetas[t])) == -90
