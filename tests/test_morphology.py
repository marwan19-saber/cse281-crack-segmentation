"""Tests for src/morphology.py (MEM5): hand example, OpenCV/SciPy/skimage comparisons."""
import cv2
import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import label as sk_label

from src import morphology as mor

RNG = np.random.default_rng(4)
MASK = RNG.random((40, 50)) < 0.35
BLOCK = np.zeros((5, 5), bool); BLOCK[1:4, 1:4] = True


# ---------------------------------------------------------------- structuring elements
def test_structuring_element_shapes():
    assert mor.structuring_element("square", 3).sum() == 9
    assert mor.structuring_element("cross", 3).tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    disk = mor.structuring_element("disk", 5)
    assert disk[2, 2] and disk[0, 2] and not disk[0, 0]
    assert mor.structuring_element("line", 5, angle=0)[2].all()
    assert np.array_equal(mor.structuring_element("line", 3, angle=45),
                          np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], bool))


def test_even_size_raises():
    with pytest.raises(ValueError):
        mor.structuring_element("square", 4)


# ---------------------------------------------------------------- erosion / dilation
def test_erosion_hand_example():
    # plan Section 19.5: only the centre of the 3x3 block survives erosion by a 3x3 square
    out = mor.erode(BLOCK, mor.structuring_element("square", 3))
    assert out.sum() == 1 and out[2, 2]


def test_opening_of_block_restores_it():
    assert np.array_equal(mor.opening(BLOCK, mor.structuring_element("square", 3)), BLOCK)


@pytest.mark.parametrize("shape, size", [("square", 3), ("disk", 5), ("cross", 3)])
def test_erode_dilate_match_opencv(shape, size):
    se = mor.structuring_element(shape, size)
    m8 = MASK.astype(np.uint8)
    assert np.array_equal(mor.erode(MASK, se), cv2.erode(m8, se.astype(np.uint8)) > 0)
    assert np.array_equal(mor.dilate(MASK, se), cv2.dilate(m8, se.astype(np.uint8)) > 0)


def test_duality_away_from_border():
    # erode(A) = not dilate(not A) for a symmetric SE; the border convention (outside = 1 for
    # erosion, 0 for dilation) breaks the identity only on the outermost ring
    se = mor.structuring_element("square", 3)
    assert np.array_equal(mor.erode(MASK, se)[1:-1, 1:-1], ~mor.dilate(~MASK, se)[1:-1, 1:-1])


def test_opening_and_closing_are_idempotent():
    se = mor.structuring_element("disk", 5)
    o = mor.opening(MASK, se)
    c = mor.closing(MASK, se)
    assert np.array_equal(mor.opening(o, se), o)
    assert np.array_equal(mor.closing(c, se), c)


# ---------------------------------------------------------------- hole filling, labelling
def test_fill_holes_matches_scipy():
    assert np.array_equal(mor.fill_holes(MASK), ndimage.binary_fill_holes(MASK))


def test_fill_holes_ring_becomes_disk():
    ring = np.zeros((9, 9), bool); ring[2:7, 2:7] = True; ring[4, 4] = False
    assert mor.fill_holes(ring)[4, 4]
    assert not mor.fill_holes(np.zeros((4, 4), bool)).any()


@pytest.mark.parametrize("connectivity", [4, 8])
def test_label_matches_skimage_partition(connectivity):
    labels, n = mor.label(MASK, connectivity)
    ref = sk_label(MASK, connectivity=1 if connectivity == 4 else 2)
    assert n == ref.max()
    pairs = set(zip(labels[MASK].tolist(), ref[MASK].tolist()))
    assert len(pairs) == n                   # one-to-one mapping between label sets
    assert (labels[~MASK] == 0).all()


def test_label_diagonal_neighbours_depend_on_connectivity():
    diag = np.array([[1, 0], [0, 1]], bool)
    assert mor.label(diag, 4)[1] == 2 and mor.label(diag, 8)[1] == 1


def test_remove_small_objects():
    m = np.zeros((10, 10), bool); m[0, 0] = True; m[5:8, 5:8] = True
    out = mor.remove_small_objects(m, min_area=4)
    assert not out[0, 0] and out[5:8, 5:8].all()


# ---------------------------------------------------------------- measurement and decision
def test_measure_square_area_bbox_centroid():
    m = np.zeros((10, 12), bool); m[2:5, 3:7] = True
    (obj,) = mor.measure(mor.label(m)[0])
    assert obj["area"] == 12 and obj["bbox"] == (2, 3, 5, 7)
    assert obj["centroid"] == pytest.approx((3.0, 4.5))


def test_round_blob_is_more_circular_than_thin_line():
    blob = mor.structuring_element("disk", 15)
    line = np.zeros((5, 40), bool); line[2, 2:38] = True
    c_blob = mor.measure(mor.label(blob)[0])[0]["circularity"]
    c_line = mor.measure(mor.label(line)[0])[0]["circularity"]
    # 1-px line: perimeter = area, so circularity = 4 pi / 36 = 0.35 (falls as cracks lengthen)
    assert c_blob > 0.6 and c_line < 0.4


def test_select_objects_drops_small_and_round():
    m = np.zeros((40, 60), bool)
    m[5, 5:55] = True                                      # crack-like line, kept
    m[20:35, 20:35] = mor.structuring_element("disk", 15)  # round stain, dropped
    m[38, 2] = True                                        # speck, dropped
    out = mor.select_objects(m, min_area=10, max_circularity=0.5)
    assert out[5, 5:55].all() and not out[27, 27] and not out[38, 2]


def test_crack_measurements_and_severity():
    m = np.zeros((10, 10), bool); m[5, 1:9] = True
    meas = mor.crack_measurements(m)
    assert meas["area_pct"] == pytest.approx(8.0) and meas["length_px"] == 8
    assert mor.severity(0.5, (1.0, 3.0)) == "minor"
    assert mor.severity(2.0, (1.0, 3.0)) == "moderate"
    assert mor.severity(3.0, (1.0, 3.0)) == "severe"
