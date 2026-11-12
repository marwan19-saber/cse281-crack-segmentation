"""Tests for src/metrics.py: hand-worked examples (plan Section 19.1), edge cases, library check."""
import math

import numpy as np
import pytest
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

from src.metrics import count_abs_error, dice, iou, mse, psnr

# Hand example: diff = [[-2, 0], [0, 4]] -> squares 4, 0, 0, 16 -> MSE = 20 / 4 = 5
A = np.array([[10, 20], [30, 40]], dtype=np.uint8)
B = np.array([[12, 20], [30, 36]], dtype=np.uint8)

# Hand example: |pred AND gt| = 2, |pred OR gt| = 4 -> IoU = 0.5, Dice = 2*2/(3+3) = 0.667
PRED = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
GT = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)


def test_mse_hand_example():
    assert mse(A, B) == 5.0


def test_mse_uint8_does_not_wrap_around():
    # 10 - 12 in uint8 arithmetic is 254, not -2; the result must still be 5.0
    assert mse(A, B) == mse(B, A) == 5.0


def test_psnr_hand_example():
    # 10 * log10(255^2 / 5) = 10 * log10(13005) = 41.14 dB
    assert psnr(A, B) == pytest.approx(41.1411, abs=1e-4)


def test_psnr_identical_images_is_infinite():
    assert psnr(A, A) == math.inf


def test_mse_and_psnr_match_skimage():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    y = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert mse(x, y) == pytest.approx(mean_squared_error(x, y))
    assert psnr(x, y) == pytest.approx(peak_signal_noise_ratio(x, y, data_range=255))


def test_iou_hand_example():
    assert iou(PRED, GT) == 0.5


def test_dice_hand_example():
    assert dice(PRED, GT) == pytest.approx(2 / 3)


def test_iou_and_dice_two_empty_masks_score_one():
    empty = np.zeros((3, 3), dtype=bool)
    assert iou(empty, empty) == 1.0
    assert dice(empty, empty) == 1.0


def test_iou_empty_prediction_against_crack_is_zero():
    assert iou(np.zeros((3, 3), dtype=bool), GT) == 0.0


def test_iou_accepts_0_255_uint8_masks():
    assert iou(PRED.astype(np.uint8) * 255, GT.astype(np.uint8) * 255) == 0.5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mse(A, np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        iou(PRED, np.zeros((2, 2), dtype=bool))


def test_count_abs_error():
    assert count_abs_error(3, 5) == 2
    assert count_abs_error(5, 3) == 2
