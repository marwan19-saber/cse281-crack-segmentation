"""Tests for bonus/compression/codec.py (MEM3, Bonus 4): exact round trips and edge cases."""
import numpy as np
import pytest

from bonus.compression import codec

RNG = np.random.default_rng(5)


def test_rle_hand_example():
    # row [0,0,0,1,1,0] -> runs (0,3), (1,2), (0,1)
    mask = np.array([[0, 0, 0, 1, 1, 0]], bool)
    assert codec.rle_runs(mask) == [(0, 3), (1, 2), (0, 1)]


@pytest.mark.parametrize("mask", [
    np.zeros((7, 9), bool), np.ones((7, 9), bool), RNG.random((40, 50)) < 0.1,
    np.eye(5, dtype=bool)])
def test_rle_round_trip_is_exact(mask):
    assert np.array_equal(codec.rle_decode(codec.rle_encode(mask)), mask)


def test_rle_compresses_a_sparse_crack_mask():
    mask = np.zeros((100, 200), bool); mask[50, 10:190] = True
    assert len(codec.rle_encode(mask)) < mask.size / 8 / 10


def test_huffman_hand_example_code_lengths():
    # frequencies 5, 2, 1, 1 -> optimal code lengths 1, 2, 3, 3
    lengths = codec.huffman_code_lengths(np.array([5, 2, 1, 1]))
    assert sorted(lengths.values()) == [1, 2, 3, 3]


@pytest.mark.parametrize("img", [
    RNG.integers(0, 256, (30, 40), dtype=np.uint8), np.full((5, 6), 77, np.uint8),
    np.array([[0]], np.uint8)])
def test_huffman_round_trip_is_exact(img):
    assert np.array_equal(codec.huffman_decode(codec.huffman_encode(img)), img)


def test_huffman_beats_8_bits_on_a_skewed_image():
    img = np.full((50, 50), 200, np.uint8); img[::7] = 30
    assert len(codec.huffman_encode(img)) < img.size
