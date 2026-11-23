"""Tests for src/frequency.py (MEM4): DFT round trip, transfer functions at D0, filtering."""
import cv2
import numpy as np
import pytest

from src import frequency as freq

RNG = np.random.default_rng(2)
IMG = RNG.integers(0, 256, (32, 48), dtype=np.uint8)


def test_naive_dft_matches_numpy_fft():
    small = IMG[:8, :12].astype(np.float64)
    assert np.allclose(freq.dft2(small), np.fft.fft2(small))


def test_idft_inverts_dft():
    assert np.allclose(freq.idft2(freq.dft2(IMG[:8, :8])).real, IMG[:8, :8])


def test_centering_by_minus_one_power_equals_fftshift():
    f = IMG.astype(np.float64)
    centred = np.fft.fft2(f * freq.centering_sign(f.shape))
    assert np.allclose(centred, np.fft.fftshift(np.fft.fft2(f)))


def test_dc_term_is_sum_of_pixels():
    spec = np.fft.fft2(IMG.astype(np.float64))
    assert spec[0, 0].real == pytest.approx(IMG.sum())


def test_centered_spectrum_peak_is_in_the_middle():
    s = freq.centered_spectrum(IMG)
    assert np.unravel_index(np.argmax(s), s.shape) == (16, 24)


def test_transfer_functions_at_cutoff():
    shape = (64, 64)
    d = freq.distance_grid(shape)
    at = np.argwhere(np.isclose(d, 10))[0]
    assert freq.ideal_lowpass(shape, 10)[tuple(at)] == 1.0
    assert freq.butterworth_lowpass(shape, 10, n=2)[tuple(at)] == pytest.approx(0.5)
    assert freq.gaussian_lowpass(shape, 10)[tuple(at)] == pytest.approx(np.exp(-0.5))


def test_lowpass_plus_highpass_is_one():
    h = freq.butterworth_lowpass((40, 40), 8)
    assert np.allclose(h + freq.highpass(h), 1.0)


def test_all_pass_filter_returns_input():
    ones = np.ones((2 * IMG.shape[0], 2 * IMG.shape[1]))
    assert np.array_equal(freq.apply_filter(IMG, ones), IMG)


def test_gaussian_lowpass_is_close_to_matched_spatial_gaussian():
    img = cv2.GaussianBlur(IMG, (7, 7), 3)  # smooth input avoids border effects dominating
    d0 = 20.0
    shape = (2 * img.shape[0], 2 * img.shape[1])
    # matched spatial sigma per axis: P / (2 pi D0) for rows, Q / (2 pi D0) for columns
    sy, sx = (s / (2 * np.pi * d0) for s in shape)
    ours = freq.apply_filter(img, freq.gaussian_lowpass(shape, d0)).astype(float)
    ref = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy,
                           borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - ref)[4:-4, 4:-4].mean() < 3


def test_lowpass_removes_high_frequency_energy():
    x = np.indices((64, 64)).sum(axis=0) % 2 * 255   # checkerboard = highest frequency
    out = freq.apply_filter(x.astype(np.uint8), freq.ideal_lowpass((128, 128), 10))
    assert out[16:48, 16:48].std() < 5
