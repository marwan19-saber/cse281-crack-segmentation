"""DFT, centred spectrum and ideal / Butterworth / Gaussian frequency-domain filters (MEM3).

Filtering steps (Gonzalez & Woods): pad the M x N image to P x Q (normally 2M x 2N) to avoid
wrap-around, multiply by (-1)^(x+y) to centre the spectrum, DFT, multiply by H(u, v), inverse
DFT, take the real part, multiply by (-1)^(x+y) again, crop back to M x N.
Padding uses reflection (not zeros) so image borders do not darken.
"""
import numpy as np


def _dft_matrix(n):
    """n x n DFT matrix W[u, x] = exp(-2 pi i u x / n)."""
    idx = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n)


def dft2(f):
    """Naive 2-D DFT, F = W_M f W_N (O(MN(M+N))): for small arrays and for understanding."""
    f = np.asarray(f, np.float64)
    return _dft_matrix(f.shape[0]) @ f @ _dft_matrix(f.shape[1])


def idft2(spectrum):
    """Inverse of dft2: f = conj(W_M) F conj(W_N) / (M N)."""
    m, n = spectrum.shape
    return np.conj(_dft_matrix(m)) @ spectrum @ np.conj(_dft_matrix(n)) / (m * n)


def centering_sign(shape):
    """(-1)^(x + y): multiplying by it moves F(0, 0) to the centre (M/2, N/2)."""
    y, x = np.indices(shape)
    return np.where((x + y) % 2 == 0, 1.0, -1.0)


def centered_spectrum(img):
    """log(1 + |F|) with the zero frequency in the centre, for display."""
    f = img.astype(np.float64)
    return np.log1p(np.abs(np.fft.fft2(f * centering_sign(f.shape))))


def distance_grid(shape):
    """D(u, v): distance of each frequency from the centre (P/2, Q/2)."""
    u, v = np.indices(shape)
    return np.hypot(u - shape[0] // 2, v - shape[1] // 2)


def ideal_lowpass(shape, d0):
    """H = 1 inside radius d0, 0 outside: sharp cutoff, causes ringing (sinc in space)."""
    return (distance_grid(shape) <= d0).astype(np.float64)


def butterworth_lowpass(shape, d0, n=2):
    """H = 1 / (1 + (D / d0)^(2n)); H = 0.5 at D = d0; low n gives little ringing."""
    return 1.0 / (1.0 + (distance_grid(shape) / d0) ** (2 * n))


def gaussian_lowpass(shape, d0):
    """H = exp(-D^2 / (2 d0^2)); H = 0.607 at D = d0; no ringing."""
    return np.exp(-distance_grid(shape) ** 2 / (2.0 * d0 ** 2))


def highpass(h_lowpass):
    """High-pass from low-pass: H_hp = 1 - H_lp."""
    return 1.0 - h_lowpass


def apply_filter(img, h):
    """Filter a uint8 image with a centred transfer function h of the padded size (P, Q)."""
    m, n = img.shape
    p, q = h.shape
    padded = np.pad(img.astype(np.float64), ((0, p - m), (0, q - n)), mode="reflect")
    sign = centering_sign((p, q))
    # Speed (Bonus 5): the input is real and h depends only on the distance from (P/2, Q/2),
    # so h(k) = h(-k mod P) and the output is real: the half spectrum of rfft2 is enough.
    half = np.fft.rfft2(padded * sign) * h[:, : q // 2 + 1]
    g = np.fft.irfft2(half, s=(p, q)) * sign
    return np.clip(np.rint(g[:m, :n]), 0, 255).astype(np.uint8)
