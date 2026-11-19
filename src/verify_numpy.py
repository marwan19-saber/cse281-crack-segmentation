"""Measure every team NumPy implementation against its library counterpart (MEM1).

    python -m src.verify_numpy        # writes experiments/results/numpy_vs_library.csv

Input: a real dev image at working size 1008 (and its mask for binary operations).
Max |diff| = 0 means identical output; other values are explained in the notes column.
"""
import csv
import time

import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import adjust_gamma
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label

from src import data, enhancement, frequency, metrics, morphology, restoration, segmentation


def _diff(a, b):
    """Max absolute difference as float."""
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def _timed(fn):
    """(result, seconds)."""
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def rows(gray, mask):
    """One dict per comparison."""
    f = gray.astype(np.float64)
    r101 = cv2.BORDER_REFLECT_101
    out = []

    def add(member, op, ours_fn, ref_fn, border, note=""):
        ours, t_ours = _timed(ours_fn)
        ref, t_ref = _timed(ref_fn)
        out.append({"member": member, "operation": op, "max_abs_diff": _diff(ours, ref),
                    "border": border, "numpy_s": round(t_ours, 4),
                    "library_s": round(t_ref, 4),
                    "note": note})

    k5 = np.random.default_rng(0).normal(size=(5, 5))
    add("MEM1", "block-average downsampling x4", lambda: data.reduce_resolution(gray, 4),
        lambda: cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4),
                           interpolation=cv2.INTER_AREA), "n/a", "rounding of .5 averages")
    add("MEM1", "PSNR", lambda: metrics.psnr(gray, enhancement.mean_filter(gray, 3)),
        lambda: __import__("skimage.metrics").metrics.peak_signal_noise_ratio(
            gray, enhancement.mean_filter(gray, 3), data_range=255), "n/a")
    add("MEM2", "convolve2d (random 5x5)", lambda: enhancement.convolve2d(gray, k5),
        lambda: cv2.filter2D(f, -1, k5, borderType=r101), "reflect-101", "float64 round-off")
    add("MEM2", "histogram equalization", lambda: enhancement.hist_equalize(gray, "cv"),
        lambda: cv2.equalizeHist(gray), "n/a")
    add("MEM2", "Gaussian smoothing k=5", lambda: enhancement.gaussian_filter(gray, 5),
        lambda: cv2.GaussianBlur(gray, (5, 5), 0, borderType=r101), "reflect-101",
        "OpenCV 8-bit fixed-point arithmetic")
    add("MEM2", "gamma 0.5", lambda: enhancement.gamma_transform(gray, 0.5),
        lambda: adjust_gamma(gray, 0.5), "n/a", "rounding vs truncation")
    add("MEM3", "median 5x5", lambda: restoration.median_filter(gray, 5),
        lambda: ndimage.median_filter(gray, 5, mode="mirror"), "reflect-101 (= mirror)")
    add("MEM3", "max 3x3", lambda: restoration.max_filter(gray, 3),
        lambda: ndimage.maximum_filter(gray, 3, mode="mirror"), "reflect-101 (= mirror)")
    add("MEM3", "centred spectrum", lambda: frequency.centered_spectrum(gray),
        lambda: np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f)))), "n/a",
        "(-1)^(x+y) centring vs fftshift")
    add("MEM4", "Sobel magnitude", lambda: segmentation.sobel(gray)[2],
        lambda: np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3, borderType=r101),
                         cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3, borderType=r101)),
        "reflect-101")
    add("MEM4", "Otsu threshold", lambda: segmentation.otsu_threshold(gray),
        lambda: threshold_otsu(gray), "n/a", "skimage foreground is > t")
    ours_ad = segmentation.adaptive_threshold(gray, 31, 15)
    ref_ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15) > 0
    add("MEM4", "adaptive mean 31/15", lambda: ours_ad.astype(float),
        lambda: ref_ad.astype(float), "replicate",
        f"OpenCV rounds the local mean; {np.mean(ours_ad != ref_ad):.3%} of pixels differ")
    se = morphology.structuring_element("disk", 5)
    add("MEM5", "erosion disk 5", lambda: morphology.erode(mask, se),
        lambda: cv2.erode(mask.astype(np.uint8), se.astype(np.uint8)) > 0,
        "outside = 1 (OpenCV default)")
    add("MEM5", "hole filling", lambda: morphology.fill_holes(mask),
        lambda: ndimage.binary_fill_holes(mask), "border-seeded")
    add("MEM5", "labelling (object count difference)", lambda: morphology.label(mask)[1],
        lambda: sk_label(mask, connectivity=2).max(), "n/a", "8-connectivity")
    return out


def main():
    """Run on the first dev image and write the CSV."""
    pairs = {img.stem: (img, gt) for img, gt in data.list_pairs("data/raw/concrete_crack")}
    name = data.load_split("data/split.json")["dev"][0]
    gray = data.load_image(pairs[name][0], size=1008)
    mask = data.load_ground_truth(pairs[name][1], size=1008)
    result = rows(gray, mask)
    path = "experiments/results/numpy_vs_library.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    for r in result:
        print(f"{r['member']} {r['operation']:<52} max|diff| {r['max_abs_diff']:<10.4g} "
              f"NumPy {r['numpy_s']:.3f}s  lib {r['library_s']:.3f}s  {r['note']}")


if __name__ == "__main__":
    main()
