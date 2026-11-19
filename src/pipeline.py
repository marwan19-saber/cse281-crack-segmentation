"""Baselines, pipeline integration and experiment logging (MEM1).

Cracks are darker than the concrete, so every threshold here marks pixels BELOW T as crack.
All stages use the team's NumPy modules (enhancement, restoration, frequency, segmentation,
morphology); the earlier OpenCV stand-ins were replaced on the integration branch.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import (data, enhancement, frequency, metrics, morphology, restoration,
                 segmentation)

METHODS = {}  # name -> function(gray uint8) -> bool mask; filled below

LOG_COLUMNS = ["run_id", "date", "author", "pipeline", "stage", "method", "params_json",
               "seed", "split", "n_images", "time_s", "metric", "value", "notes"]


def baseline_trivial(gray):
    """Spec trivial baseline: fixed threshold 127 on the raw gray image (crack = dark)."""
    return np.asarray(gray) < 127


def baseline_simple(gray):
    """Spec simple baseline: Otsu threshold (crack = at or below T), then one 3x3 opening."""
    binary = segmentation.threshold_dark(gray, segmentation.otsu_threshold(gray))
    return morphology.opening(binary, morphology.structuring_element("square", 3))


def log_run(path, *, author="", pipeline="", stage="", method="", params=None, seed="",
            split="", n_images="", time_s="", metric="", value="", notes=""):
    """Append one experiment row to the CSV log (creates the file with a header if needed).

    Only the LOG_COLUMNS fields are accepted, so a misspelled field fails loudly.
    run_id is the date plus the row number, which is unique within the file.
    """
    path = Path(path)
    is_new = not path.exists() or path.stat().st_size == 0
    n_rows = 0
    if not is_new:
        with path.open(encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
    today = datetime.now().strftime("%Y-%m-%d")
    row = {
        "run_id": f"{today.replace('-', '')}-{n_rows + 1:04d}", "date": today,
        "author": author, "pipeline": pipeline, "stage": stage, "method": method,
        "params_json": json.dumps(params or {}, sort_keys=True), "seed": seed, "split": split,
        "n_images": n_images, "time_s": time_s, "metric": metric, "value": value,
        "notes": notes,
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row["run_id"]


METHODS.update(baseline_trivial=baseline_trivial, baseline_simple=baseline_simple)


# ---------------------------------------------------------------- stage registry
# Every stage takes the previous output (uint8 image or bool mask) plus keyword parameters.
# Stages call the team's NumPy modules; owner in brackets.
def _gaussian(img, ksize=5, sigma=0):
    """[MEM2] Gaussian smoothing."""
    return enhancement.gaussian_filter(img, ksize, sigma)


def _mean(img, ksize=3):
    """[MEM2] Box smoothing."""
    return enhancement.mean_filter(img, ksize)


def _median(img, ksize=3):
    """[MEM3] Median filter."""
    return restoration.median_filter(img, ksize)


def _flatten_background(img, d0=6.0):
    """[MEM3] Divide by the low-frequency background (Gaussian low-pass, cutoff d0) to remove
    shadows and vignetting; the result is rescaled to the image's mean brightness."""
    f = img.astype(np.float64)
    shape = (2 * img.shape[0], 2 * img.shape[1])
    background = frequency.apply_filter(img, frequency.gaussian_lowpass(shape, d0))
    flat = f / np.maximum(background.astype(np.float64), 1.0) * f.mean()
    return np.clip(np.rint(flat), 0, 255).astype(np.uint8)


def _threshold_fixed(img, t=127):
    """Crack = pixels darker than t (spec trivial baseline form)."""
    return img < t


def _threshold_otsu(img):
    """[MEM4] Crack = pixels at or below Otsu's threshold."""
    return segmentation.threshold_dark(img, segmentation.otsu_threshold(img))


def _threshold_adaptive(img, block=31, c=10, method="mean"):
    """[MEM4] Crack = pixel <= local mean - c."""
    return segmentation.adaptive_threshold(img, block, c, method)


def _threshold_lines(img, t=60):
    """[MEM4] Crack = dark-line mask response (max over 4 directions) >= t."""
    return segmentation.line_detect(img, t)


def _threshold_log(img, sigma=2.0, t=2.0):
    """[MEM4] Crack = LoG response >= t (dark thin structures give a positive LoG)."""
    return segmentation.log_filter(img, sigma) >= t


def _link_gaps(mask, max_gap=3):
    """[MEM4] Bridge short gaps along rows and columns."""
    return segmentation.link_gaps(mask, max_gap)


def _opening(mask, size=3, shape="square"):
    """[MEM5] Opening: remove specks smaller than the structuring element."""
    return morphology.opening(mask, morphology.structuring_element(shape, size))


def _closing(mask, size=3, shape="disk"):
    """[MEM5] Closing: bridge narrow gaps along cracks."""
    return morphology.closing(mask, morphology.structuring_element(shape, size))


def _fill_small_holes(mask, max_area=30):
    """[MEM5] Fill holes up to max_area pixels. Large holes (background enclosed by crossing
    cracks) are kept: filling them would turn concrete into crack."""
    holes = morphology.fill_holes(mask) & ~mask
    return mask | (holes & ~morphology.remove_small_objects(holes, max_area + 1))


def _select_objects(mask, min_area=30, max_circularity=0.5):
    """[MEM5] Keep long thin objects: drop specks and round stains."""
    return morphology.select_objects(mask, min_area, max_circularity)


def _lowpass(img, d0=80.0):
    """[MEM3] Gaussian low-pass smoothing in the frequency domain (cutoff d0 on the 2x padded
    grid); the frequency-domain counterpart of the spatial Gaussian (ablation AB2)."""
    shape = (2 * img.shape[0], 2 * img.shape[1])
    return frequency.apply_filter(img, frequency.gaussian_lowpass(shape, d0))


STAGES = {
    "negative": lambda img: enhancement.negative(img),
    "log": lambda img: enhancement.log_transform(img),
    "gamma": lambda img, gamma=1.0: enhancement.gamma_transform(img, gamma),
    "hist_equalize": lambda img: enhancement.hist_equalize(img),
    "laplacian_sharpen": lambda img: enhancement.laplacian_sharpen(img),
    "gradient_sharpen": lambda img: enhancement.gradient_sharpen(img),
    "lowpass": _lowpass,
    "quantize": lambda img, levels=256: data.quantize_gray_levels(img, levels),
    "gaussian": _gaussian, "mean": _mean, "median": _median,
    "contrast_stretch": lambda img, low_pct=2, high_pct=98: enhancement.contrast_stretch(
        img, low_pct, high_pct),
    "flatten_background": _flatten_background,
    "threshold_fixed": _threshold_fixed, "threshold_otsu": _threshold_otsu,
    "threshold_adaptive": _threshold_adaptive, "threshold_lines": _threshold_lines,
    "threshold_log": _threshold_log, "link_gaps": _link_gaps, "opening": _opening,
    "closing": _closing, "fill_small_holes": _fill_small_holes,
    "select_objects": _select_objects,
}


def load_config(path):
    """Read a pipeline config: {"name": ..., "stages": [{"stage": name, **params}, ...]}."""
    import yaml  # local import: only config users need PyYAML
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(gray, config):
    """Apply the config's stages in order to a uint8 gray image.

    Returns (final bool mask, steps); each step holds stage, params, output and time_s.
    Unknown stage names raise KeyError; misspelled parameters raise TypeError.
    """
    current, steps = gray, []
    for spec in config["stages"]:
        params = {k: v for k, v in spec.items() if k != "stage"}
        stage = STAGES[spec["stage"]]
        start = time.perf_counter()
        current = stage(current, **params)
        steps.append({"stage": spec["stage"], "params": params, "output": current,
                      "time_s": time.perf_counter() - start})
    if current.dtype != bool:
        raise ValueError(f"pipeline {config['name']!r} must end in a bool mask")
    return current, steps


def evaluate_split(method, names, root, size=None, levels=256):
    """Run `method` on every named image and score it against its mask.

    size:   long side in pixels for image and mask (None = full resolution)
    levels: number of gray levels kept before the method runs (256 = unchanged)
    Returns one dict per image: name, iou, dice, area_pct_pred, area_pct_gt, time_s.
    """
    paths = {img.stem: (img, gt) for img, gt in data.list_pairs(root)}
    rows = []
    for name in names:
        img_path, gt_path = paths[name]
        gray = data.load_image(img_path, size=size)
        gt = data.load_ground_truth(gt_path, size=size)
        if levels != 256:
            gray = data.quantize_gray_levels(gray, levels)
        start = time.perf_counter()
        pred = method(gray)
        rows.append({
            "name": name, "iou": metrics.iou(pred, gt), "dice": metrics.dice(pred, gt),
            "area_pct_pred": 100.0 * pred.mean(), "area_pct_gt": 100.0 * gt.mean(),
            "time_s": time.perf_counter() - start,
        })
    return rows


def main(argv=None):
    """CLI: evaluate one method on a split, save per-image CSV, log the mean IoU."""
    p = argparse.ArgumentParser(description="Evaluate a method on the dev or test split.")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--method", choices=sorted(METHODS))
    which.add_argument("--config", help="pipeline YAML in configs/")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--size", type=int, default=None, help="long side in pixels")
    p.add_argument("--levels", type=int, default=256)
    p.add_argument("--root", default="data/raw/concrete_crack")
    p.add_argument("--split-file", default="data/split.json")
    p.add_argument("--out", required=True, help="per-image CSV to write")
    p.add_argument("--log", default="experiments/experiment_log.csv")
    p.add_argument("--author", default="MEM1")
    p.add_argument("--stage", default="baseline")
    args = p.parse_args(argv)
    if args.split == "test":
        raise SystemExit("Test split is locked until the final config is frozen (T1.20).")

    if args.config:
        config = load_config(args.config)
        args.method = config["name"]
        method = lambda gray: run(gray, config)[0]  # noqa: E731
    else:
        method = METHODS[args.method]
    names = data.load_split(args.split_file)[args.split]
    start = time.perf_counter()
    rows = evaluate_split(method, names, args.root, args.size, args.levels)
    elapsed = time.perf_counter() - start
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ious = np.array([r["iou"] for r in rows])
    params = {"size": args.size, "levels": args.levels}
    log_run(args.log, author=args.author, pipeline=args.method, stage=args.stage,
            method=args.method, params=params, seed=42, split=args.split,
            n_images=len(rows), time_s=round(elapsed, 2), metric="iou_mean",
            value=round(float(ious.mean()), 4), notes=f"std={ious.std():.4f}; out={args.out}")
    print(f"{args.method} size={args.size} levels={args.levels}: IoU {ious.mean():.4f} "
          f"+/- {ious.std():.4f} over {len(rows)} images ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
