"""Baselines, pipeline integration and experiment logging (MEM1).

Cracks are darker than the concrete, so every threshold here marks pixels BELOW T as crack.
The simple baseline uses OpenCV's Otsu and opening until the team's NumPy versions
(MEM4 segmentation.otsu_threshold, MEM5 morphology.opening) are merged; the swap is logged.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src import data, metrics

METHODS = {}  # name -> function(gray uint8) -> bool mask; filled below

LOG_COLUMNS = ["run_id", "date", "author", "pipeline", "stage", "method", "params_json",
               "seed", "split", "n_images", "time_s", "metric", "value", "notes"]


def baseline_trivial(gray):
    """Spec trivial baseline: fixed threshold 127 on the raw gray image (crack = dark)."""
    return np.asarray(gray) < 127


def baseline_simple(gray):
    """Spec simple baseline: inverted Otsu threshold, then one 3x3 opening."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return opened > 0


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
# Library stand-ins: each stage calls OpenCV/scikit-image until its owner's NumPy version is
# merged, then the registry entry is switched (and the switch logged). Owner in brackets.
def _contrast_stretch(img, low_pct=2, high_pct=98):
    """[MEM2 stand-in] Stretch the low_pct..high_pct percentile range to 0-255."""
    lo, hi = np.percentile(img, (low_pct, high_pct))
    if hi <= lo:
        return img.copy()
    return np.clip((img.astype(np.float64) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _gaussian(img, ksize=5, sigma=0):
    """[MEM2 stand-in] Gaussian smoothing (sigma 0 = OpenCV's rule from ksize)."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)


def _median(img, ksize=3):
    """[MEM3 stand-in] Median filter."""
    return cv2.medianBlur(img, ksize)


def _threshold_fixed(img, t=127):
    """[MEM4 stand-in] Crack = pixels darker than t."""
    return img < t


def _threshold_otsu(img):
    """[MEM4 stand-in] Inverted Otsu: crack = pixels at or below Otsu's threshold."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _opening(mask, size=3):
    """[MEM5 stand-in] Opening with a size x size square structuring element."""
    se = np.ones((size, size), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, se) > 0


STAGES = {
    "contrast_stretch": _contrast_stretch, "gaussian": _gaussian, "median": _median,
    "threshold_fixed": _threshold_fixed, "threshold_otsu": _threshold_otsu,
    "opening": _opening,
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
