"""Baselines, pipeline integration and experiment logging (MEM1).

Cracks are darker than the concrete, so every threshold here marks pixels BELOW T as crack.
The simple baseline uses OpenCV's Otsu and opening until the team's NumPy versions
(MEM4 segmentation.otsu_threshold, MEM5 morphology.opening) are merged; the swap is logged.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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
