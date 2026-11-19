"""Per-stage timing of the final pipeline on 20 dev images (MEM1 coordinates, Bonus 5).

    python -m bonus.optimization.profile_pipeline --label before
    python -m bonus.optimization.profile_pipeline --label after

Appends one row per stage to bonus/optimization/timings.csv and saves the masks of the run
to out/profile_<label>.npz so 'after' can be checked to be identical to 'before'.
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import data, pipeline


def main(argv=None):
    """Time every stage, store masks, compare with the 'before' masks when available."""
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    label = p.parse_args(argv).label
    config = pipeline.load_config("configs/final_pipeline.yaml")
    pairs = {img.stem: img for img, _ in data.list_pairs("data/raw/concrete_crack")}
    names = data.load_split("data/split.json")["dev"][:20]
    totals, masks = defaultdict(float), {}
    for name in names:
        gray = data.load_image(pairs[name], size=1008)
        start = time.perf_counter()
        mask, steps = pipeline.run(gray, config)
        totals["TOTAL (per image)"] += time.perf_counter() - start
        for s in steps:
            totals[s["stage"]] += s["time_s"]
        masks[name] = mask
    Path("out").mkdir(exist_ok=True)
    np.savez_compressed(f"out/profile_{label}.npz", **masks)
    same = ""
    if label != "before" and Path("out/profile_before.npz").exists():
        before = np.load("out/profile_before.npz")
        same = all(np.array_equal(before[n], masks[n]) for n in names)
    path = Path("bonus/optimization/timings.csv")
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "stage", "seconds_per_image", "masks_identical_to_before"])
        for stage, t in totals.items():
            w.writerow([label, stage, round(t / len(names), 4), same])
            print(f"{label:7} {stage:20} {t / len(names):.4f} s/img")
    print("masks identical to 'before':", same)


if __name__ == "__main__":
    main()
