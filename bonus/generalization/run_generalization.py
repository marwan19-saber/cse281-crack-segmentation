"""Frozen final pipeline on a second dataset, DeepCrack (MEM1 + MEM6, Bonus 3).

    python -m bonus.generalization.run_generalization

Dataset: DeepCrack (Liu et al., Neurocomputing 2019), github.com/yhlleo/DeepCrack,
"RESTRICTED to non-commercial research and educational purposes". Extract DeepCrack.zip to
data/raw/deepcrack/. Only its 237 test images are used; nothing is tuned: the config is
configs/final_pipeline.yaml unchanged, images at native size (544 x 384), masks = PNG > 127.
Writes bonus/generalization/results.csv (per image) and prints the summary.
"""
import csv
from pathlib import Path

import numpy as np

from src import data, metrics, morphology, pipeline

ROOT = Path("data/raw/deepcrack")


def main():
    """Score the frozen final pipeline and the trivial baseline on DeepCrack test images."""
    config = pipeline.load_config("configs/final_pipeline.yaml")
    trivial = pipeline.load_config("configs/baseline_trivial.yaml")
    rows = []
    for img_path in sorted((ROOT / "test_img").glob("*.jpg")):
        gray = data.load_image(img_path)
        gt = data.load_ground_truth(ROOT / "test_lab" / f"{img_path.stem}.png")
        final, _ = pipeline.run(gray, config)
        base, _ = pipeline.run(gray, trivial)
        rows.append({"image": img_path.stem, "iou_final": metrics.iou(final, gt),
                     "iou_trivial": metrics.iou(base, gt),
                     "dice_final": metrics.dice(final, gt),
                     "area_pct_gt": 100 * gt.mean(),
                     "area_pct_final": morphology.crack_measurements(final)["area_pct"]})
    with open("bonus/generalization/results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("iou_final", "iou_trivial", "dice_final", "area_pct_gt"):
        v = np.array([r[key] for r in rows])
        print(f"{key:14} mean {v.mean():.4f} +/- {v.std():.4f}")
    print(f"{len(rows)} DeepCrack test images")


if __name__ == "__main__":
    main()
