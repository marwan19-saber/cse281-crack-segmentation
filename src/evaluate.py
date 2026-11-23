"""Per-image evaluation of a pipeline config, worst cases and failure panels (MEM6).

    python -m src.evaluate --config configs/pipeline_c.yaml --split dev --out res.csv

The test split is refused until the git tag `results-frozen` exists (test-set discipline).
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np

from src import data, metrics, morphology, pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

WORKING_SIZE = 1008


def results_frozen():
    """True when the git tag `results-frozen` exists (final config chosen on dev)."""
    tags = subprocess.run(["git", "tag", "-l", "results-frozen"], capture_output=True,
                          text=True, check=False).stdout
    return bool(tags.strip())


def evaluate(config, split="dev", root="data/raw/concrete_crack",
             split_file="data/split.json", size=WORKING_SIZE):
    """Run a config on every image of a split; one row per image with metrics,
    measurements (predicted vs ground truth) and processing time."""
    if split == "test" and not results_frozen():
        raise SystemExit("Test split locked: tag the frozen final config as results-frozen.")
    paths = {img.stem: (img, gt) for img, gt in data.list_pairs(root)}
    rows = []
    for name in data.load_split(split_file)[split]:
        gray = data.load_image(paths[name][0], size=size)
        gt = data.load_ground_truth(paths[name][1], size=size)
        start = time.perf_counter()
        pred, _ = pipeline.run(gray, config)
        elapsed = time.perf_counter() - start
        m_pred, m_gt = morphology.crack_measurements(pred), morphology.crack_measurements(gt)
        rows.append({
            "name": name, "iou": metrics.iou(pred, gt), "dice": metrics.dice(pred, gt),
            "area_pct_pred": m_pred["area_pct"], "area_pct_gt": m_gt["area_pct"],
            "length_pred": m_pred["length_px"], "length_gt": m_gt["length_px"],
            "objects_pred": m_pred["n_objects"], "objects_gt": m_gt["n_objects"],
            "time_s": elapsed,
        })
    return rows


def worst_k(rows, k=20):
    """The k rows with the lowest IoU (worst first)."""
    return sorted(rows, key=lambda r: r["iou"])[:k]


def save_failure_panels(rows, config, out_dir, k=20, root="data/raw/concrete_crack",
                        size=WORKING_SIZE):
    """For the k worst images save one figure: input, every stage output, ground truth."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {img.stem: (img, gt) for img, gt in data.list_pairs(root)}
    saved = []
    for row in worst_k(rows, k):
        gray = data.load_image(paths[row["name"]][0], size=size)
        gt = data.load_ground_truth(paths[row["name"]][1], size=size)
        _, steps = pipeline.run(gray, config)
        panels = [("input", gray)] + [(s["stage"], s["output"]) for s in steps]
        panels.append(("ground truth", gt))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
        for ax, (title, img) in zip(np.atleast_1d(axes), panels):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{config['name']} | image {row['name']} | IoU {row['iou']:.3f}")
        path = out_dir / f"{config['name']}_{row['name']}.png"
        fig.savefig(path, dpi=70, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def write_csv(rows, path):
    """Write per-image rows to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    """CLI: evaluate a config, write per-image CSV, optional failure panels, log the mean."""
    p = argparse.ArgumentParser(description="Evaluate a pipeline config on a split.")
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--size", type=int, default=WORKING_SIZE)
    p.add_argument("--out", required=True)
    p.add_argument("--panels", default=None, help="folder for the 20 worst failure panels")
    p.add_argument("--author", default="MEM6")
    p.add_argument("--stage", default="evaluation")
    p.add_argument("--log", default="experiments/experiment_log.csv")
    args = p.parse_args(argv)
    config = pipeline.load_config(args.config)
    start = time.perf_counter()
    rows = evaluate(config, args.split, size=args.size)
    elapsed = time.perf_counter() - start
    write_csv(rows, args.out)
    if args.panels:
        save_failure_panels(rows, config, args.panels, size=args.size)
    iou = np.array([r["iou"] for r in rows])
    pipeline.log_run(args.log, author=args.author, pipeline=config["name"], stage=args.stage,
                     method=config["name"], params={"size": args.size, "config": args.config},
                     seed=42, split=args.split, n_images=len(rows),
                     time_s=round(elapsed, 2), metric="iou_mean",
                     value=round(float(iou.mean()), 4),
                     notes=f"std={iou.std():.4f}; out={args.out}")
    print(f"{config['name']} on {args.split}: IoU {iou.mean():.4f} +/- {iou.std():.4f} "
          f"over {len(rows)} images ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
