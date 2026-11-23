"""Experiment runner: one-factor-at-a-time studies, ablations and robustness runs (MEM1).

    python -m src.studies --study threshold_method --set tune

Each study is a list of (label, config, noise) variants that differ from a reference
configuration in exactly one factor. Variants run in parallel worker processes; every worker
loads the image set once. Results go to experiments/results/study_<name>.csv and every
variant is appended to experiments/experiment_log.csv.

Image sets (development data only; the test split is never touched here):
    tune: a fixed random 60-image subset of dev (seed 0), used for parameter tuning
    dev:  all 321 dev images, used to compare the finished pipelines and for ablations
"""
import argparse
import copy
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src import data, metrics, pipeline

SIZE = 1008
ROOT = "data/raw/concrete_crack"
SPLIT_FILE = "data/split.json"
RESULTS = "experiments/results"
_IMAGES = []  # filled in each worker by _load


def image_names(which):
    """Names of the 'tune' subset or the full 'dev' split."""
    dev = data.load_split(SPLIT_FILE)["dev"]
    if which == "dev":
        return dev
    return sorted(np.random.default_rng(0).choice(dev, 60, replace=False).tolist())


def _load(names, size=SIZE, root=ROOT):
    """Worker initializer: load gray images and masks once."""
    pairs = {img.stem: (img, gt) for img, gt in data.list_pairs(root)}
    _IMAGES.clear()
    _IMAGES.extend((data.load_image(pairs[n][0], size=size),
                    data.load_ground_truth(pairs[n][1], size=size)) for n in names)


def score(config, images, noise=None):
    """Mean/std IoU and Dice of a config over images; noise = (kind, level, seed) or None."""
    ious, dices, start = [], [], time.perf_counter()
    for gray, gt in images:
        if noise:
            gray = data.add_noise(gray, *noise)
        pred, _ = pipeline.run(gray, config)
        ious.append(metrics.iou(pred, gt))
        dices.append(metrics.dice(pred, gt))
    ious = np.array(ious)
    return {"iou_mean": float(ious.mean()), "iou_std": float(ious.std()),
            "dice_mean": float(np.mean(dices)),
            "time_per_img_s": (time.perf_counter() - start) / len(images)}


def _score_variant(variant):
    """Worker task: score one (label, config, noise) variant on the worker's images."""
    label, config, noise = variant
    return label, score(config, _IMAGES, noise)


def vary(config, stage, **params):
    """Copy of config with the first stage named `stage` given new parameter values."""
    new = copy.deepcopy(config)
    spec = next(s for s in new["stages"] if s["stage"] == stage)
    spec.update(params)
    return new


def replace_stage(config, stage, new_spec):
    """Copy of config with stage `stage` replaced by new_spec (None removes it)."""
    new = copy.deepcopy(config)
    i = next(k for k, s in enumerate(new["stages"]) if s["stage"] == stage)
    if new_spec is None:
        del new["stages"][i]
    else:
        new["stages"][i] = new_spec
    return new


def run_study(name, variants, which="tune", workers=12, author="MEM1"):
    """Score every variant in parallel, write the summary CSV and log each variant."""
    names = image_names(which)
    start = time.perf_counter()
    with ProcessPoolExecutor(min(workers, len(variants)), initializer=_load,
                             initargs=(names,)) as pool:
        results = dict(pool.map(_score_variant, variants))
    rows = []
    for label, config, noise in variants:
        r = results[label]
        rows.append({"study": name, "variant": label, **r, "set": which,
                     "n_images": len(names), "noise": json.dumps(noise),
                     "config": json.dumps(config["stages"])})
        pipeline.log_run("experiments/experiment_log.csv", author=author,
                         pipeline=config["name"], stage=name, method=label,
                         params={"stages": config["stages"], "noise": noise}, seed=0,
                         split=f"dev:{which}", n_images=len(names),
                         time_s=round(r["time_per_img_s"] * len(names), 2), metric="iou_mean",
                         value=round(r["iou_mean"], 4), notes=f"std={r['iou_std']:.4f}")
    path = f"{RESULTS}/study_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] {len(variants)} variants on {len(names)} images "
          f"({time.perf_counter() - start:.0f} s) -> {path}")
    for row in sorted(rows, key=lambda r: -r["iou_mean"]):
        print(f"  {row['variant']:<40} IoU {row['iou_mean']:.4f} +/- {row['iou_std']:.4f}"
              f"  {row['time_per_img_s']:.2f} s/img")
    return rows


NOISES = [("gaussian", 10), ("gaussian", 20), ("gaussian", 30),
          ("salt_pepper", 0.02), ("salt_pepper", 0.05), ("salt_pepper", 0.10),
          ("speckle", 0.1), ("speckle", 0.2), ("speckle", 0.3)]


def _restoration_filters():
    """name -> function(uint8) -> uint8: spatial restoration filters (k = 3) and frequency
    low-pass filters at three cutoffs D0 on the 2x padded grid (MEM4)."""
    from src import frequency as fq
    from src import restoration as rs

    def lowpass(kind, d0):
        """Frequency-domain low-pass of the given kind and cutoff, as an image function."""
        def f(img):
            """Filter one uint8 image on the 2x padded grid."""
            shape = (2 * img.shape[0], 2 * img.shape[1])
            h = {"ideal": fq.ideal_lowpass(shape, d0),
                 "butterworth": fq.butterworth_lowpass(shape, d0, 2),
                 "gaussian": fq.gaussian_lowpass(shape, d0)}[kind]
            return fq.apply_filter(img, h)
        return f
    filters = {"none (noisy input)": lambda x: x,
               "arithmetic mean": lambda x: rs.arithmetic_mean(x, 3),
               "geometric mean": lambda x: rs.geometric_mean(x, 3),
               "harmonic mean": lambda x: rs.harmonic_mean(x, 3),
               "contraharmonic q=+1.5": lambda x: rs.contraharmonic_mean(x, 3, 1.5),
               "contraharmonic q=-1.5": lambda x: rs.contraharmonic_mean(x, 3, -1.5),
               "median": lambda x: rs.median_filter(x, 3),
               "max": lambda x: rs.max_filter(x, 3),
               "min": lambda x: rs.min_filter(x, 3),
               "midpoint": lambda x: rs.midpoint_filter(x, 3),
               "alpha-trimmed d=2": lambda x: rs.alpha_trimmed_mean(x, 3, 2)}
    for kind in ("ideal", "butterworth", "gaussian"):
        for d0 in (100, 200, 400):
            filters[f"{kind} LP D0={d0}"] = lowpass(kind, d0)
    return filters


def _psnr_task(noise):
    """Worker task: MSE/PSNR of every filter for one (kind, level) noise on 20 images."""
    from src import metrics as mt
    kind, level = noise
    out = []
    for name, fn in _restoration_filters().items():
        mse, psnr = [], []
        for seed, (clean, _) in enumerate(_IMAGES[:20]):
            restored = fn(data.add_noise(clean, kind, level, seed))
            mse.append(mt.mse(restored, clean))
            psnr.append(mt.psnr(restored, clean))
        out.append({"noise": kind, "level": level, "filter": name,
                    "mse": float(np.mean(mse)), "psnr_db": float(np.mean(psnr))})
    return out


def run_psnr_study(workers=9):
    """MEM4: restoration filters vs noise type and level, scored against the clean image."""
    names = image_names("tune")[:20]
    with ProcessPoolExecutor(workers, initializer=_load, initargs=(names,)) as pool:
        rows = [r for chunk in pool.map(_psnr_task, NOISES) for r in chunk]
    path = f"{RESULTS}/study_restoration_psnr.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for kind, level in NOISES:
        part = [r for r in rows if r["noise"] == kind and r["level"] == level]
        best = max(part, key=lambda r: r["psnr_db"])
        base = next(r for r in part if r["filter"].startswith("none"))
        print(f"{kind:12} {level:<5} input {base['psnr_db']:.2f} dB | best {best['filter']} "
              f"{best['psnr_db']:.2f} dB")
        pipeline.log_run("experiments/experiment_log.csv", author="MEM4",
                         pipeline="restoration", stage="restoration_psnr",
                         method=best["filter"], params={"noise": kind, "level": level},
                         seed=0, split="dev:tune20", n_images=20, metric="psnr_db_best",
                         value=round(best["psnr_db"], 3),
                         notes=f"input {base['psnr_db']:.2f} dB; full table {path}")
    return rows


def main(argv=None):
    """CLI: run one study defined in src/study_defs.py."""
    from src import study_defs
    p = argparse.ArgumentParser(description="Run a one-factor-at-a-time study.")
    p.add_argument("--study", required=True,
                   choices=sorted(study_defs.STUDIES) + ["restoration_psnr"])
    p.add_argument("--set", default="tune", choices=["tune", "dev"])
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(argv)
    if args.study == "restoration_psnr":
        run_psnr_study()
        return
    author, variants = study_defs.STUDIES[args.study]()
    run_study(args.study, variants, args.set, args.workers, author)


if __name__ == "__main__":
    main()
