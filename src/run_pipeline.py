"""Single entry point: raw image file in, crack mask + measurement + decision out (MEM6).

    python -m src.run_pipeline --image photo.jpg --out out/   (config: --config, default final)

Writes <name>_mask.png, <name>_overlay.png (cracks in red) and <name>_result.json with the
cracked-area %, crack length, number of objects, severity class and time per stage.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src import data, morphology, pipeline

DEFAULT_SIZE = 1008
DEFAULT_CUTOFFS = (1.0, 2.0)  # area-% cut-offs; final values come from the config


def process(image_path, config, size=DEFAULT_SIZE, cutoffs=None):
    """Run the configured pipeline on one image file and measure the result."""
    start = time.perf_counter()
    gray = data.load_image(image_path, size=size)
    mask, steps = pipeline.run(gray, config)
    measurements = morphology.crack_measurements(mask)
    cutoffs = cutoffs or tuple(config.get("severity_cutoffs", DEFAULT_CUTOFFS))
    return {
        "image": str(image_path), "gray": gray, "mask": mask, "steps": steps,
        "measurements": measurements,
        "severity": morphology.severity(measurements["area_pct"], cutoffs),
        "time_s": time.perf_counter() - start,
    }


def overlay(gray, mask):
    """RGB image with crack pixels painted red."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[mask] = (255, 0, 0)
    return rgb


def main(argv=None):
    """CLI wrapper around process()."""
    p = argparse.ArgumentParser(description="Segment and measure cracks in one image.")
    p.add_argument("--image", required=True)
    p.add_argument("--config", default="configs/final_pipeline.yaml")
    p.add_argument("--out", default="out")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="long side; 0 = full size")
    args = p.parse_args(argv)
    config = pipeline.load_config(args.config)
    result = process(args.image, config, size=args.size or None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"{stem}_mask.png"), result["mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(out / f"{stem}_overlay.png"),
                cv2.cvtColor(overlay(result["gray"], result["mask"]), cv2.COLOR_RGB2BGR))
    summary = {
        "image": result["image"], "config": config["name"], **result["measurements"],
        "severity": result["severity"], "time_s": round(result["time_s"], 4),
        "stage_times_s": {s["stage"]: round(s["time_s"], 4) for s in result["steps"]},
    }
    (out / f"{stem}_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
