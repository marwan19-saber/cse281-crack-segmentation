"""Gradio demo: upload a concrete photo, see every processing stage, the crack mask and the
measurement (MEM6).

    python app/app.py            # then open the printed local URL

Uses configs/final_pipeline.yaml (falls back to configs/pipeline_c.yaml before it exists).
Classical image processing only: no learned model is involved.
"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline, run_pipeline  # noqa: E402  (needs ROOT on sys.path)

CONFIG_PATH = ROOT / "configs" / "final_pipeline.yaml"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT / "configs" / "pipeline_c.yaml"
CONFIG = pipeline.load_config(CONFIG_PATH)
EXAMPLES = sorted(str(p) for p in (ROOT / "app" / "examples").glob("*.jpg"))


def _viewable(img):
    """Stage outputs are uint8 images or bool masks; show masks as black/white."""
    return img.astype(np.uint8) * 255 if img.dtype == bool else img


def analyse(image_path):
    """Run the pipeline on an uploaded file and return (overlay, stages, summary)."""
    if image_path is None:
        raise gr.Error("Please upload a photo first (JPEG or PNG).")
    try:
        result = run_pipeline.process(image_path, CONFIG)
    except (FileNotFoundError, ValueError) as err:
        raise gr.Error(f"Could not read this file as an image: {err}") from err
    stages = [(result["gray"], "input (gray, long side 1008 px)")] + [
        (_viewable(s["output"]), f"{s['stage']} ({s['time_s'] * 1000:.0f} ms)")
        for s in result["steps"]]
    m = result["measurements"]
    summary = {
        "severity": result["severity"],
        "cracked_area_percent": round(float(m["area_pct"]), 3),
        "crack_length_pixels": m["length_px"],
        "crack_objects": m["n_objects"],
        "processing_time_s": round(result["time_s"], 3),
        "pipeline": CONFIG["name"],
    }
    return run_pipeline.overlay(result["gray"], result["mask"]), stages, summary


with gr.Blocks(title="Concrete crack measurement") as demo:
    gr.Markdown("# Concrete crack segmentation and measurement\n"
                "Upload a photo of a concrete surface. The classical pipeline (smoothing, "
                "illumination flattening, adaptive threshold, edge linking, morphology) marks "
                "the cracks in red and reports the cracked area, length and a severity class. "
                "Course project, not a certified inspection tool.")
    with gr.Row():
        inp = gr.Image(type="filepath", label="Concrete photo (JPEG or PNG)")
        out = gr.Image(label="Detected cracks (red overlay)")
    run = gr.Button("Analyse", variant="primary")
    summary = gr.JSON(label="Measurement and decision")
    stages = gr.Gallery(label="Every processing stage, in order", columns=4, height="auto")
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp, label="Example photos (dataset test images)")
    run.click(analyse, inputs=inp, outputs=[out, stages, summary])

if __name__ == "__main__":
    demo.launch()
