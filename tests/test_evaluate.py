"""Tests for src/evaluate.py and src/run_pipeline.py (MEM5)."""
import json

import cv2
import numpy as np
import pytest

from src import evaluate, run_pipeline

CONFIG = {"name": "t", "stages": [{"stage": "threshold_fixed", "t": 127}]}


@pytest.fixture
def dataset(tmp_path):
    """Two fake photos with a dark stripe; image 002's mask misses half of its stripe."""
    root = tmp_path / "ds"
    (root / "rgb").mkdir(parents=True)
    (root / "BW").mkdir()
    img = np.full((40, 40, 3), 200, np.uint8); img[10:20] = 50
    full = np.zeros((40, 40), np.uint8); full[10:20] = 255
    half = np.zeros((40, 40), np.uint8); half[10:15] = 255
    for name, mask in (("001", full), ("002", half)):
        cv2.imwrite(str(root / "rgb" / f"{name}.png"), img)
        cv2.imwrite(str(root / "BW" / f"{name}.png"), mask)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["001", "002"], "test": []}), encoding="utf-8")
    return root, split


def test_evaluate_reports_metrics_and_measurements(dataset):
    root, split = dataset
    rows = evaluate.evaluate(CONFIG, "dev", root=root, split_file=split, size=None)
    by = {r["name"]: r for r in rows}
    assert by["001"]["iou"] == 1.0 and by["002"]["iou"] == pytest.approx(0.5)
    assert by["001"]["area_pct_pred"] == pytest.approx(25.0)
    assert by["001"]["length_pred"] > 0 and by["001"]["time_s"] >= 0


def test_worst_k_orders_by_iou(dataset):
    root, split = dataset
    rows = evaluate.evaluate(CONFIG, "dev", root=root, split_file=split, size=None)
    assert [r["name"] for r in evaluate.worst_k(rows, 1)] == ["002"]


def test_test_split_is_locked_without_frozen_tag(dataset, monkeypatch):
    root, split = dataset
    monkeypatch.setattr(evaluate, "results_frozen", lambda: False)
    with pytest.raises(SystemExit):
        evaluate.evaluate(CONFIG, "test", root=root, split_file=split)


def test_failure_panel_is_written(dataset, tmp_path):
    root, split = dataset
    rows = evaluate.evaluate(CONFIG, "dev", root=root, split_file=split, size=None)
    paths = evaluate.save_failure_panels(rows, CONFIG, tmp_path / "fail", k=1, root=root,
                                         size=None)
    assert len(paths) == 1 and paths[0].exists()


def test_run_pipeline_process_returns_mask_measurement_and_timing(dataset):
    root, _ = dataset
    out = run_pipeline.process(root / "rgb" / "001.png", CONFIG, size=None, cutoffs=(1, 30))
    assert out["mask"].dtype == bool and out["mask"].sum() == 400
    assert out["measurements"]["area_pct"] == pytest.approx(25.0)
    assert out["severity"] == "moderate"
    assert [s["stage"] for s in out["steps"]] == ["threshold_fixed"]


def test_run_pipeline_cli_writes_outputs(dataset, tmp_path):
    root, _ = dataset
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: t\nstages:\n  - stage: threshold_fixed\n    t: 127\n",
                   encoding="utf-8")
    run_pipeline.main(["--image", str(root / "rgb" / "001.png"), "--config", str(cfg),
                       "--out", str(tmp_path / "o"), "--size", "0"])
    result = json.loads((tmp_path / "o" / "001_result.json").read_text(encoding="utf-8"))
    assert result["severity"] in ("minor", "moderate", "severe")
    assert (tmp_path / "o" / "001_mask.png").exists()
    assert (tmp_path / "o" / "001_overlay.png").exists()


def test_run_pipeline_bad_image_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline.process(tmp_path / "missing.jpg", CONFIG)
