"""Tests for src/pipeline.py: baselines and the experiment logger."""
import csv

import cv2
import numpy as np
import pytest

from src import pipeline


@pytest.fixture
def crack_dataset(tmp_path):
    """Two fake photos whose dark stripe is exactly the crack in their mask."""
    (tmp_path / "rgb").mkdir()
    (tmp_path / "BW").mkdir()
    img = np.full((40, 40, 3), 200, np.uint8)
    img[10:20] = 50
    mask = np.zeros((40, 40), np.uint8)
    mask[10:20] = 255
    for name in ("001", "002"):
        cv2.imwrite(str(tmp_path / "rgb" / f"{name}.png"), img)
        cv2.imwrite(str(tmp_path / "BW" / f"{name}.png"), mask)
    return tmp_path


def test_evaluate_split_scores_each_image(crack_dataset):
    rows = pipeline.evaluate_split(pipeline.baseline_trivial, ["001", "002"], crack_dataset)
    assert [r["name"] for r in rows] == ["001", "002"]
    assert all(r["iou"] == 1.0 and r["dice"] == 1.0 for r in rows)
    assert rows[0]["area_pct_gt"] == pytest.approx(25.0)
    assert rows[0]["area_pct_pred"] == pytest.approx(25.0)
    assert rows[0]["time_s"] >= 0


def test_evaluate_split_applies_gray_level_reduction(crack_dataset):
    # 2 levels map 50 -> 0 and 200 -> 255, so the dark stripe is still found exactly
    rows = pipeline.evaluate_split(pipeline.baseline_trivial, ["001"], crack_dataset, levels=2)
    assert rows[0]["iou"] == 1.0


def test_evaluate_split_resizes_image_and_mask_together(crack_dataset):
    rows = pipeline.evaluate_split(pipeline.baseline_trivial, ["001"], crack_dataset, size=20)
    assert rows[0]["iou"] == 1.0 and rows[0]["area_pct_gt"] == pytest.approx(25.0)


def crack_image():
    """Bright concrete (200) with a dark 3-pixel-wide crack (50) and one dark speck."""
    img = np.full((30, 30), 200, np.uint8)
    img[14:17, 2:28] = 50
    img[5, 5] = 50
    return img


def test_trivial_baseline_marks_dark_pixels_as_crack():
    mask = pipeline.baseline_trivial(crack_image())
    assert mask.dtype == bool
    assert mask[15, 10] and not mask[0, 0]
    assert mask[5, 5]                       # no clean-up in the trivial baseline


def test_trivial_baseline_threshold_is_strict():
    img = np.array([[126, 127, 128]], np.uint8)
    assert pipeline.baseline_trivial(img).tolist() == [[True, False, False]]


def test_simple_baseline_keeps_crack_and_removes_speck():
    mask = pipeline.baseline_simple(crack_image())
    assert mask.dtype == bool
    assert mask[15, 10]                     # 3-wide crack survives a 3x3 opening
    assert not mask[5, 5]                   # single dark pixel is removed
    assert not mask[0, 0]


def test_log_run_appends_rows_with_fixed_columns(tmp_path):
    log = tmp_path / "log.csv"
    pipeline.log_run(log, author="MEM1", pipeline="baseline_trivial", stage="threshold",
                     method="fixed", params={"t": 127}, seed=0, split="dev", n_images=3,
                     time_s=0.5, metric="iou", value=0.42)
    pipeline.log_run(log, author="MEM1", pipeline="baseline_simple", metric="iou", value=0.1)
    rows = list(csv.DictReader(log.read_text(encoding="utf-8").splitlines()))
    assert list(rows[0]) == pipeline.LOG_COLUMNS
    assert [r["pipeline"] for r in rows] == ["baseline_trivial", "baseline_simple"]
    assert rows[0]["params_json"] == '{"t": 127}'
    assert rows[0]["run_id"] != rows[1]["run_id"]


def test_log_run_rejects_unknown_field(tmp_path):
    with pytest.raises(TypeError):
        pipeline.log_run(tmp_path / "log.csv", autor="typo")


# ---------------------------------------------------------------- config + stage registry
def test_run_applies_stages_in_order_and_keeps_intermediates():
    config = {"name": "demo", "stages": [
        {"stage": "gaussian", "ksize": 3},
        {"stage": "threshold_fixed", "t": 127},
        {"stage": "opening", "size": 3},
    ]}
    mask, steps = pipeline.run(crack_image(), config)
    assert mask.dtype == bool and mask[15, 10] and not mask[5, 5]
    assert [s["stage"] for s in steps] == ["gaussian", "threshold_fixed", "opening"]
    assert steps[0]["output"].dtype == np.uint8 and steps[-1]["output"].dtype == bool
    assert all(s["time_s"] >= 0 for s in steps)


def test_run_rejects_unknown_stage():
    with pytest.raises(KeyError):
        pipeline.run(crack_image(), {"name": "x", "stages": [{"stage": "magic"}]})


def test_run_rejects_misspelled_parameter():
    with pytest.raises(TypeError):
        pipeline.run(crack_image(), {"name": "x", "stages": [{"stage": "gaussian", "ksz": 3}]})


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("name: p\nstages:\n  - stage: threshold_otsu\n", encoding="utf-8")
    assert pipeline.load_config(path) == {"name": "p", "stages": [{"stage": "threshold_otsu"}]}


def test_shipped_configs_run_end_to_end():
    from pathlib import Path
    for path in sorted(Path("configs").glob("*.yaml")):
        mask, _ = pipeline.run(crack_image(), pipeline.load_config(path))
        assert mask.dtype == bool and mask.shape == (30, 30), path
