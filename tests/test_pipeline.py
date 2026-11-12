"""Tests for src/pipeline.py: baselines and the experiment logger."""
import csv

import numpy as np
import pytest

from src import pipeline


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
