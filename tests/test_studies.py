"""Tests for src/studies.py helpers (MEM1)."""
import numpy as np

from src import studies

BASE = {"name": "b", "stages": [{"stage": "median", "ksize": 3},
                                {"stage": "threshold_fixed", "t": 127}]}


def test_vary_changes_one_parameter_and_leaves_base_untouched():
    new = studies.vary(BASE, "median", ksize=5)
    assert new["stages"][0] == {"stage": "median", "ksize": 5}
    assert BASE["stages"][0]["ksize"] == 3


def test_replace_and_remove_stage():
    swapped = studies.replace_stage(BASE, "median", {"stage": "gaussian", "ksize": 5})
    assert swapped["stages"][0]["stage"] == "gaussian"
    removed = studies.replace_stage(BASE, "median", None)
    assert [s["stage"] for s in removed["stages"]] == ["threshold_fixed"]


def test_score_perfect_prediction():
    gray = np.full((20, 20), 200, np.uint8); gray[5:8] = 40
    gt = gray < 127
    result = studies.score(BASE, [(gray, gt)])
    assert result["iou_mean"] == 1.0 and result["dice_mean"] == 1.0


def test_score_with_noise_is_reproducible():
    gray = np.full((20, 20), 200, np.uint8); gray[5:8] = 40
    gt = gray < 127
    a = studies.score(BASE, [(gray, gt)], noise=("gaussian", 30, 1))
    b = studies.score(BASE, [(gray, gt)], noise=("gaussian", 30, 1))
    assert a["iou_mean"] == b["iou_mean"]


def test_tune_subset_is_fixed_and_inside_dev():
    names = studies.image_names("tune")
    assert names == studies.image_names("tune") and len(names) == 60
    assert set(names) <= set(studies.image_names("dev"))
