"""Tests for src/data.py: loading, pairing, split, degradation, resolution and gray levels."""
import cv2
import numpy as np
import pytest

from src import data


@pytest.fixture
def dataset(tmp_path):
    """A tiny fake dataset laid out like data/raw/concrete_crack/{rgb,BW}."""
    (tmp_path / "rgb").mkdir()
    (tmp_path / "BW").mkdir()
    img = np.zeros((40, 80, 3), np.uint8)
    img[..., 2] = 200          # red channel in OpenCV's BGR order
    mask = np.zeros((40, 80, 3), np.uint8)
    mask[10:20, :] = 255
    for name, ext in (("001", ".jpg"), ("002", ".JPG"), ("015", ".jpg")):
        cv2.imwrite(str(tmp_path / "rgb" / f"{name}{ext}"), img)
        cv2.imwrite(str(tmp_path / "BW" / f"{name}.png"), mask)
    return tmp_path


# ---------------------------------------------------------------- loading
def test_load_image_gray_is_uint8_2d(dataset):
    img = data.load_image(dataset / "rgb" / "001.jpg")
    assert img.dtype == np.uint8 and img.shape == (40, 80)


def test_load_image_rgb_converts_from_bgr(dataset):
    img = data.load_image(dataset / "rgb" / "001.jpg", mode="rgb")
    assert img.shape == (40, 80, 3)
    assert img[0, 0, 0] > 150 and img[0, 0, 2] < 50     # red ends up in channel 0


def test_load_image_resizes_long_side(dataset):
    img = data.load_image(dataset / "rgb" / "001.jpg", size=40)
    assert img.shape == (20, 40)


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_image(tmp_path / "nope.jpg")


def test_load_ground_truth_is_bool_and_binarized(dataset):
    gt = data.load_ground_truth(dataset / "BW" / "001.png")
    assert gt.dtype == bool and gt.shape == (40, 80)
    assert gt[15, 5] and not gt[30, 5]
    assert gt.sum() == 10 * 80


def test_load_ground_truth_resized_matches_image_size(dataset):
    gt = data.load_ground_truth(dataset / "BW" / "001.png", size=40)
    assert gt.shape == (20, 40) and gt.dtype == bool


# ---------------------------------------------------------------- pairing
def test_list_pairs_matches_by_stem_ignoring_extension_case(dataset):
    pairs = data.list_pairs(dataset)
    assert [p[0].stem for p in pairs] == ["001", "002", "015"]
    assert all(img.stem == gt.stem for img, gt in pairs)


def test_list_pairs_unpaired_image_raises(dataset):
    (dataset / "BW" / "015.png").unlink()
    with pytest.raises(ValueError):
        data.list_pairs(dataset)


# ---------------------------------------------------------------- split
NAMES = [f"{i:03d}" for i in range(1, 201)]


def test_split_covers_everything_without_overlap():
    split = data.make_split(NAMES, dev_frac=0.7, seed=42)
    assert set(split["dev"]).isdisjoint(split["test"])
    assert sorted(split["dev"] + split["test"]) == NAMES


def test_split_is_deterministic():
    assert data.make_split(NAMES, seed=42) == data.make_split(NAMES, seed=42)


def test_split_keeps_id_blocks_together():
    split = data.make_split(NAMES, dev_frac=0.7, seed=42)
    dev_groups = {int(n) // 10 for n in split["dev"]}
    test_groups = {int(n) // 10 for n in split["test"]}
    assert dev_groups.isdisjoint(test_groups)


def test_split_test_fraction_is_close_to_target():
    split = data.make_split(NAMES, dev_frac=0.7, seed=42)
    assert 0.25 <= len(split["test"]) / len(NAMES) <= 0.35


def test_split_roundtrip_json(tmp_path):
    split = data.make_split(NAMES, seed=1)
    path = tmp_path / "split.json"
    data.save_split(split, path)
    assert data.load_split(path) == split


# ---------------------------------------------------------------- degradation
FLAT = np.full((200, 200), 128, np.uint8)


@pytest.mark.parametrize("kind", ["gaussian", "salt_pepper", "speckle", "poisson"])
def test_add_noise_is_reproducible_and_uint8(kind):
    level = {"gaussian": 20, "salt_pepper": 0.05, "speckle": 0.1, "poisson": 1}[kind]
    a = data.add_noise(FLAT, kind, level, seed=7)
    b = data.add_noise(FLAT, kind, level, seed=7)
    assert a.dtype == np.uint8 and a.shape == FLAT.shape
    assert np.array_equal(a, b)
    assert not np.array_equal(a, data.add_noise(FLAT, kind, level, seed=8))


def test_gaussian_noise_has_requested_sigma():
    noisy = data.add_noise(FLAT, "gaussian", 20, seed=0).astype(float)
    assert noisy.std() == pytest.approx(20, rel=0.05)
    assert noisy.mean() == pytest.approx(128, abs=1)


def test_salt_pepper_fraction_and_values():
    noisy = data.add_noise(FLAT, "salt_pepper", 0.1, seed=0)
    changed = noisy != 128
    assert changed.mean() == pytest.approx(0.1, abs=0.01)
    assert set(np.unique(noisy[changed])) == {0, 255}


def test_unknown_noise_kind_raises():
    with pytest.raises(ValueError):
        data.add_noise(FLAT, "blur", 1, seed=0)


# ---------------------------------------------------------------- resolution / gray levels
def test_reduce_resolution_hand_example():
    # each 2x2 block is averaged: (0+2+4+6)/4 = 3, (8+10+12+14)/4 = 11
    x = np.array([[0, 2, 8, 10], [4, 6, 12, 14]], np.uint8)
    assert np.array_equal(data.reduce_resolution(x, 2), np.array([[3, 11]], np.uint8))


def test_reduce_resolution_matches_cv2_inter_area():
    x = np.random.default_rng(0).integers(0, 256, (64, 96), dtype=np.uint8)
    ours = data.reduce_resolution(x, 4)
    ref = cv2.resize(x, (24, 16), interpolation=cv2.INTER_AREA)
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_quantize_gray_levels_hand_example():
    # 4 levels: bins of width 64 map to 0, 85, 170, 255
    x = np.array([[0, 63, 64, 127], [128, 191, 192, 255]], np.uint8)
    expected = np.array([[0, 0, 85, 85], [170, 170, 255, 255]], np.uint8)
    assert np.array_equal(data.quantize_gray_levels(x, 4), expected)


def test_quantize_256_levels_is_identity():
    x = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(data.quantize_gray_levels(x, 256), x)
