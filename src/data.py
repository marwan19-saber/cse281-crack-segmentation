"""Data loading, dev/test split, controlled degradation and resolution study helpers (MEM1).

Conventions used by the whole project:
    images between stages: np.uint8, values 0-255, shape (H, W)
    masks:                 bool, shape (H, W), True = crack
This is the only module that reads image files. No other stage reloads images its own way.
"""
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = "rgb"
MASK_DIR = "BW"


def _resize_long_side(img, size, interpolation=cv2.INTER_AREA):
    """Resize so that the longer side equals `size` pixels (aspect ratio kept)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def _read(path, flag):
    """Read an image file with OpenCV; raise instead of returning None."""
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def load_image(path, mode="gray", size=None):
    """Load one photo as uint8.

    mode "gray" returns (H, W); mode "rgb" returns (H, W, 3) in RGB order (OpenCV reads BGR).
    size, if given, resizes so the long side has that many pixels (INTER_AREA).
    """
    if mode not in ("gray", "rgb"):
        raise ValueError(f"mode must be 'gray' or 'rgb', got {mode!r}")
    img = _read(path, cv2.IMREAD_COLOR)
    code = cv2.COLOR_BGR2GRAY if mode == "gray" else cv2.COLOR_BGR2RGB
    img = cv2.cvtColor(img, code)
    if size is not None:
        img = _resize_long_side(img, size)
    return img


def load_ground_truth(path, size=None):
    """Load a crack mask as bool (True = crack).

    The dataset stores masks as JPEG, so values near 0/255 are blurred: threshold at 127.
    When resizing, the 0/1 mask is area-averaged and kept where at least half is crack.
    """
    mask = _read(path, cv2.IMREAD_GRAYSCALE) > 127
    if size is not None:
        mask = _resize_long_side(mask.astype(np.float32), size) >= 0.5
    return mask


def list_pairs(root):
    """Return sorted (image_path, mask_path) pairs from root/rgb and root/BW, by stem."""
    root = Path(root)
    images = {p.stem: p for p in (root / IMAGE_DIR).iterdir() if p.is_file()}
    masks = {p.stem: p for p in (root / MASK_DIR).iterdir() if p.is_file()}
    unpaired = sorted(set(images) ^ set(masks))
    if unpaired:
        raise ValueError(f"images without a mask or masks without an image: {unpaired[:10]}")
    return [(images[k], masks[k]) for k in sorted(images)]


def make_split(names, dev_frac=0.7, seed=42, block=10):
    """Split image names into dev and test, keeping blocks of consecutive IDs together.

    Neighbouring IDs often show the same wall, so whole blocks (ID // block) go to one side;
    otherwise near-duplicate photos would leak from dev into test.
    """
    groups = {}
    for name in names:
        groups.setdefault(int(name) // block, []).append(name)
    order = sorted(groups)
    np.random.default_rng(seed).shuffle(order)
    target_test = round(len(names) * (1 - dev_frac))
    test = []
    for g in order:
        if len(test) >= target_test:
            break
        test.extend(groups[g])
    test_set = set(test)
    return {"dev": sorted(n for n in names if n not in test_set), "test": sorted(test)}


def save_split(split, path):
    """Write the split to a JSON file (committed to git so everyone uses the same split)."""
    Path(path).write_text(json.dumps(split, indent=1), encoding="utf-8")


def load_split(path):
    """Read a split written by save_split."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_noise(img, kind, level, seed):
    """Return a degraded uint8 copy of img with a known noise model.

    kind "gaussian":    additive N(0, level^2), level = sigma in gray levels
    kind "salt_pepper": a fraction `level` of pixels set to 0 or 255 (half each)
    kind "speckle":     multiplicative, f + f * n with n ~ N(0, level^2)
    kind "poisson":     Poisson(f * level) / level; larger level = weaker noise
    """
    rng = np.random.default_rng(seed)
    f = img.astype(np.float64)
    if kind == "gaussian":
        out = f + rng.normal(0.0, level, f.shape)
    elif kind == "salt_pepper":
        u = rng.random(f.shape)
        out = f.copy()
        out[u < level / 2] = 0
        out[u > 1 - level / 2] = 255
    elif kind == "speckle":
        out = f + f * rng.normal(0.0, level, f.shape)
    elif kind == "poisson":
        out = rng.poisson(f * level) / level
    else:
        raise ValueError(f"unknown noise kind: {kind!r}")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def reduce_resolution(img, factor):
    """Downsample by an integer factor, averaging each factor x factor block (NumPy).

    Edge rows/columns that do not fill a whole block are dropped.
    """
    h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
    blocks = img[:h, :w].astype(np.float64).reshape(h // factor, factor, w // factor, factor)
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


def quantize_gray_levels(img, levels):
    """Reduce a uint8 image to `levels` equally spaced gray levels spread over 0-255."""
    if not 2 <= levels <= 256:
        raise ValueError("levels must be between 2 and 256")
    bins = img.astype(np.int64) * levels // 256
    return np.rint(bins * 255.0 / (levels - 1)).astype(np.uint8)
