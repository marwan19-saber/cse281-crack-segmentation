"""Binary morphology, hole filling, connected components, measurement and decision (MEM6).

Masks are bool (H, W), True = crack. Border convention matches OpenCV's default: pixels
outside the image count as 1 for erosion and 0 for dilation, so borders never erode away.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.morphology import skeletonize


# ---------------------------------------------------------------- structuring elements
def structuring_element(shape, size, angle=0):
    """Odd size x size bool SE: "square", "disk", "cross" or "line" (angle 0, 45, 90, 135)."""
    if size % 2 == 0:
        raise ValueError("size must be odd")
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    if shape == "square":
        return np.ones((size, size), bool)
    if shape == "disk":
        return x ** 2 + y ** 2 <= r ** 2
    if shape == "cross":
        return (x == 0) | (y == 0)
    if shape == "line":
        return {0: y == 0, 90: x == 0, 45: x == -y, 135: x == y}[angle]
    raise ValueError(f"unknown shape {shape!r}")


# ---------------------------------------------------------------- erosion, dilation
def _footprint_windows(mask, se, outside):
    """Values under the SE footprint for every pixel: shape (H, W, n_footprint)."""
    r, c = se.shape[0] // 2, se.shape[1] // 2
    padded = np.pad(mask, ((r, r), (c, c)), constant_values=outside)
    return sliding_window_view(padded, se.shape)[..., se]


def erode(mask, se):
    """Erosion: a pixel stays 1 only if every pixel under the SE is 1 (min filter)."""
    return _footprint_windows(np.asarray(mask, bool), se, True).all(axis=-1)


def dilate(mask, se):
    """Dilation: a pixel becomes 1 if any pixel under the SE is 1 (max filter)."""
    return _footprint_windows(np.asarray(mask, bool), se, False).any(axis=-1)


def opening(mask, se):
    """Erode then dilate: removes specks and thin spurs smaller than the SE."""
    return dilate(erode(mask, se), se)


def closing(mask, se):
    """Dilate then erode: fills small holes and bridges narrow gaps."""
    return erode(dilate(mask, se), se)


# ---------------------------------------------------------------- hole filling
def fill_holes(mask):
    """Fill background regions not 4-connected to the image border.

    Definition: X0 = background pixels on the border; X(k+1) = dilate(X(k), cross) AND
    background until stable; holes = background never reached (same result as
    scipy.ndimage.binary_fill_holes). Implemented with one 4-connected labelling of the
    background instead of repeated dilation (Bonus 5: same output, far fewer passes).
    """
    mask = np.asarray(mask, bool)
    labels, _ = label(~mask, connectivity=4)
    edge = np.concatenate((labels[[0, -1], :].ravel(), labels[:, [0, -1]].ravel()))
    border = np.unique(edge)
    return mask | (labels > 0) & ~np.isin(labels, border)


# ---------------------------------------------------------------- connected components
def _find(parent, i):
    """Union-find root with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def label(mask, connectivity=8):
    """Connected-component labelling by runs: returns (int32 labels, number of objects).

    Each row's foreground runs get provisional ids; runs in consecutive rows that touch
    (overlap, or overlap after widening by 1 for 8-connectivity) are merged with union-find.
    """
    mask = np.asarray(mask, bool)
    reach = 1 if connectivity == 8 else 0
    runs, rows = [], []                       # (row, start, end_exclusive)
    for r, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        starts, ends = edges[::2], edges[1::2]
        rows.append(range(len(runs), len(runs) + len(starts)))
        runs.extend((r, s, e) for s, e in zip(starts, ends))
    parent = list(range(len(runs)))
    for r in range(1, len(rows)):
        prev, cur = rows[r - 1], rows[r]
        i = prev.start
        for j in cur:
            _, s, e = runs[j]
            while i < prev.stop and runs[i][2] + reach <= s:
                i += 1
            k = i
            while k < prev.stop and runs[k][1] < e + reach:
                a, b = _find(parent, k), _find(parent, j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                k += 1
    labels = np.zeros(mask.shape, np.int32)
    ids = {}
    for j, (r, s, e) in enumerate(runs):
        labels[r, s:e] = ids.setdefault(_find(parent, j), len(ids) + 1)
    return labels, len(ids)


def remove_small_objects(mask, min_area):
    """Drop connected objects with fewer than min_area pixels."""
    labels, _ = label(mask)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


# ---------------------------------------------------------------- measurement and decision
def measure(labels):
    """Per object: area, bbox (r0, c0, r1, c1 exclusive), centroid, perimeter, circularity.

    perimeter = pixels of the object not covered by its 4-neighbour erosion;
    circularity = 4 pi area / perimeter^2 (about 1 for a disk, near 0 for a thin line).
    """
    cross = structuring_element("cross", 3)
    objects = []
    for i in range(1, labels.max() + 1):
        rr, cc = np.nonzero(labels == i)
        if rr.size == 0:
            continue
        r0, c0, r1, c1 = rr.min(), cc.min(), rr.max() + 1, cc.max() + 1
        sub = np.pad(labels[r0:r1, c0:c1] == i, 1)
        perimeter = int((sub & ~erode(sub, cross)).sum())
        objects.append({
            "label": i, "area": int(rr.size), "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "centroid": (float(rr.mean()), float(cc.mean())), "perimeter": perimeter,
            "circularity": 4 * np.pi * rr.size / perimeter ** 2,
        })
    return objects


def select_objects(mask, min_area=0, max_circularity=1.0):
    """Keep objects that are large enough and not round (cracks are long and thin)."""
    labels, n = label(mask)
    # All objects at once: separate components are never 4-adjacent, so each boundary pixel
    # (object pixel not kept by a 4-neighbour erosion) belongs to exactly one object.
    padded = np.pad(np.asarray(mask, bool), 1)
    boundary = (padded & ~erode(padded, structuring_element("cross", 3)))[1:-1, 1:-1]
    area = np.bincount(labels.ravel(), minlength=n + 1)
    perimeter = np.bincount(labels[boundary], minlength=n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = np.where(perimeter > 0, 4 * np.pi * area / perimeter ** 2, np.inf)
    keep = (area >= min_area) & (circularity <= max_circularity)
    keep[0] = False
    return keep[labels]


def crack_measurements(mask):
    """Cracked area %, crack length (skeleton pixels) and number of crack objects."""
    mask = np.asarray(mask, bool)
    return {"area_pct": 100.0 * mask.mean(), "length_px": int(skeletonize(mask).sum()),
            "n_objects": label(mask)[1]}


def severity(area_pct, cutoffs):
    """Severity class from cracked-area %: below cutoffs[0] minor, below cutoffs[1] moderate,
    otherwise severe. Cut-offs come from dev-set ground-truth quantiles."""
    if area_pct < cutoffs[0]:
        return "minor"
    return "moderate" if area_pct < cutoffs[1] else "severe"
