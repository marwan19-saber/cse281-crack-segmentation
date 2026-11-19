"""Study definitions: one factor changed per variant against a reference config (MEM1).

Each entry returns (author, variants); a variant is (label, config, noise_or_None).
Reference = configs/pipeline_c.yaml unless stated. Owners follow the execution plan.
"""
from src import pipeline
from src.studies import replace_stage, vary

ADAPTIVE = {"stage": "threshold_adaptive", "block": 31, "c": 10, "method": "mean"}


def _ref():
    """Reference configuration for one-factor-at-a-time tuning."""
    return pipeline.load_config("configs/pipeline_c.yaml")


def threshold_method():
    """MEM4: threshold method after the same noise removal and illumination flattening."""
    base = {"name": "thr", "stages": [{"stage": "median", "ksize": 3},
                                      {"stage": "flatten_background", "d0": 6.0}]}
    options = {
        "fixed t=100": {"stage": "threshold_fixed", "t": 100},
        "fixed t=115": {"stage": "threshold_fixed", "t": 115},
        "fixed t=127": {"stage": "threshold_fixed", "t": 127},
        "otsu": {"stage": "threshold_otsu"},
        "adaptive mean 31/10": ADAPTIVE,
        "adaptive gaussian 31/10": {**ADAPTIVE, "method": "gaussian"},
        "line masks t=60": {"stage": "threshold_lines", "t": 60},
        "LoG sigma=2 t=2": {"stage": "threshold_log", "sigma": 2.0, "t": 2.0},
    }
    return "MEM4", [(k, {**base, "name": "threshold_method",
                         "stages": base["stages"] + [v]}, None) for k, v in options.items()]


def kernel_size():
    """MEM2: smoothing filter type and kernel size (3, 5, 7) as the first stage."""
    ref = _ref()
    out = [("no smoothing", replace_stage(ref, "gaussian", None), None)]
    for kind in ("mean", "gaussian", "median"):
        for k in (3, 5, 7):
            out.append((f"{kind} k={k}",
                        replace_stage(ref, "gaussian", {"stage": kind, "ksize": k}), None))
    return "MEM2", out


def enhancement_methods():
    """MEM2: each enhancement method in front of the same adaptive threshold (raw image)."""
    methods = {
        "none": [], "negative": [{"stage": "negative"}], "log": [{"stage": "log"}],
        "gamma 0.5": [{"stage": "gamma", "gamma": 0.5}],
        "gamma 2.0": [{"stage": "gamma", "gamma": 2.0}],
        "contrast stretch": [{"stage": "contrast_stretch"}],
        "hist equalize": [{"stage": "hist_equalize"}],
        "mean k=5": [{"stage": "mean", "ksize": 5}],
        "gaussian k=5": [{"stage": "gaussian", "ksize": 5}],
        "median k=5": [{"stage": "median", "ksize": 5}],
        "laplacian sharpen": [{"stage": "laplacian_sharpen"}],
        "gradient sharpen": [{"stage": "gradient_sharpen"}],
    }
    return "MEM2", [(k, {"name": "enhancement", "stages": v + [ADAPTIVE]}, None)
                    for k, v in methods.items()]


def cutoff_d0():
    """MEM3: background-flattening cutoff D0 (Gaussian low-pass on the 2x padded grid)."""
    ref = _ref()
    out = [(f"d0={d}", vary(ref, "flatten_background", d0=d), None) for d in (6.0, 12.0, 24.0)]
    out.append(("no flattening", replace_stage(ref, "flatten_background", None), None))
    return "MEM3", out


def se_size_shape():
    """MEM5: closing structuring-element size (3, 5, 7) and shape (disk, square)."""
    ref = _ref()
    out = [("no closing", replace_stage(ref, "closing", None), None)]
    for shape in ("disk", "square"):
        for size in (3, 5, 7):
            out.append((f"{shape} {size}", vary(ref, "closing", size=size, shape=shape), None))
    return "MEM5", out


def object_selection():
    """MEM5: object selection by minimum area and maximum circularity."""
    ref = _ref()
    out = [(f"min_area={a}", vary(ref, "select_objects", min_area=a), None)
           for a in (30, 60, 120, 240)]
    out += [(f"max_circ={c}", vary(ref, "select_objects", max_circularity=c), None)
            for c in (0.3, 0.8)]
    return "MEM5", out


def gap_linking():
    """MEM4: edge linking gap length."""
    ref = _ref()
    out = [("no linking", replace_stage(ref, "link_gaps", None), None)]
    out += [(f"max_gap={g}", vary(ref, "link_gaps", max_gap=g), None) for g in (3, 6, 10)]
    return "MEM4", out


def pipeline_a_threshold():
    """MEM1: global threshold value for Pipeline A (after Gaussian smoothing)."""
    ref = pipeline.load_config("configs/pipeline_a.yaml")
    return "MEM1", [(f"t={t}", vary(ref, "threshold_fixed", t=t), None)
                    for t in (100, 115, 127, 140, 150)]


def adaptive_params():
    """MEM4: classic adaptive threshold block size and offset c, on the reference."""
    ref = _ref()
    out = [(f"block={b}", vary(ref, "threshold_adaptive", block=b), None) for b in (15, 31, 61)]
    out += [(f"c={c}", vary(ref, "threshold_adaptive", c=c), None) for c in (10, 20, 25)]
    out.append(("method=gaussian", vary(ref, "threshold_adaptive", method="gaussian"), None))
    return "MEM4", out


STUDIES = {
    "pipeline_a_threshold": pipeline_a_threshold,
    "threshold_method": threshold_method, "adaptive_params": adaptive_params,
    "kernel_size": kernel_size, "enhancement_methods": enhancement_methods,
    "cutoff_d0": cutoff_d0, "se_size_shape": se_size_shape,
    "object_selection": object_selection, "gap_linking": gap_linking,
}
