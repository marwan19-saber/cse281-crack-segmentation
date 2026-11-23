"""Study definitions: one factor changed per variant against a reference config (MEM1).

Each entry returns (author, variants); a variant is (label, config, noise_or_None).
Reference = configs/pipeline_c.yaml unless stated. Owners follow the execution plan.
"""
import copy

from src import pipeline
from src.studies import replace_stage, vary

ADAPTIVE = {"stage": "threshold_adaptive", "block": 31, "c": 10, "method": "mean"}


def _ref():
    """Reference configuration for one-factor-at-a-time tuning."""
    return pipeline.load_config("configs/pipeline_c.yaml")


def threshold_method():
    """MEM5: threshold method after the same noise removal and illumination flattening."""
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
    return "MEM5", [(k, {**base, "name": "threshold_method",
                         "stages": base["stages"] + [v]}, None) for k, v in options.items()]


def kernel_size():
    """MEM3: smoothing filter type and kernel size (3, 5, 7) as the first stage."""
    ref = _ref()
    out = [("no smoothing", replace_stage(ref, "gaussian", None), None)]
    for kind in ("mean", "gaussian", "median"):
        for k in (3, 5, 7):
            out.append((f"{kind} k={k}",
                        replace_stage(ref, "gaussian", {"stage": kind, "ksize": k}), None))
    return "MEM3", out


def enhancement_methods():
    """MEM3: each enhancement method in front of the same adaptive threshold (raw image)."""
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
    return "MEM3", [(k, {"name": "enhancement", "stages": v + [ADAPTIVE]}, None)
                    for k, v in methods.items()]


def cutoff_d0():
    """MEM4: background-flattening cutoff D0 (Gaussian low-pass on the 2x padded grid)."""
    ref = _ref()
    out = [(f"d0={d}", vary(ref, "flatten_background", d0=d), None) for d in (6.0, 12.0, 24.0)]
    out.append(("no flattening", replace_stage(ref, "flatten_background", None), None))
    return "MEM4", out


def se_size_shape():
    """MEM6: closing structuring-element size (3, 5, 7) and shape (disk, square)."""
    ref = _ref()
    out = [("no closing", replace_stage(ref, "closing", None), None)]
    for shape in ("disk", "square"):
        for size in (3, 5, 7):
            out.append((f"{shape} {size}", vary(ref, "closing", size=size, shape=shape), None))
    return "MEM6", out


def object_selection():
    """MEM6: object selection by minimum area and maximum circularity."""
    ref = _ref()
    out = [(f"min_area={a}", vary(ref, "select_objects", min_area=a), None)
           for a in (30, 60, 120, 240)]
    out += [(f"max_circ={c}", vary(ref, "select_objects", max_circularity=c), None)
            for c in (0.3, 0.8)]
    return "MEM6", out


def gap_linking():
    """MEM5: edge linking gap length."""
    ref = _ref()
    out = [("no linking", replace_stage(ref, "link_gaps", None), None)]
    out += [(f"max_gap={g}", vary(ref, "link_gaps", max_gap=g), None) for g in (3, 6, 10)]
    return "MEM5", out


def pipeline_a_threshold():
    """MEM1: global threshold value for Pipeline A (after Gaussian smoothing)."""
    ref = pipeline.load_config("configs/pipeline_a.yaml")
    return "MEM1", [(f"t={t}", vary(ref, "threshold_fixed", t=t), None)
                    for t in (100, 115, 127, 140, 150)]


def adaptive_params():
    """MEM5: classic adaptive threshold block size and offset c, on the reference."""
    ref = _ref()
    out = [(f"block={b}", vary(ref, "threshold_adaptive", block=b), None)
           for b in (15, 31, 61)]
    out += [(f"c={c}", vary(ref, "threshold_adaptive", c=c), None) for c in (10, 20, 25)]
    out.append(("method=gaussian", vary(ref, "threshold_adaptive", method="gaussian"), None))
    return "MEM5", out


def _final():
    """The frozen final pipeline (configs/final_pipeline.yaml)."""
    return pipeline.load_config("configs/final_pipeline.yaml")


def ablations():
    """AB1-AB4 (+ extras): remove or swap one stage of the final pipeline (full dev set)."""
    f = _final()
    smooth = next(s for s in f["stages"] if s["stage"] in ("gaussian", "mean", "median"))
    no_morph = {**f, "stages": [s for s in f["stages"] if s["stage"] not in
                                ("closing", "fill_small_holes", "select_objects")]}
    morph_open_first = copy.deepcopy(f)
    i = next(k for k, s in enumerate(morph_open_first["stages"]) if s["stage"] == "closing")
    morph_open_first["stages"].insert(i, {"stage": "opening", "size": 3, "shape": "square"})
    return "MEM1", [
        ("final pipeline", f, None),
        ("AB1 no enhancement (no smoothing, no flattening)",
         replace_stage(replace_stage(f, smooth["stage"], None), "flatten_background", None),
         None),
        ("AB2 frequency low-pass instead of spatial smoothing",
         replace_stage(f, smooth["stage"], {"stage": "lowpass", "d0": 150.0}), None),
        ("AB3 global Otsu instead of adaptive threshold",
         replace_stage(f, "threshold_adaptive", {"stage": "threshold_otsu"}), None),
        ("AB4a no morphology / object selection", no_morph, None),
        ("AB4b opening then closing (order)", morph_open_first, None),
        ("AB5 no edge linking", replace_stage(f, "link_gaps", None), None),
        ("AB6 no illumination flattening", replace_stage(f, "flatten_background", None), None),
    ]


def robustness():
    """Final pipeline under Gaussian noise sigma 10/20/30, seeds 0/1/2 (full dev set)."""
    f = _final()
    return "MEM1", [("clean", f, None)] + [
        (f"gaussian sigma={s} seed={seed}", f, ("gaussian", s, seed))
        for s in (10, 20, 30) for seed in (0, 1, 2)]


def gray_levels():
    """Gray-level study on the final pipeline: quantize the input to K levels first."""
    f = _final()
    return "MEM1", [(f"{k} levels", {**f, "stages": [{"stage": "quantize", "levels": k}]
                                     + f["stages"]}, None) for k in (256, 64, 32, 16, 8, 4)]


def combine_round2():
    """MEM1: round-2 winners combined vs the round-2 reference (interaction check)."""
    ref = _ref()
    old = vary(vary(vary(vary(ref, "gaussian", ksize=7), "flatten_background", d0=12.0),
                    "link_gaps", max_gap=3), "closing", size=3)
    return "MEM1", [("round-2 reference", old, None), ("combined winners", ref, None)]


def final_selection():
    """MEM1: every pipeline and both baselines on the full dev set; the best is frozen."""
    load = pipeline.load_config
    old_c = combine_round2()[1][0][1]
    return "MEM1", [
        ("trivial baseline", load("configs/baseline_trivial.yaml"), None),
        ("simple baseline", load("configs/baseline_simple.yaml"), None),
        ("pipeline A", load("configs/pipeline_a.yaml"), None),
        ("pipeline B", load("configs/pipeline_b.yaml"), None),
        ("pipeline C (round-2 reference)", {**old_c, "name": "pipeline_c_r2ref"}, None),
        ("pipeline C (combined winners)", load("configs/pipeline_c.yaml"), None),
    ]


def failure_fixes():
    """Fixes proposed by the test-set failure analysis, tested on the DEV set only (the frozen
    final config is not changed)."""
    f = _final()
    wide = replace_stage(f, "threshold_adaptive",
                         {"stage": "threshold_adaptive_or_otsu", "block": 31, "c": 15})
    return "MEM6", [
        ("final pipeline", f, None),
        ("fix wide cracks: adaptive OR Otsu", wide, None),
        ("fix texture: min_area 120, max_circularity 0.3",
         vary(f, "select_objects", min_area=120, max_circularity=0.3), None),
    ]


def detectors():
    """MEM5: edge/line detectors with the same rule (top 3 % strongest responses) after the
    final pipeline's smoothing and flattening."""
    front = [s for s in _final()["stages"] if s["stage"] in ("gaussian", "flatten_background")]
    return "MEM5", [(op, {"name": "detectors", "stages": front + [
        {"stage": "edges_top_percent", "operator": op, "percent": 3.0}]}, None)
        for op in ("roberts", "prewitt", "sobel", "laplacian", "log", "lines")]


STUDIES = {
    "detectors": detectors,
    "failure_fixes": failure_fixes,
    "final_selection": final_selection,
    "combine_round2": combine_round2,
    "ablations": ablations, "robustness": robustness, "gray_levels": gray_levels,
    "pipeline_a_threshold": pipeline_a_threshold,
    "threshold_method": threshold_method, "adaptive_params": adaptive_params,
    "kernel_size": kernel_size, "enhancement_methods": enhancement_methods,
    "cutoff_d0": cutoff_d0, "se_size_shape": se_size_shape,
    "object_selection": object_selection, "gap_linking": gap_linking,
}
