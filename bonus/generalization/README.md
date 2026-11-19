# Bonus 3 — Second dataset and generalization (owners MEM1 + MEM5, reviewer MEM2)

## Dataset

DeepCrack — Y. Liu, J. Yao, X. Lu, R. Xie, L. Li, "DeepCrack: A Deep Hierarchical Feature Learning
Architecture for Crack Segmentation," *Neurocomputing* 338 (2019) 139–153,
doi:10.1016/j.neucom.2019.01.036. Source: github.com/yhlleo/DeepCrack, `dataset/DeepCrack.zip`
(67,708,808 bytes, SHA-256 `ec3fc2bee3b71c2cc3c74739cbc51c97b77f78193d08ce3dda0e16d7d41bf585`,
downloaded 19 Nov 2026).

Licence: the dataset README states use is "RESTRICTED to non-commercial research and educational
purposes" (no SPDX licence). A course project is educational use; the images are not redistributed
(`data/raw/` is git-ignored). It does not reuse Özgenel's photos, so it is an independent test.

537 images with binary PNG masks (300 train / 237 test), 544×384, mostly pavement/road and some
concrete surfaces.

## Protocol

`python -m bonus.generalization.run_generalization` (after extracting the zip to
`data/raw/deepcrack/`). Only the 237 **test** images are used. The final configuration
(`configs/final_pipeline.yaml`) is used **unchanged** — no parameter was tuned on DeepCrack. Images
are processed at native size (no loader change needed: masks are PNG with crack = 255).

## Results

| Dataset | Images | Final pipeline IoU | Trivial baseline IoU | Crack pixels (GT) |
|---|---|---|---|---|
| Özgenel (our dev set, L = 1008) | 321 | 0.666 ± 0.120 | 0.442 ± 0.193 | 1.4 % |
| DeepCrack test (native 544×384) | 237 | 0.520 ± 0.201 | 0.346 ± 0.263 | 4.3 % |

Per-image values: `results.csv`. Final-pipeline Dice on DeepCrack: 0.659 ± 0.189.

## Discussion

- The pipeline still beats the trivial baseline by +0.17 IoU, so the design (smoothing, local
  threshold, linking, morphology, shape selection) transfers to another camera and surface type.
- IoU drops by 0.15. Likely causes: (1) scale — the 31-pixel adaptive block and the 60-pixel
  minimum object area were chosen for 1008-pixel images; on 544-pixel images cracks are relatively
  wider and the block covers more of the scene; (2) surface — asphalt texture is coarser than
  concrete, and DeepCrack masks mark wider cracks (4.3 % vs 1.4 % of pixels).
- Limitation: a fair use of the method on a new dataset would rescale images so cracks have a
  similar pixel width, or re-tune the block size on that dataset's own training split; neither was
  done here, on purpose, to measure generalization of the frozen pipeline.
