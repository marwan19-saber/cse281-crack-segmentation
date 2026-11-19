# CSE281 — Classical Crack Segmentation and Measurement on Concrete

Digital Image Processing course project (Alamein International University, CSE281).
Classical image processing only: no machine-learning models.

## Problem (approved by the team, 19 Nov 2026)

A building-inspection engineer needs to know where a concrete surface is cracked and how much.
The pipeline takes one photo and outputs a binary crack mask, the cracked-area percentage, the
crack length and a severity class. A missed crack is a safety cost; a false crack (stain, shadow,
pore) wastes inspection time. Cracks cover only ~1.4% of the pixels, so the primary metric is
**IoU of the crack class**.

## Dataset

Concrete Crack Segmentation Dataset — Ç. F. Özgenel, Mendeley Data v1 (2019),
DOI [10.17632/jwsn7tfbrp.1](https://data.mendeley.com/datasets/jwsn7tfbrp/1), licence CC BY 4.0.
458 photos with pixel masks. See [data/README.md](data/README.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python data/download_dataset.py --out data/raw
pytest
```

## Team

| Member | Name | Work package | Main files |
|---|---|---|---|
| MEM1 | Marwan Saber | Tech lead, data, metrics, baselines | `src/data.py`, `src/metrics.py`, `src/pipeline.py`, `configs/` |
| MEM2 | Hassan Hisham | Enhancement | `src/enhancement.py` |
| MEM3 | Mohamed Wageh | Restoration and frequency | `src/restoration.py`, `src/frequency.py` |
| MEM4 | Saad Mohamed | Segmentation | `src/segmentation.py` |
| MEM5 | Ahmed Abdelwahed | Morphology, evaluation, demo | `src/morphology.py`, `src/evaluate.py`, `src/run_pipeline.py`, `app/` |

## Results

Mean IoU ± std over images. Methods and parameters were chosen on the development set only; the
test set was evaluated once, after the final configuration was tagged `results-frozen`.

| Method | Dev (321 images) | Test (137 images) |
|---|---|---|
| Trivial baseline (gray < 127) | 0.442 ± 0.193 | 0.449 ± 0.189 |
| Simple baseline (Otsu + 3×3 opening) | 0.066 ± 0.144 | 0.066 ± 0.140 |
| Pipeline A (Gaussian + global threshold) | 0.458 ± 0.238 | 0.443 ± 0.243 |
| Pipeline B (+ flattening, adaptive threshold, linking) | 0.512 ± 0.129 | 0.513 ± 0.139 |
| **Pipeline C = final** (+ morphology, object selection) | **0.666 ± 0.120** | **0.670 ± 0.116** |

- Final pipeline beats the trivial baseline on 131 of 137 test images; best 0.881, worst 0.330.
- Cracked area: mean absolute error 0.33 points; severity class agrees with GT on 62.8 %.
- Robustness (dev, Gaussian noise, seeds 0/1/2): σ 10 → 0.662, σ 20 → 0.650, σ 30 → 0.622.
- Speed: 0.50 s per image at 1008 px (≈ 1.4 min for the test set locally).
- Details: `report/report.pdf`, `notebooks/02_results.ipynb`, `experiments/results/`.

## Reproduce

```bash
python -m src.evaluate --config configs/final_pipeline.yaml --split dev --out dev.csv
python -m src.studies --study ablations --set dev           # any study in src/study_defs.py
python -m src.evaluate --config configs/final_pipeline.yaml --split test --out test.csv
python -m src.run_pipeline --image app/examples/test_249_failure_wide_crack.jpg --out out/
python app/app.py                                           # demo (pip install -r app/requirements.txt)
```

The test split only runs when the `results-frozen` tag exists (`git fetch --tags`).

## Bonus work

| Bonus | Result | Evidence |
|---|---|---|
| Tests + CI | 157 pytest tests on every push and PR (Python 3.12, 3.13) | `.github/workflows/tests.yml` |
| Speed | 0.922 → 0.502 s/img (1.84×), identical masks | `bonus/optimization/` |
| Lossless compression | RLE on masks ratio 441, Huffman on gray 1.36, exact | `bonus/compression/` |
| Second dataset | DeepCrack test, no retuning: IoU 0.520 (trivial 0.346) | `bonus/generalization/` |
| Public demo | Gradio app runs locally; public hosting pending | `app/` |

## AI assistance

An AI coding assistant (Claude Code) drafted much of the code, tests, experiment scripts and
documents. Each commit names it as co-author; every member reviews and must be able to explain
their own work package and the whole pipeline.

## Documents

`report/report.pdf` (technical report), `report/presentation.pdf` (slides),
`CSE281_Team_Execution_Plan.pdf` (team plan). HTML sources in `docs/`.
