# CSE281 — Classical Crack Segmentation and Measurement on Concrete

Digital Image Processing course project (Alamein International University, CSE281).
Classical image processing only: no machine-learning models.

## Problem (draft, awaiting team approval)

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

| Member | Work package | Main files |
|---|---|---|
| MEM1 | Tech lead, data, metrics, baselines | `src/data.py`, `src/metrics.py`, `src/pipeline.py`, `configs/` |
| MEM2 | Enhancement | `src/enhancement.py` |
| MEM3 | Restoration and frequency | `src/restoration.py`, `src/frequency.py` |
| MEM4 | Segmentation | `src/segmentation.py` |
| MEM5 | Morphology, evaluation, demo | `src/morphology.py`, `src/evaluate.py`, `src/run_pipeline.py`, `app/` |

## Results

Not available yet. Every number here will come from `experiments/results/`.

## Planning documents

`CSE281_Team_Execution_Plan.pdf` (team plan) and `report/` (drafts). Sources in `docs/execution/`.
