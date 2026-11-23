"""Assemble the Hugging Face Static Space into out/space/ (MEM6, Bonus 1).

    python -m bonus.deployment.build_space

The Space runs the pipeline in the visitor's browser with Pyodide (index.html), so it needs
the src/ modules, the frozen config and the example photos. The same folder is what the Colab
timing check clones (it also needs data/split.json, data/download_dataset.py, evaluate.py).
"""
import shutil
from pathlib import Path

OUT = Path("out/space")
SPACE_README = """---
title: Concrete Crack Measurement
emoji: 🧱
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
---

# Concrete crack segmentation and measurement (CSE281)

Classical image processing (no learned model) running in the browser with Pyodide: upload a
concrete photo and see every pipeline stage, the crack mask, the cracked area, the crack length
and a severity class. Source code and report: the team's GitHub repository.

Dataset for the examples: Ç. F. Özgenel, Concrete Crack Segmentation Dataset, Mendeley Data,
2019, doi:10.17632/jwsn7tfbrp.1 (CC BY 4.0).

Files: `index.html` (page), `src/` (pipeline modules), `configs/final_pipeline.yaml` (frozen
configuration), `examples/` (three test-set photos), `data/` (split and download script, used by
the Colab timing check).
"""


def main():
    """Copy the files the Space and the Colab check need."""
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    for f in Path("src").glob("*.py"):
        shutil.copy(f, OUT / "src" / f.name)
    for sub, files in (("configs", ["configs/final_pipeline.yaml"]),
                       ("data", ["data/split.json", "data/download_dataset.py"]),
                       ("examples", sorted(Path("app/examples").glob("*")))):
        (OUT / sub).mkdir()
        for f in files:
            shutil.copy(f, OUT / sub / Path(f).name)
    shutil.copy("bonus/deployment/index.html", OUT / "index.html")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"built {OUT}: {sum(1 for _ in OUT.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
