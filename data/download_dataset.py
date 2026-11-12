"""Download and extract the Concrete Crack Segmentation Dataset (Ozgenel, Mendeley Data, CC BY 4.0).

Usage:
    python data/download_dataset.py --out data/raw

Result: <out>/concrete_crack/rgb (458 photos) and <out>/concrete_crack/BW (458 masks).
The archive is a RAR file. Extraction uses the first tool found: bsdtar / Windows tar.exe
(libarchive), unrar, or 7z. On Colab/Ubuntu: `apt-get install -y unrar` if none is present.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

URL = ("https://data.mendeley.com/public-files/datasets/jwsn7tfbrp/files/"
       "88e685a6-e3c5-423d-845f-89e35a457867/file_downloaded")
SHA256 = "1b8458ab6f84dc5086e9af8579e2cabccd3d209df97b46622096a4c105d5a6b2"
ARCHIVE = "concreteCrackSegmentationDataset.rar"
N_PAIRS = 458


def sha256(path):
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_command(archive, dest):
    """Return the first available command that can extract a RAR archive into dest."""
    windows_tar = Path("C:/Windows/System32/tar.exe")
    candidates = [
        (shutil.which("bsdtar"), ["-xf", str(archive), "-C", str(dest)]),
        (str(windows_tar) if windows_tar.exists() else None,
         ["-xf", str(archive), "-C", str(dest)]),
        (shutil.which("unrar"), ["x", "-o+", str(archive), str(dest) + "/"]),
        (shutil.which("7z"), ["x", "-y", f"-o{dest}", str(archive)]),
    ]
    for exe, args in candidates:
        if exe:
            return [exe, *args]
    sys.exit("No RAR extractor found. Install unrar (apt-get install -y unrar) or 7-Zip.")


def main(argv=None):
    """Download (if needed), verify the checksum, extract, and check the pair count."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw", help="folder for the raw dataset")
    out = Path(parser.parse_args(argv).out)
    dataset = out / "concrete_crack"
    if all((dataset / d).is_dir() and len(list((dataset / d).iterdir())) == N_PAIRS
           for d in ("rgb", "BW")):
        print(f"Dataset already present: {dataset}")
        return

    archive = out / "_download" / ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or sha256(archive) != SHA256:
        print(f"Downloading {ARCHIVE} (746 MB) ...")
        urllib.request.urlretrieve(URL, archive)
    if sha256(archive) != SHA256:
        sys.exit(f"Checksum mismatch for {archive}: delete it and run again.")
    print("Checksum OK.")

    dataset.mkdir(parents=True, exist_ok=True)
    subprocess.run(extract_command(archive, dataset), check=True)
    counts = {d: len(list((dataset / d).iterdir())) for d in ("rgb", "BW")}
    if counts != {"rgb": N_PAIRS, "BW": N_PAIRS}:
        sys.exit(f"Unexpected file counts after extraction: {counts}")
    print(f"Done: {N_PAIRS} image/mask pairs in {dataset}")


if __name__ == "__main__":
    main()
