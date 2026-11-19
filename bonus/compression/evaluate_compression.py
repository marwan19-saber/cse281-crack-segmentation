"""Measure RLE and Huffman on real final-pipeline outputs (MEM3, Bonus 4).

    python -m bonus.compression.evaluate_compression

For 10 dev images: the final crack mask (RLE) and the gray input (Huffman). Original size is
1 byte per pixel (how a uint8 mask/image is stored in memory); PNG is shown for reference.
Writes bonus/compression/results.csv.
"""
import csv
import time

import cv2
import numpy as np

from bonus.compression import codec
from src import data, pipeline


def _timed(fn, *args):
    """(result, seconds)."""
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def main():
    """Compress, decompress, verify and tabulate."""
    config = pipeline.load_config("configs/final_pipeline.yaml")
    pairs = {img.stem: img for img, _ in data.list_pairs("data/raw/concrete_crack")}
    rows = []
    for name in data.load_split("data/split.json")["dev"][:10]:
        gray = data.load_image(pairs[name], size=1008)
        mask, _ = pipeline.run(gray, config)
        for kind, arr, enc, dec in (("mask / RLE", mask, codec.rle_encode, codec.rle_decode),
                                    ("gray / Huffman", gray, codec.huffman_encode,
                                     codec.huffman_decode)):
            blob, t_enc = _timed(enc, arr)
            back, t_dec = _timed(dec, blob)
            scale = 255 if arr.dtype == bool else 1
            png = cv2.imencode(".png", arr.astype(np.uint8) * scale)
            rows.append({
                "image": name, "data": kind, "original_bytes": arr.size,
                "compressed_bytes": len(blob), "ratio": round(arr.size / len(blob), 2),
                "png_bytes": len(png[1]), "encode_s": round(t_enc, 4),
                "decode_s": round(t_dec, 4), "exact": bool(np.array_equal(back, arr)),
            })
    with open("bonus/compression/results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for kind in ("mask / RLE", "gray / Huffman"):
        part = [r for r in rows if r["data"] == kind]
        print(f"{kind:15} ratio mean {np.mean([r['ratio'] for r in part]):6.2f} "
              f"(min {min(r['ratio'] for r in part)}, max {max(r['ratio'] for r in part)}) | "
              f"encode {np.mean([r['encode_s'] for r in part]):.3f} s, decode "
              f"{np.mean([r['decode_s'] for r in part]):.3f} s | all exact: "
              f"{all(r['exact'] for r in part)} | PNG ratio "
              f"{np.mean([r['original_bytes'] / r['png_bytes'] for r in part]):.2f}")


if __name__ == "__main__":
    main()
