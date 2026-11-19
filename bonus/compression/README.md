# Bonus 4 — Lossless compression (owner MEM3, reviewer MEM1)

Code: `codec.py` (RLE and canonical Huffman, encoder + decoder). Tests: `tests/test_compression.py`
(hand examples, exact round trips with `np.array_equal`, empty / constant / 1-pixel inputs).

Run: `python -m bonus.compression.evaluate_compression` → `results.csv`.

## Results (final-pipeline outputs, 10 dev images, working size 1008)

| Data | Method | Mean ratio (min–max) | Encode | Decode | Exact | PNG ratio (reference) |
|---|---|---|---|---|---|---|
| Crack mask (bool) | RLE | 440.7 (264.6–851.5) | 0.002 s | 0.001 s | all 10 | 163.7 |
| Gray input (uint8) | Huffman | 1.36 (1.27–1.50) | 0.52 s | 0.49 s | all 10 | 1.93 |

Original size = 1 byte per pixel. Ratio = original bytes / compressed bytes.

## Limitations

- RLE suits masks (cracks are ~1% of pixels in a few long runs). On a noisy image it would expand
  the data: every pixel change starts a new run.
- Huffman only uses the gray-level histogram (coding redundancy). Neighbouring pixels are strongly
  correlated, which Huffman ignores; PNG predicts each pixel from its neighbours first and wins
  (1.93 vs 1.36).
- The Huffman decoder walks the bit stream in Python (about 0.5 s per 1008×756 image); a lookup
  table on fixed-size bit chunks would make it much faster.
