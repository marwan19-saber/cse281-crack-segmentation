# Dataset card

| Item | Value |
|---|---|
| Name | Concrete Crack Segmentation Dataset |
| Author | Çağlar Fırat Özgenel |
| Source | Mendeley Data, v1, 3 April 2019, DOI `10.17632/jwsn7tfbrp.1` |
| Licence | CC BY 4.0 (academic use allowed; cite the source) |
| Archive | `concreteCrackSegmentationDataset.rar`, 745,914,150 bytes |
| SHA-256 | `1b8458ab6f84dc5086e9af8579e2cabccd3d209df97b46622096a4c105d5a6b2` |
| Contents | `rgb/` 458 photos (JPEG), `BW/` 458 masks (JPEG), paired by file name |
| Sizes | 437 × 4032×3024 (landscape or portrait), 20 × 3264×2448; each mask matches its photo |
| Masks | JPEG, mostly 3-channel; crack = gray > 127 (JPEG blur affects ~0.005% of pixels) |
| Foreground | 1.36% crack pixels on average (0.39–8.67%); every image contains a crack |

Checked by MEM1 on 12 Nov 2026. Second-member check (MEM6): pending.

## Download

```bash
python data/download_dataset.py --out data/raw
```

The script downloads the archive from Mendeley, checks the SHA-256, and extracts it to
`data/raw/concrete_crack/{rgb,BW}`. `data/raw/` is git-ignored: never commit the images.

## Citation

Ç. F. Özgenel, "Concrete Crack Segmentation Dataset," Mendeley Data, v1, 2019,
doi: 10.17632/jwsn7tfbrp.1.
