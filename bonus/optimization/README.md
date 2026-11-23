# Bonus 5 — Speed optimization (coordinated by MEM1; each owner optimizes their module)

Measured with `python -m bonus.optimization.profile_pipeline --label before|after`: final pipeline,
20 dev images, working size 1008, idle CPU, one process (`timings.csv`). The script saves the masks
of every run; **the optimized masks are identical to the original ones** (`np.array_equal` on all
20 images).

Speedup = original time / optimized time.

| Stage | Owner | Original (s/img) | Optimized (s/img) | Speedup | Change |
|---|---|---|---|---|---|
| flatten_background | MEM4 | 0.621 | 0.375 | 1.66× | real-input FFT (`rfft2`/`irfft2`) |
| fill_small_holes | MEM6 | 0.185 | 0.031 | 6.0× | one background labelling instead of repeated dilation |
| select_objects | MEM6 | 2.87* | 0.04* | ~70× | area and perimeter for all objects with `np.bincount` |
| other stages | — | 0.116 | 0.095 | — | unchanged (run-to-run variation) |
| **Total per image** | | **0.922** | **0.502** | **1.84×** | |

\* `select_objects` was optimized earlier during integration (commit "perf(morphology):
select_objects measures all objects at once"); its numbers come from a 756×1008 mask with 5 %
random noise, measured against the per-object `measure()` path, which gives the same output.
The "original" total above already includes that change.

Test set: 137 images × 0.502 s plus JPEG decoding (~0.1 s/img) ≈ 1.4 min locally, well inside the
10-minute Colab budget (Colab CPUs are slower; still to be confirmed on Colab).

## Why each change is safe

- **rfft2**: the padded image is real and the transfer function h(u, v) depends only on the
  distance from (P/2, Q/2), so h(k) = h(−k mod P). The filtered result is therefore real, and the
  half spectrum returned by `rfft2` holds all the information. Same arithmetic, about half the work.
- **Hole filling**: "background not 4-connected to the border" is exactly the set of background
  components (4-connectivity) that touch no border pixel. One labelling pass finds them; the
  iterative version needed hundreds of full-image dilations on large backgrounds.
- **Object selection**: separate 8-connected components can never be 4-adjacent, so every boundary
  pixel belongs to exactly one object and per-object perimeters come from one `bincount`.

## Trade-offs

- `rfft2` relies on the symmetry of h; a non-symmetric (e.g. directional) filter would need the
  full FFT again.
- The largest remaining cost is still the FFT on the 2× padded grid. Dropping illumination
  flattening entirely would make the pipeline ~5× faster for −0.003 IoU on dev (ablation AB6), but
  the frozen final pipeline keeps it.
