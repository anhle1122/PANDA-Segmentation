# Mask balance pyramid verification summary

## 1. What pyramid level was used?

The original `plot_mask_class_balance.py` reads each mask with:

```python
lvl = slide.level_count - 1  # coarsest pre-baked pyramid level
arr = slide.read_region((0, 0, lvl, (width, height))  # channel 0 = class ID
```

This is **OpenSlide `read_region` at a pre-baked TIFF pyramid level**, NOT `get_thumbnail()` and NOT a runtime PIL resize.

Across the verification sample (30 slides):
- Pyramid level index: **2** (mode; range 2–2)
- Downsample factor vs L0: **16× to 16×** (typically 16× when level_count=3)

Pre-baked pyramid pixels are produced at scan/export time by the WSI toolchain — not by our code.

## 2. Largest single-slide discrepancy (percentage points)

- **G3**: max |diff| = 0.0088 pp on `b470c2276d978f63d5b1c8a5a9446212` (L0 3.5015% vs pyramid 3.4927%)
- **G4**: max |diff| = 0.0079 pp on `687d89772580c411d434b346e1e6c505` (L0 4.6310% vs pyramid 4.6389%)
- **G5**: max |diff| = 0.0096 pp on `1364c10e1e7f1ad0457f649a44d74888` (L0 3.9940% vs pyramid 3.9844%)

## 3. Systematic bias (30-slide sample)

- **background**: mean diff (L0−pyr) = -0.0013 pp, median = -0.0004 pp
- **stroma**: mean diff (L0−pyr) = +0.0022 pp, median = +0.0014 pp
- **benign**: mean diff (L0−pyr) = -0.0001 pp, median = -0.0001 pp
- **G3**: mean diff (L0−pyr) = +0.0002 pp, median = +0.0000 pp
- **G4**: mean diff (L0−pyr) = -0.0012 pp, median = +0.0000 pp
- **G5**: mean diff (L0−pyr) = +0.0003 pp, median = +0.0000 pp

## 4. Known failure patterns in sample

### Vanishing (L0>0, pyramid=0 pixels)
- **None** in the 30-slide sample.

### Fabricated (L0=0, pyramid>0 pixels)
- **None** in the 30-slide sample.

## 5. Aggregate comparison (30-slide sample)

| Class | 30-slide L0 % | 30-slide pyramid % | diff (pp) | Full-dataset pyramid % (post-QC) |
|---|---:|---:|---:|---:|
| background | 84.5842 | 84.5847 | -0.0005 | 83.3164 |
| stroma | 12.3058 | 12.3046 | +0.0012 | 13.6469 |
| benign | 0.3433 | 0.3436 | -0.0003 | 0.7653 |
| G3 | 1.0917 | 1.0916 | +0.0001 | 0.7784 |
| G4 | 0.9833 | 0.9838 | -0.0005 | 1.2462 |
| G5 | 0.6917 | 0.6917 | -0.0000 | 0.2468 |

**30-slide test:** L0 vs pyramid aggregate differs by at most 0.0012 pp on any single class; combined cancer grades (G3+G4+G5) diff = -0.0004 pp.

## 6. Recommendation

The pyramid-level dataset-wide percentages look **trustworthy for reporting coarse balance** (background/stroma/cancer split). Per-slide rare-grade percentages can differ more, but aggregate bias across the 30-slide sample is sub-0.05 pp per class.

A full level-0 Slurm job is **not required** before using these numbers for high-level class-balance discussion. Run L0 aggregate only if you need exact cancer-grade percentages beyond ~0.01 pp precision.