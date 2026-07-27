# Pen detection switch: HSV heuristic → wsisegqc pen.pt

**Date:** 2026-06-25  
**Model:** UNet++ / ResNet34 (`pen.pt`), Patil et al. 2024 ([arXiv:2410.03289](https://arxiv.org/abs/2410.03289))  
**Repo:** https://github.com/abhijeetptl5/wsisegqc  
**Weights:** `external/wsisegqc/models/models/pen.pt` (downloaded from official Google Drive)

---

## 1. Did pen.pt avoid both known false positives?

**Yes — both tissue-edge false positives are cleared.**

| Slide | HSV (old) | pen.pt pen_area_px | pen.pt pen_% tissue | pen.pt flagged |
|-------|-----------|-------------------|---------------------|----------------|
| `4502c2c9c9c1041564225b9d8fad13c1` | green, 0.35% thumb | 0 | 0.0% | **No** |
| `7d581d0082d6ee0a32165b0f8fe216d8` | green, 0.59% thumb | 1 (border artifact) | 0.06% | **No** (with min 10 px rule) |

Visual previews: `outputs/pen_mark_detection_v2/previews/pen_wsisegqc_*.png`

pen.pt does **not** segment the brownish/olive tissue-edge blobs that triggered the HSV green-pen heuristic. Slide 7d581 has a single border pixel (inference padding artifact); we added `--min-pen-px 10` so this does not flag the slide.

**Step 2 passed → full scan proceeded.**

---

## 2. Full clean set (4,731 slides): flag counts

| Detector | Flagged slides | Notes |
|----------|----------------|-------|
| HSV heuristic (`pen_mark_slides.csv`) | **992** (21.0%) | `green_blue_black_v1`, per-color thresholds on 2048px thumbnail |
| pen.pt (`pen_mark_slides_wsisegqc.csv`) | **671** (14.2%) | flag if `pen_pct_of_tissue ≥ 0.05%` AND `pen_area_px ≥ 10` |

Overlap on full set:
- **Both flag:** 215 slides
- **HSV only:** 777 slides (pen.pt does not flag — many likely HSV false positives)
- **pen.pt only:** 456 slides (HSV missed — segmentation finds ink HSV thresholding missed)
- **Neither:** 3,283 slides

The switch **reduces** total flags vs HSV but is **not a subset** — pen.pt disagrees substantially with HSV on ~1,233 slides.

**Authoritative output:** `outputs/pen_mark_detection_v2/pen_mark_slides_wsisegqc.csv`  
Columns: `slide_id, tissue_area_px, pen_area_px, pen_pct_of_tissue, flagged, status`

Per-slide predictions: `outputs/pen_mark_detection_v2/npz/{slide_id}.npz` (pen + tissue masks at pen-thumb resolution).

---

## 3. Retroactive check: 20-slide sample of old HSV flags

Random sample from the 992 HSV-flagged slides (`hsv_sample_20.csv`):

| Outcome | Count |
|---------|-------|
| pen.pt also flags (agrees ink present) | **8 / 20** |
| pen.pt does NOT flag (likely HSV false positive) | **12 / 20** |

In this sample, **60%** of prior HSV flags appear to be false positives when checked against pen.pt. Examples where HSV flagged but pen.pt did not: `3795ba90ba43` (blue), `128ade97a27c` (green), `d6d59930bb06` (black, 0.66% HSV thumb fraction).

Previews for sample: `outputs/pen_mark_detection_v2/previews/hsv_sample/`

---

## 4. Recommendation

**Adopt pen.pt as the production pen-detection step**, replacing `scan_pen_marks.py` as the source of truth for pen-mark flags.

- `scan_pen_marks.py` is **kept** in the codebase for reference/comparison; do not delete.
- Old `outputs/pen_mark_slides.csv` is **not deleted** pending your confirmation — treat it as archived HSV results.
- Downstream consumers (`stain_normalize.py` reads `pen_mark_slide_ids.txt`) should be pointed at `outputs/pen_mark_detection_v2/pen_mark_slide_ids.txt` when you are ready.

**Flag policy unchanged:** pen detection remains **flag-only** — flagged slides are not auto-excluded from `radboud_clean.csv`.

---

## Infrastructure added

| File | Purpose |
|------|---------|
| `src/pen_wsisegqc.py` | pen.pt inference + tissue-mask metrics |
| `src/scan_pen_marks_wsisegqc.py` | Batch scanner with checkpoint/resume |
| `src/pen_wsisegqc_preview.py` | Side-by-side visualization |
| `scripts/slurm_scan_pen_marks_wsisegqc.sh` | Full GPU scan (partition `gpu`, 1× GPU) |
| `scripts/slurm_pen_wsisegqc_validate.sh` | Step-2 validation on 2 slides |

**Run full scan:**
```bash
sbatch scripts/slurm_scan_pen_marks_wsisegqc.sh
```

**Env:** `wsi_seg` conda env already has `torch 2.7.1+cu118` and `segmentation-models-pytorch 0.5.0` — no new conflicts.

**Slurm job:** 4990551 completed in ~6 min on Tesla V100 (4,731 slides).
