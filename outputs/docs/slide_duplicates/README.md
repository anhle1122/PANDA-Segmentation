# Slide duplicate scan (job 5382352)

**Host:** esplhpc-cp011 (CPU `defq`, not GPU) · **Wall:** ~11 min (12:22–12:33 PDT)

## Do not treat loose results as true duplicates

Tissue-silhouette IoU on thin prostate cores is **not distinctive**. Same-gleason needle biopsies collapse into huge clusters after bbox-normalize + flip/rot.

| Threshold | Pairs | Cross-split pairs | Note |
|---|---:|---:|---|
| IoU ≥ 0.28 (review) | 36,735 | 12,735 | Too loose — almost all thin cores |
| IoU ≥ 0.50 (high) | 5,268 | 1,852 | Still over-clusters (100 clusters, 4248 slides) |

User-flagged true lookalikes **were recovered**:
- `48440a60`/`4889d110` IoU **0.860**
- `dd11c914`/`72e64850` IoU **0.427**
- `e707ef8c`/`507ac341` IoU **0.343**

## Next (needed before dedupe)

Use a stronger cue than silhouette: RGB thumbnail perceptual hash / ORB, require near-equal dims (±15%), same gleason, and **human review** of small candidate pairs. Then re-map train/val/test leakage.
