# Live 5443101 — launch config (read-only)

Pulled from `logs/train_opt3_slidebag_5443101.out` (process start **Sat Aug 15 13:37:28 PDT 2026**), `training_log.csv`, and the in-memory trainer semantics. **Not** the current repo trainer. Live job not modified.

Per-epoch table with LR + λ_slide warmup: `scorecard_lr_warmup.csv` (same folder + next to the ckpt log).

## Launch (actual argv)

`--lr` not passed → **1e-4**. `--epochs 100`. `--micro-batch-size 4 --live-patches 64 --slides-per-epoch 256 --max-val-patches 20000 --lambda-slide-warmup --lora --decode-norm gn --val-batch-size` not passed → **8**. No `--grad-accum*`.

Python printed `lora=True decode_norm=gn live_patches=64`. Epoch banner is still `=== Epoch N: λ_slide=0.300 λ_grade=0.3 ===` (no `live=`). The commit that wired `live_n` into the ISUP `perm` landed **after** this process started, so **ISUP/grade still uses micro=4**, not 64.

## LR schedule (no LR warmup)

`CosineAnnealingLR(T_max=100, eta_min=1e-6)`. Scheduler steps **once per epoch**. Decay starts at epoch 1 (slow cosine, not a step decay).

| epoch | logged LR | fraction of 1e-4 |
|------:|----------:|-----------------:|
| 1 | 9.9976e-5 | 99.98% |
| 7 | 9.8808e-5 | 98.81% |
| 10 | 9.7577e-5 | 97.58% |
| 19 | 9.1440e-5 | 91.44% |
| 20 | 9.0546e-5 | 90.55% |
| 26 | 8.4385e-5 | 84.39% |

At ep19–20 LR is **not** still at init, but it has only decayed ~8.5–9.5%. There is **no LR warmup**. `--lambda-slide-warmup` is **λ_slide** only.

## λ_slide warmup (finished long before ep19)

| epochs | λ_slide |
|--------|---------|
| 1–5 | 0.00 |
| 6–9 | 0.06 / 0.12 / 0.18 / 0.24 |
| ≥10 | 0.30 |

Ep19–20: warmup **done**; λ_slide = 0.3 (matches the live epoch banners).

## Batch / accumulation

| path | live 5443101 |
|------|----------------|
| Pixel micro-batch | **4** per GPU (2 GPUs) |
| ISUP/grade live bag | **4** in-memory (flag said 64, not wired) |
| Named grad-accumulation | **none** |
| Optimizer step | **1 per slide bag** (pixel micros `no_sync` accumulate grads, then one synced ISUP backward, then `step`) |
| Slides / epoch / rank | 256 |
| Val loader batch | 8 |

## Val set for “val cancer Dice”

`--max-val-patches 20000`, `subsample_split_csv(..., seed=42)`. Full PANDA val = **55,516 patches / 472 slides**. Subsample = **20,000 patches still covering all 472 slides**. Same 20k every epoch (deterministic). Val Dice is a deterministic function of weights on a **fixed** pixel set (GN, no val aug).

**ep18 0.460 → ep19 0.545 (+0.085) is not val-set sampling noise.** Median |Δ| cancer Dice on this run is 0.047; max 0.099. The jump is a real model-level swing, large vs the median but inside this run’s own volatility.

## Epoch 7 PANDA vs PANDA+

| | PANDA (train val) | PANDA+ |
|--|-------------------|--------|
| **Cancer Dice** | **0.608312** (20k-patch val, 472 slides) | **0.587217** (4,688 patches, 48 slides, gt≥2) |
| **ISUP match** | **never run** on in-domain PANDA val | **30/48 (62.5%)**; primary 37/48 (77.1%); both 30/48 (62.5%) |
| **G5 precision** | (not in training_log) | **0.569370** |

Dice does **not** match (0.608 vs 0.587). ISUP cannot be compared: only PANDA+ was scored.
