# Omar-6 default recipe — audit (2026-08-16)

Compare Omar’s original 6 + later add-ons against:

- **Live** `5443101` — in-memory trainer from Aug 15 13:37 (do not restart)
- **Disk / future r2** — `src/train_uni2_opt3_slidebag.py` + `scripts/slurm_train_opt3_slidebag.sh` at `c40efb7`
- **Dead r2** `5445233` — started 12:29 on that day’s disk (pre-LoRA-wiring), OOMd 12:57

One Opt3 entry point only. Do not use `src/train_uni2_upernet.py` (freeze=5).

## Omar’s original 6

| # | Agreed rule | Live 5443101 | Disk (future r2) | Lock? |
|---|-------------|--------------|------------------|-------|
| **1** | n&lt;5 → `L_seg` only; no `L_slide` / `L_grade` | yes `--min-slide-patches 5` | yes, default 5 | **yes** |
| **2** | Warm `λ_slide` 0 ep1–5, ramp 6–9, 0.3 from ep10. `λ_grade` constant | yes | yes, default on | **yes** |
| **3** | Decoder GroupNorm; ViT stays LayerNorm | yes | yes, default `gn` | **yes** |
| **4** | UNI2 **frozen**; LoRA QKV adapters train (r=8 α=16). No 681M unfreeze | freeze=100, LoRA wrapped, **not stepped** | freeze=100, LoRA **in AdamW**, skip `no_grad` when LoRA trainable | **yes on disk** — confirm log `lora_params_in_optimizer=1179648` |
| **5a** | 1×1 FPN proj **outside** backbone `no_grad` (train from ep1) | no | yes | **yes on disk** |
| **5b** | Live ISUP **64** + grad-checkpoint (not a 4-patch estimate) | **no** (ISUP=4) | flags yes; **5445233 still OOMd** at FPN ~slide 56 with chunk=8 | **not yet** — 64/8 is not memory-safe |
| **6** | Soft ISUP keeps **absolute tumor burden**; check soft argmax vs hard `derive_grade` | no (cancer-only renormalize; hard discarded) | absolute p3/p4/p5; epoch `soft_hard_isup_agree` | **yes on disk** (coefficients still hand-tuned; we log agreement, we did not retune) |

## Later add-ons (Omar-6 lock + this week)

| Add-on | Live 5443101 | Disk (future r2) | Lock? |
|--------|--------------|------------------|-------|
| Grouped fusion split 3739/472/472 | yes | yes (reads `panda_train.csv`) | **yes** |
| α=0.1 benign↔G3↔G4↔G5 | yes | yes, default on | **yes** |
| `min_area_pct=0` (no 5% gate) | yes | yes | **yes** |
| No patch cap | yes | yes `MAX_PATCHES_PER_SLIDE=0` | **yes** |
| Pixel micro_bs=4, all patches, `no_sync` | yes | yes | **yes** |
| Save **every** epoch `epoch_XXX_*.pth` | no (every-5 in-memory; sidecar copies) | yes `--save-every 1`, prune off | **yes** |
| Mid-epoch `latest.pth` every 8 slides | yes | yes | **yes** |
| Live-chunk FPN (memory) | n/a (live=4) | default chunk=**8** — still OOM 132G | **open** |
| 2×H200, AMP, aug on | yes | yes | **yes** |
| Cosine LR 1e-4, T_max=100, no LR warmup | yes | yes | **yes** |
| Cold start, **new tag**, never write into live ckpt dir | n/a (this is the live tag) | script default `RUN_TAG=opt3_omar6_grouped_soft01` + **auto-resume `latest.pth`** | **must override** or a “fresh r2” resumes live |
| Wall | ~7d remaining | script `#SBATCH --time=30-00:00:00` | pick 3d vs 30d per hunt |
| Between-round ISUP referee / teacher pack | not in-train | not in-train (separate jobs) | keep out of train recipe |

## Proposed default (lock after 5b memory is real)

```
UNI2-h + UPerNet GN
freeze UNI2 100 epochs; LoRA QKV r=8 α=16 in optimizer
λ_slide warmup 0→0.3 by ep10; λ_grade=0.3
α=0.1 benign↔G3–G5; min_area=0; n<5 skip ISUP
no patch cap; micro_bs=4; slides/ep=256
live=64 + grad-checkpoint + live-chunk=??? (8 OOMd)
save every epoch; new RUN_TAG; resume=none
grouped split; max-val-patches=20000 seed=42
```

**Do not submit another r2 until live-chunk (or equivalent) actually bounds FPN below ~80G on bag 1.** Dead job `5445233` already proved 64/8 + ckpt is not enough.

**Slurm must** `--export=ALL,RUN_TAG=<new>` and pass no auto-resume (empty `latest` in that new dir, or pass a dummy so the script does not pick the live `latest.pth`).
