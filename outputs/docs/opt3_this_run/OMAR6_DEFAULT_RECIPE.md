# Omar-6 default recipe — LOCKED 2026-08-16

User lock: keep this for all future Opt3 trains unless they say otherwise.
Entry: `scripts/slurm_train_opt3_slidebag.sh` → `src/train_uni2_opt3_slidebag.py` only.

Fail-closed wiring (abort, never silent like live **5443101**):

- `WIRING_OK live=64 (not micro=4)`
- `lora_params_in_optimizer>0`
- decoder-chunk `checkpoint` so 64-patch ISUP does not keep 64 FPN graphs

## New run vs live 5443101

| | Live **5443101** (leave running) | Locked default / new r2 |
|--|--|--|
| Tag | `opt3_omar6_grouped_soft01` | `opt3_omar6_locked` |
| UNI2 | frozen 100 | frozen 100 |
| LoRA QKV | wrapped, **not in AdamW** | **in AdamW** (abort if 0) |
| Decoder GN | yes | yes |
| Proj outside `no_grad` | no | yes |
| ISUP live | **4** (flag said 64) | **64**, perm=`live_n` (abort if collapsed to micro) |
| 64-patch memory | n/a | chunk=**4** + **decoder checkpoint** (recompute; grads on all 64) |
| Backbone grad-ckpt | no | yes |
| n&lt;5 skip | yes | yes |
| λ_slide warmup | yes | yes |
| Soft ISUP | cancer-only renormalize | absolute p3/p4/p5 + soft vs hard log |
| α=0.1 + benign | yes | yes |
| Save every epoch | no (sidecar) | yes |
| Resume | live `latest.pth` | none (new dir) |

## Locked knobs

```
--freeze-backbone-epochs 100 --lora --decode-norm gn
--live-patches 64 --live-chunk 4 --decoder-checkpoint --grad-checkpoint
--lambda-slide-warmup --lambda-slide 0.3 --lambda-grade 0.3
--min-slide-patches 5 --min-area-pct 0 --adjacent-soft-alpha 0.1 --include-benign-soft
--micro-batch-size 4 --slides-per-epoch 256 --save-every 1
no patch cap; grouped split; max-val-patches 20000
```
