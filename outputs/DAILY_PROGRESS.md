# Daily progress — PANDA Gleason project

**How to use:** Open this file each day (also mirrored at `docs/DAILY_PROGRESS.md`). Newest date at the top.  
Agents auto-append when training/eval/protocol work lands (see `.cursor/rules/daily-progress.mdc`).

Format per bullet: `- HH:MM TZ | What | Why | Result / next`

---

## 2026-07-26

- **12:15 PDT** | Restore wiped train/eval code from git + commit handoff pack | Disk had deleted `train_uni2_upernet.py`, `evaluate.py`, almost all `src/train/*` (still in git); R5 had failed earlier on a corrupted syntax line | Restored from `HEAD`; re-added `scripts/phase3_apply_corrections.py` (5% diagnostic); wrote root `README.md` for Omar pseudo-label-loop handoff; docs refreshed
- **R5 status** | Job `5260610` | Got H200 `cp098` Jul 23 14:57 → **FAILED** in ~9s (`SyntaxError` on then-corrupt `train_uni2_upernet.py`). No `h200x4_10k_adj015` ckpt. Needs resubmit after restore
- **Handoff note** | Data/ckpts/evals solid; code was not — now restored and being committed so wipe cannot recur

### Open / next
- [ ] `git push` when network/approval ready (local commits only so far)
- [ ] Resubmit R5 `scripts/slurm_train_uni2_10k_adj015_h200x4.sh`
- [ ] Clean re-run Phase 3 from 5% `diagnostic_report.csv` → review summary → wire `.npy` soft targets into train
- [ ] Design ISUP-in-loss / 10-ep audit with Dr. Omar (see README)
- [ ] Phase 1 aux cls head after Phase 3 confirm

---

## 2026-07-23

- **05:22 PDT** | Submit 10k adj-0.15-only ablation | Fair complement to R4 (40k+g45) and teacher A (40k+adj0.15) | Job **`5260610`** `uni2_10k_adj015`; later **FAILED** (see Jul 26)
- **~22:15 PDT (Jul 22)** | Valid PANDA+ for R4 ep35 + ep61 | After BN fix | ep35 cancer **0.539**; ep61 **0.528**. Both below teacher A **0.554**
- **ISUP P/R check (chat)** | Trust Rule B/C given low derived precision | Rule B swaps are dual-grade (≠ false ISUP-1 pure 3+3). ISUP1 recall 77% / precision 54%. Always-predict-ISUP1 would only hit 16.4% overall (actual 53.9%)

### R4 snapshot (40k + g45 soft α=0.22)
| Item | Value |
|------|--------|
| Tag | `uni2_upernet_raw_h200x4_40k_g45soft_bf16` |
| Train job | `5247423` (4×H200) — done |
| Best val cancer | **0.7394 @ ep61** |
| PANDA+ | ep35 **0.539**; ep61 **0.528** |
| Teacher A (ref) | val 0.742 / PANDA+ 0.554 |

---

## 2026-07-22

- See git history / older bullets in `docs/` if needed; R4 finished; PANDA+ BN eval saga resolved with valid scores above.

---

## Template (copy for new days)

```md
## YYYY-MM-DD

- **HH:MM TZ** | What | Why | Result / job id / paths
```
