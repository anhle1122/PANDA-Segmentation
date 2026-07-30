# Daily progress — PANDA Gleason project

**How to use:** Open this file each day (also mirrored at `docs/DAILY_PROGRESS.md`). Newest date at the top.  
Agents auto-append when training/eval/protocol work lands (see `.cursor/rules/daily-progress.mdc`).

Format per bullet: `- HH:MM TZ | What | Why | Result / next`

---

## 2026-07-30

- **14:38 PDT** | Option 3 lazy bags + mem/worker retune | Eager full-slide stacks caused 64G OOM; 200G wouldn’t schedule | Dataset returns indices only; micro-batch load on the fly. Slurm: **96G**, workers=4, max 160 patches/slide. Resubmit after cancel pending.
- **14:35 PDT** | Documented WeGleNet LSE as next \(L_\mathrm{slide}\) upgrade | Our linear soft-ISUP ≠ published LSE (\(r=8\)) / secondary damp \(d=0.70\) | Keep current run; after PANDA+ + leak, replace mean+linear with tunable LSE (+ optional \(d\)). Stub `aggregate_logsumexp_logits` exists but unwired / incomplete. Protocol updated.
- **14:33 PDT** | Option 3 mem/sched fix: 110G + max 128 patches/slide | 200G couldn’t land on cp098 (FreeMem ~118G) despite 2 free H200s | Cancelled 200G pending; **5324107** @ 110G + `--max-patches-per-slide 128` (median was 119).
- **14:28 PDT** | Option 3 **5323073** OOM-killed @ 41min; resubmit | 64G too small (slide bags + val); SIGKILL/oom_kill; no `latest.pth` | Cold restart path; later retuned to 110G+128 cap.
- **14:00 PDT** | Flagged Option 3 core hypothesis: mask CE vs \(L_\mathrm{slide}\) tension | Original masks uncorrected; CE@1.0 can reinforce G3/G4 bias while slide@0.3 pulls aggregate ISUP | Documented in `OMAR_ROUND_PROTOCOL.md`. Post-run: teacher-A-style G3→G4 leak analysis (not Dice alone). \(L_\mathrm{slide}\) grads *do* reach seg pixels — open Q is whether λ=0.3 wins.
- **13:43 PDT** | Option 3 BN N=1 crash fix + resubmit 2×H200 | Remainder micro-batch size 1 broke train BN | Pad singleton→2 (loss on real); resubmit preemptable. Prior **5323068** failed @ first steps. 3‑min monitor armed. ETA ~20–40 min/ep → **~1.5–3 days**/100 ep.
- **13:40 PDT** | Option 3 started 2×H200 on cp098 (preemptable) | `gpu`+normal → ReqNodeNotAvail on all H200 | Job **5323068** (later crashed BN); need preemptable QOS; auto-resume `latest.pth`.
- **13:37 PDT** | Option 3 → **2×H200** + mem fix + auto-resume monitor | 4× pending stuck; shared-node mem too high at first | Cancelled prior 4× pending; scripts: `monitor_opt3_h200.sh`, Slurm auto-resume.
- **13:29 PDT** | Option 3 α→**0.15**; cancel/resubmit vs teacher A | Not comparing to `wmfix` now — drop 0.22 confound | Cancelled **5322322**; resubmitted with adj soft 0.15 (teacher A). Frame: Option 3 vs A (PANDA+ 0.554 / G5 0.528). Rules-vs-Opt3 later needs same-α pair. Protocol + Slurm/train defaults updated.
- **13:06 PDT** | Documented Option 3 result-attribution caveat | Looser \(L_\mathrm{slide}\) ≠ `derive_grade()` — avoid misreading a null/negative PANDA+ | If underperforms: try soft-sort/soft-threshold upgrade **before** concluding dual ISUP doesn't help. Train/Slurm already in git (`9d282e8`). Protocol: `OMAR_ROUND_PROTOCOL.md`
- **12:55 PDT** | Full smoke re-validated recovered tree + started Option 3 | Confirm latest validated Rules stack before slide-bag work | Manifest **62/187** of 249; smoke **6/6 PASSED**; OEEM flagged weight=1.0 OK; session git reminder committed; PANDA+ eval job **5322295**; Option 3 train **5322322** tag `pseudo_r1_opt3_slidebag` (λ=0.3/0.3)
- **12:35 PDT** | Recovered wiped pseudo-label stack from Cursor local history + **committed** `e9add14` | Files were never in git and got deleted from disk | Restored `pseudo_label_rules/dataset`, `round_control`, ISUP-informed `losses`, `train_uni2_upernet`, Slurm/smoke/cache, Omar protocol docs. Stubs deleted. **Option 3 still blocked until smoke re-check; commit early going forward**

### Open / next
- [x] Smoke re-validate recovered Rules stack (6/6 + 62/187 + OEEM flag=1.0)
- [x] Document soft-ISUP caveat + winner-then-combine framework
- [ ] Await PANDA+ eval on `pseudo_r1_isup_wmfix/best.pth` (resubmit **5322326** if needed)
- [ ] Await Option 3 train (2×H200, α=0.15; monitor may upgrade→4× with resume)
- [ ] `git push` when ready

---


## 2026-07-29

- **16:32 PDT** | Drafted Omar clarifying Q on grade vs seg slide loss | Disambiguate A/B/C before implement | Options: grade=derived-from-seg vs own classifier vs both (seg gets derived-ISUP loss + grade gets feature classifier)
- **16:10 PDT** | Omar loss split clarified | Seg vs grade supervision | **Seg head:** CE+Dice **+ slide-level loss**; **Grade head:** grade loss only. Still not dual pixel-pseudo. Updated `OMAR_ROUND_PROTOCOL.md`
- **14:10 PDT** | Omar protocol clarification recorded | Correct round-based A; reject mid-run B | Mask regenerates 1×/round; live honesty = **grade head/loss** every batch; one cleaned seg target (gate→ISUP swap); no fighting dual pixel loss; ≤2 rounds. Docs: `OMAR_ROUND_PROTOCOL.md`; EMA design marked **REJECTED**
- **11:58 PDT** | Documented EMA + ISUP within-run design | Future path to refresh targets mid-run without discrete rounds | Later **rejected by Omar** — see 14:10; archived at `EMA_ISUP_WITHIN_RUN_DESIGN.md`
- **11:52 PDT** | Confirmed `5307779` protocol isolation | Avoid confusing with buggy `5307347` | **Clean Round 1 baseline:** `seg_target=original_mask` only; ISUP direct target edit on **442** slides (`soft_tie` 62 + R2 9 + R3 371); **187** `wide_margin_unresolved` = no rewrite; **OEEM not wired**. Tag `pseudo_r1_isup_wmfix` ≠ `pseudo_r1_isup_seg` (5307347, cancelled @ ep12, used hard wide-margin rewrite)
- **11:22 PDT** | Wide-margin Rule 1 → `wide_margin_unresolved` (no pixel rewrite) | No reliable pixel signal for real vs over-extended G3/G4; hard-correct erased true secondary | Cancelled `5307347` @ **ep12** (best cancer **0.570**); regen manifest **442** correcting / **187** unresolved; resubmit **`5307779`** tag `pseudo_r1_isup_wmfix`
- **10:42 PDT** | Confirmed flagged→weight=1.0 rationale in code | Avoid Rules 1–3 vs OEEM fighting on same pixel | Comments on `oeem_weight_map_for_unflagged` updated. Real pixel-count probe (320 patches): when G5 present, n≥1024 always in sample (0% with n<32); raised default `min_pixels` **2→8**. Real OEEM *weight* means still need GPU (QOS-blocked behind `5307347`)
- **10:28 PDT** | Researched OEEM exact formula from `vendor/OEEM` | Candidate within-round noise downweight vs `bias_too_heavy` | Real formula is \(\mathrm{softmax}(-\mathrm{CE})/\mathrm{mean}\) (Eq.6), **not** max-confidence. Notes: `outputs/pseudo_label/OEEM_FORMULA_NOTES.md`
- **10:12 PDT** | Tightened `G3_G4_LEAK_TOLERANCE` 1.10 → **1.05** | Full-weight ISUP edits make uncaught bias more consequential than under 0.70/0.30 | cancer/g5 still trip on any decline; docstring states full-weight (not diluted) explicitly
- **10:02 PDT** | Confirmed bias-fallback survives single-loss redesign | Full-weight ISUP edits | `apply_bias_fallback` kept; eval reports `g5_dice` + leak
- **09:09 PDT** | BN-fix resubmit `5307259` | Failed run BN mismatch | Got to ep5 cancer **0.396**, unfreeze ep6 val finite — better than NaN run, superseded by loss redesign
- **08:55 PDT** | Cancel `5305879` | val_loss NaN / cancer collapse | Architecture BN issue confirmed

### Open / next
- [ ] Let `5307779` finish as Round 1 mask-start Rules baseline (no grade head yet)
- [ ] Fresh `--panda-plus-eval` on best; compare teacher A; `bias_too_heavy` gate
- [ ] **Next design (Omar):** add live ISUP **grade head/loss**; Round 2 cleaned-target builder (agree→keep; low-conf disagree→ignore; conf disagree→ISUP referee); one seg loss; ≤2 rounds
- [ ] Do **not** build EMA mid-run relabel (B) or fighting dual pixel losses

---

## 2026-07-28

- **23:17 PDT** | Submit Round 1 pseudo-label train | Rules 1–3 + cache ready | Job **`5305879`** 2×H200; peak **0.411 @ ep3**; failed after unfreeze (see Jul 29)
- **~22:30 PDT** | Source-pred cache + PART 8 smoke | Needed before Round 1 | 629 slides / 1.2G; 6/6 smoke PASSED
- **Rule gate note** | Pure-pattern → Rule 3 not Rule 2 | R2=9, R3=371, R1=249

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
