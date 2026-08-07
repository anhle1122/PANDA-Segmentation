# Daily progress — PANDA Gleason project

**How to use:** Open this file each day (also mirrored at `docs/DAILY_PROGRESS.md`). Newest date at the top.  
Agents auto-append when training/eval/protocol work lands (see `.cursor/rules/daily-progress.mdc`).

Format per bullet: `- HH:MM TZ | What | Why | Result / next`

---

## 2026-08-07

- **12:33 PDT** | Duplicate scan **DONE** (`5382352`, ~11 min, CPU `cp011`) | Shape IoU on all 4683 clean slides | Fingerprints OK; user pairs recovered (IoU 0.86/0.43/0.34). **Silhouette IoU too weak** for needle cores → 5.3k “high” pairs / giant gleason-chained clusters (not actionable as true dups). Artifacts: `outputs/docs/slide_duplicates/` + `README.md` caveat. Next: stronger matcher (pHash/ORB + dim gate + review) before dedupe; then ISUP sufficiency on kept bags.

- **12:07 PDT** | Full WSI **duplicate scan** submitted | User found same cores under different `image_id`s (leakage risk) | Job **`5382339`** FAILED instantly (`SyntaxError` global). Fixed → resubmit **`5382352`** on `defq` CPU node `esplhpc-cp011` (~7 slides/s fingerprint → **~10–15 min** fingerprints + match). Not GPU. After: ISUP sufficiency check on kept bags (esp. 5≤n&lt;32).

## 2026-08-06

- **14:05 PDT** | v33 patch-filter audit: **20% RGB-tissue gate is stain-biased** | Explain slide `39f36065811d` keeping only 3 patches | Gate reproduces that slide 100% at `tissue_pct>=20`; 125/128 dropped `low_tissue` though **73 contain cancer** (max **35.7%** vs best kept 27.6%). Cause: `tissue_pct` counts a pixel as glass if `all(RGB>210)` → pale H&E leaves only dark nuclei (tile mean RGB 223–230). Corpus scan (4731 slides, 776,709 tiles, 71.6% kept): only **24** slides drop ≥50% as `low_tissue`, but **3446 (72.8%)** lose ≥1 cancer tile (**32,465** tiles). Artifacts: `outputs/docs/lowtissue_drop_scan_v33.csv`, `outputs/docs/lowtissue_worst_slides_contact.png`, `outputs/docs/min_slide_39f36065_hires/`. **Blocker:** generator `src/patch_filtering_panda.py` absent from repo+git (Jul-30 wipe) — thresholds not re-runnable without a rewrite.

- **12:57 PDT** | Omar protocol: **thr=0** + tiny-bag skip **n&lt;5** | Remove 5% derive_grade gate; only skip dual-ISUP on very small bags | `derive_grade` default `min_area_pct=0.0`; Opt3 `--min-slide-patches 5` hard-skips `L_slide`/`L_grade` (pixel loss still runs). Soft `L_slide` never used 5% for grads. Next: re-score ISUP @ thr=0; new Opt3 tag with these defaults.

## 2026-08-05

- **21:13 PDT** | Wrote Opt3 epoch/PANDA+ README | Document all ep1–30 + available PANDA+ | `outputs/docs/OPT3_EPOCH_RESULTS_README.md` (+ ckpt dir `README.md`). Notes unstable val (big dips ep10/15); ep12 best PANDA+ 0.551 vs ep21 val-best 0.636 / PANDA+ 0.546.

- **17:20 PDT** | Part2 **DONE** (both jobs) | Gate confidence-gating / LSE before train | **5368614** LSE OK (~14G). Cancer low-conf **~50%**; LSE lowers G5 share (plain 22.6% → r=8 **18.9%**), plateaus ≥8. **5368615** A/B: A (ISUP-mismatch cancer preds) low-conf **51.0%** (borderline noise); B (pred≠mask) low-conf only **19.9%** → mostly **confident** disagreement with expert. Single-ckpt / no ep12. Confidence-gating still **not** auto-wired — needs Omar call. Artifacts: `part2_ep21_only/part2_ep21_only_summary.json`, `part2_lse_precheck/lse_r_precheck_summary.json`.

- **15:56 PDT** | Part2 resubmit **128G + streaming** | Fix 64G OOM from full-pixel conf/prob buffers | Scripts stream hist/sum/count + online LSE logsumexp (no pixel buffers). Jobs **5368614** (`part2_lse`) + **5368615** (`part2_ep21`); workers=2. Out dirs same as before.

- **15:52 PDT** | Part2 jobs both **OUT_OF_MEMORY** @64G | Peak RAM after near-full inference (logit/conf accum) | **5368457** died ~2720/2750 (partial `mismatch_slides_streamed_isup.csv` only; no A/B conf summary). **5368439** finished batches then OOM — wrote `ep21_confidence_summary.json` (low-conf all **5.5%**, cancer pixels low-conf **~50%**, mean cancer conf **0.68**) but **no LSE r-curve**. Next: resubmit ≥96–128G and/or stream+free tensors; gate report still blocked.

- **14:58 PDT** | Modified Part2 submitted (ep21-only) **5368457** | ep12 unrecoverable; weaker single-ckpt diagnosis | A) conf of pred G3/G4/G5 on clinical-ISUP mismatch slides; B) conf where pred≠expert mask. Explicitly NOT ep12-vs-ep21. Out `outputs/pseudo_label/part2_ep21_only/`. Prior LSE job **5368439** still RUNNING.

- **14:57 PDT** | Part2/LSE path started; thr stays **0.05** | Aggregation progress not blocked by min_area edge | Offline: **871/3746 (23.3%)** slides with ep12≠ep21 derived ISUP; median max|Δcancer_frac| **0.15** (only 12% <0.05) — leans systematic, not tiny nudges. ep12 ckpt still gone → differing-pixel Part2 **partial**. GPU job **5368439** (`part2_lse`): ep21 conf on 120 divergent slides + LSE r∈{{2,4,8,16,32}} on conf>0.7 pixels. Out: `outputs/pseudo_label/part2_lse_precheck/`. Confidence-gating **not** wired to train until gate reports.

- **14:53 PDT** | Visualized 15 flip secondaries (0.05→0.01 match) | Check if 1% patterns are real tissue vs noise | Contact `outputs/pseudo_label/flip_secondary_vis/`. Sampled ISUP2/3/5 × tiny(~1%) & larger(~5%). Visual read: **cohesive, morphologically distinct islands** (G4 fused glands / discrete G3 / solid G5 nests), not scattered mask speckles. Supports small-but-real secondary; still does not alone justify locking thr=0.01.

- **14:30 PDT** | Extended min_area sweep + flip audit; ep12 confirmed gone | Check degenerate→0 before locking 0.01 | Match keeps climbing: **0.001→90.6%**, 0.005→89.1%, 0.01→87.8% (still monotonic). 0.05→0.01 flips on ISUP2/3/5: **222 wrong→right, 0 right→wrong**; 100% have a grade in [1%,5%); median 2nd-pattern **2.6%**. Rule1 2↔3 swap count 261@5% → 283@1%. **Do not lock 0.01.** ep12 `epoch_012…0.6190.pth`: deleted by prune `unlink` — not in trash/archive; other `epoch_012` files are different runs. CSVs: `mask_isup_min_area_pct_sweep_extended*` + `mask_isup_flips_0.05_to_0.01_*`.

- **14:18 PDT** | Part1 min_area_pct sweep DONE (mask vs clinical, PANDA train) | Calibrate threshold before aggregation fixes | Curve: 1%→**87.8%**, 2% 85.7%, 3% 84.0%, **5% 81.5%**, 7% 79.6%, 10% 76.5%, 15% 72.8% (n=3746). Best=**0.01**; sensitive mainly on ISUP2/3/5. Artifacts: `mask_isup_min_area_pct_sweep.csv` + `_summary.json`. **STOP before Part2.** PANDA+ clinical independence still blocked (Della Corte).

- **14:16 PDT** | Disable auto checkpoint prune (keep all epoch_*.pth) | ep12 was deleted by keep=3 mtime prune | `prune_checkpoints(keep<=0)` no-ops; Opt3 `--keep-checkpoints` default **0**. Never silent-delete epoch ckpts again unless explicitly set.

- **14:15 PDT** | Committed Opt3 eval stack `b3e28a3`; push blocked | Survive disk wipe / git restore | Added `lora_vit.py` + GN/LoRA load in `evaluate`/`isup_diagnostic`/`uni2_upernet` + `slurm_isup_diagnostic_panda_plus.sh`. Restored wiped tracked docs/scripts from HEAD. Push failed: no GitHub creds on login node (HTTPS prompt / SSH no key). Still **ahead 23** of origin — push from authed machine.

- **14:09 PDT** | TeacherA PANDA+ ISUP done; Opt3 ep21 FAILED→resubmit | First external ISUP check | TeacherA **15/48=31.2%** but **all 48 derived as 4+3 (ISUP3)** — collapse, not real match. Opt3 **5367374** NameError LoRA import; fixed; resubmit **5367914**. Treat 31.2% as invalid until paint-mode verified.

- **12:21 PDT** | PANDA+ ISUP match jobs submitted (first time) | Never scored model paint→ISUP on PANDA+ before; compare with PANDA+ Dice | Split `outputs/splits/panda_plus_isup.csv` (48 slides / 4688 patches; clinical=`train.csv`). Jobs **5367373** TeacherA, **5367374** Opt3 ep21. Outs: `diagnostic_report_panda_plus_{teacherA_ep042,opt3_ep21}.csv`. **ep12 ckpt pruned** — cannot re-run; keep prior PANDA+ Dice 0.551 only. Standing rule: prefer PANDA+ over val; ep12 still preferred over ep21 on Dice.
- **12:21 PDT** | Flag Model-C pattern on Opt3 run | ep21 val↑/ISUP↑ but PANDA+↓ vs ep12 | Explicit: ep12 val 0.619 / PANDA+ **0.551** / ISUP(in-domain) 52.6% vs ep21 0.636 / **0.546** / 57.3%. Checkpoint pick for this run → **ep12** (if recovered) else report both; do not default to val-best.
- **12:21 PDT** | Lock next Opt3: hard skip n<32 for \(L_\mathrm{slide}/L_\mathrm{grade}\) | Omar: soft min(1,n/64) ≠ “shouldn't drive” | Policy A queued for next tag (not applied mid-run). Open Q: whether gate shifts toward ep12-like (PANDA+) vs ep21-like (internal) — needs ablation after gate lands.

## 2026-08-02

- **13:03 PDT** | Queued **4×H200** hold **5346813** (`opt3_h200_4q`) | Grab full node or same-node when other 2 free | PENDING; keeps 2× **5343680** training. On start: cancel 2× → resume `latest.pth`. Monitor cancels 2× if `cp098` free≥2 so 4× can pack. Script `scripts/slurm_opt3_h200_4queue.sh`.
- **12:50 PDT** | Opt3 **5343680** healthy — **ep7 done**, ep8 train 256/256 (val) | Check after overnight H200 | Best **ep5 cancer Dice 0.604** (`best.pth`); ep6–7 dipped as λ_slide warms (0.06→0.12). Wall **~1.85 h/ep** (ep6 6744s, ep7 6646s). RSS flat ~5.94G. Job 9h+ on `cp098` 2×H200. Next: watch whether cancer recovers as λ→0.3; PANDA+ after `best.pth` stabilizes.

## 2026-08-01

- **20:51 PDT** | H200 handoff live — **5343680** 2×H200 `cp098` | Queue finally got GPUs; cancelled A100 **5343825** @~192 slides | Resumed `latest.pth` skip 192; now **~202/256** ep1 @ ~22 s/slide (~0.3h to ep1 train end). Mid-epoch ckpt path worked. RSS flat ~5.94G; CUDA peak ~29G.
- **18:37 PDT** | A100 **5343678** FAILED NCCL @~slide 14; fixed DDP desync; restarted **5343822** | Variable micro-batch backwards desynced ranks (same bug class as Jul 30) | `no_sync` on non-last micros; empty-bag dummy syncs; ckpt every **8** slides. H200 queue **5343680** still PENDING.
- **17:18 PDT** | Queued **2×H200** hold **5343680** (`opt3_h200_q`) | Take over when H200 frees; A100 keeps training | On start: cancel A100 → resume `latest.pth`. Watch log `logs/opt3_h200_watch.log` every 2m.
- **17:17 PDT** | Mid-epoch ckpt every 16 slides (preempt safety) | Lost ~95 slides twice with no `latest.pth` | Atomic `latest.pth` + resume skip `slides_done`. Restarted job with protection (cheap; was only ~2 slides in).
- **17:14 PDT** | Opt3 preempted; switch to fastest free = **2×A100** | 4×A100 PENDING(Resources); H100/H200 none usable | **5343544** cancelled after preemption @~slide 95 (no ckpt). Resubmit **2×A100** — same wall-clock as 4× with per-rank 256. H200 migrate monitor still armed.
- **15:27 PDT** | Upgraded Opt3 → **4×A100**; armed H200 migrate | Only ~12 slides on 2×, no ckpt — cheap cold restart | Cancelled **5343536**. **5343544** RUNNING 4×A100 `cp040` (200G). Monitor `monitor_opt3_upgrade.sh` every 3m: when H200 frees + `latest.pth` → migrate **4×** prefer else **2×** resume. RSS sampler + mem/LoRA CSVs continue.
- **15:16 PDT** | Locked live=64; restarted Option 3 **5343536** | Omar cleared bar; all 6 fixes + mem/LoRA watches | Cold start tag `pseudo_r1_opt3_slidebag`, **2×A100** `cp058` (H200 unavailable). live=64+ckpt, GN, LoRA, gates, λ warmup, extent soft-ISUP. Timelines: `opt3_mem_timeline_*.csv`, `opt3_lora_grad_*.csv`, `logs/opt3_rss_timeline.csv` (30s sampler). Keep eye on early LoRA grad mags vs loss.
- **14:27 PDT** | Longer live=64 smoke **5343397** COMPLETED (22m) | Omar: numeric LoRA grads + mem flat over time before lock 64 | LoRA step2 A/B abs-mean >0 (`omar_fix4_lora_grads.json`). Timeline 48 slides+val: host RSS **5.93→5.93** (creep **+0**), CUDA peak **~27.5G**, Slurm MaxRSS **~8.9G**/96G. **OK to lock live=64** — say go to restart Option 3.
- **14:05 PDT** | Omar asks for fuller Fix4 + Fix5b checks before locking 64 | Silently-not-training class + peak-mem≠leak-over-time | Verify dumps numeric `lora_A/B.grad`; longer live=64 smoke **5343397** (48 slides + val).
- **13:20 PDT** | Omar GPU verify **5343281** COMPLETED (all fixes) | Unblock live-size pick + train restart | Fix1–4/5a/6 **PASS**. Fix5b same-pass: live16 → CUDA **25.0G** / host **6.3G** / **2.5s**; live64 → CUDA **26.0G** / host **6.3G** / **8.7s**. Both far under 96G; **64 looks comfortable** (your call). JSON `outputs/omar_fix5b_live16_vs_64.json`.
- **13:19 PDT** | Fix LoRA+freeze bug; resubmit verify **5343281** | Frozen `no_grad`/`detach` killed LoRA grads; also A.grad≈0 at B=0 init | **5343200** FAILED mid Fix4. Patched `uni2_upernet` (skip no_grad when LoRA trainable) + assert on B.grad. **5343281** RUNNING on A100 `cp040`.
- **13:07 PDT** | Omar GPU verify moved off H200 → **A100** | H200 nodes unavailable; smoke only needs ~5–15 min once scheduled | Cancelled stuck **5343087**. **5343200** RUNNING on `cp040` (`gpu:a100:1`, 96G): Fix4/5a + dual 5b (live 16 vs 64).
- **12:51 PDT** | Fix 5b smoke → both live=16 and 64 in one pass | Prefer largest live with real headroom under 96G (64 if under ~75–80G; fall back to 16 if ~≥85G) | Cancelled pending **5343074** (32G, 5a-only). Resubmitted **5343087** (`omar_verify`, 96G, H200): Fix4/5a + same-pass 16 vs 64 peak CUDA / host RSS / time/step. No full train until you pick live size.
- **12:50 PDT** | Implementing Omar 6-point Option 3 review | Gate ISUP by n_patches; λ_slide warmup; decoder GN; LoRA not full unfreeze; proj outside no_grad; live=16+ckpt; soft ISUP extent | Pre-commit `8b69fcf`. CPU verify **Fix1/2/3/6 PASS**. GPU verify job **5343074** for Fix4/5a. No full train until all checks reported.
- **11:11 PDT** | Status check: Option 3 dead; mask-vs-clinical done | Catch up after Jul 30 evening | **5326090** FAILED @53m (NCCL ALLREDUCE timeout / rank desync); no ep1 row, no ckpt. Host RSS stayed **~10–13G** (mem fix OK). Mask check: **3054/3746 = 81.5%** vs model **53.9%** — not a coincidence. Nothing queued. Next: resubmit Opt3 (investigate DDP hang) + write mask number into protocol.

---

## 2026-07-30

- **19:05 PDT** | Mask-vs-clinical finished (**5326054**) | Fresh raw-mask `derive_grade` @5% | **3054/3746 = 81.5%** match (`mask_isup_vs_clinical.csv`). Model-vs-clinical stays **2018/3746 = 53.9%**. Per-ISUP mask match: 0=100%, 1≈100%, 2≈55%, 3≈54%, 4≈96%, 5≈77%.
- **19:07 PDT** | Option 3 **5326090** FAILED @53m | NCCL watchdog ALLREDUCE timeout 600s (rank desync); exit -6 | Never finished ep1; `training_log` header-only; no `latest.pth`. MaxRSS peaked **~13G**/96G (flat — cache fix held).
- **18:13 PDT** | Option 3 **5326090** full mem fix (after **5326082** SyntaxError) | Accidental one-line merge on val clear insert | Fixed compile; LRU + train/val clear. Cancelled **5326064** earlier (no ckpt). Watch RSS through ep1 **val** + ep1 wall time.
- **18:12 PDT** | Option 3 restart **full** mem fix (cancel **5326064**) | Train-only clear missed val workers (separate `val_ds` + DataLoader copies) | Class LRU `max_cached_opens=2` + `val_ds.clear_open_handles()` after val. Resubmit cold start; watch RSS through **full ep1 incl. val**.
- **18:04 PDT** | Option 3 host-RSS leak: cancel **5324260**, resubmit **5326064** | MaxRSS rose ~1.2G/30s (→67G@51m) from uncapped H5/OpenSlide caches per new slide | `BaselinePatchDataset.clear_open_handles()` after each bag. No `latest.pth` → cold start. Watch `opt3_rss_timeline.csv`.
- **17:59 PDT** | **Corrected:** 53.9% is model-vs-clinical, not mask | Identical 2018/3746 to teacher A diagnostic — mislabeled in chat | Confirmed `diagnostic_report.csv` uses `pred_pixels_*`. Submitted fresh mask check **5326054** (`slurm_mask_isup_vs_clinical.sh` → `mask_isup_vs_clinical.csv`). Protocol note updated.
- **17:57 PDT** | Armed Option 3 MaxRSS timeline vs val/`latest.pth` | Prior 64G eager OOM; don’t treat mid-epoch ~27G as proof | Sampler `scripts/sample_opt3_rss.sh` → `logs/opt3_rss_timeline.csv` every 30s; flags `OPT3_RSS_EVENT` on log-row / ckpt mtime / ≥72G. Snapshot @43m: MaxRSS **~50G**/96G, still ep1 (0 log rows).
- **17:15 PDT** | Option 3 **5324260** RUNNING 2×H200 on `cp098` | Queue cleared after long PENDING | Cold start (no `latest.pth`); monitor keeps 2× unless full node frees for 4× upgrade. Tag `pseudo_r1_opt3_slidebag`.
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
- [x] Mask-vs-clinical confirmed: **81.5%** (≠ model 53.9%)
- [ ] Diagnose Option 3 DDP/NCCL hang and resubmit (**5326090** failed @53m; no ckpt)
- [ ] Await PANDA+ eval on `pseudo_r1_isup_wmfix/best.pth` (resubmit if needed)
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
