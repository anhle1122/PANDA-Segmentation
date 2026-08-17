# Experiment Log — PANDA Gleason Segmentation (Model C)

Living log of runs, changes, and decisions. **Update whenever a training/eval/protocol change lands.**  
Day-by-day Cursor log: `outputs/DAILY_PROGRESS.md` (resume text never goes here).

Format: **Why → What → Result → Decision**

---

## Teacher A PANDA+ epoch sweep (2026-08-16)

| | |
|--|--|
| **Why** | Only Teacher A ep042 was scored on PANDA+ (0.554). Live ep29 is 0.609 — need to know if any Teacher A epoch on disk beats it. |
| **What** | 13× 1×L40S `slurm_eval_uni2_panda_plus.sh` jobs **5445432–5445444** for ep5/10/…/60 periodics + ep35/41 named. Skip ep42 (done). Skip best/latest duplicates. |
| **Result** | Queued (Priority). Scorecard `outputs/docs/opt3_this_run/teacherA_panda_plus_sweep.csv`. |
| **Decision** | Compare cancer Dice vs live ep29 **0.609**. Disk only has every-5 (+035/041/042), not all 1–60 epochs. |


## r2 / λ015 wall-time (2026-08-16)

| | |
|--|--|
| **Why** | 3-day wall too short for early-peak + later PANDA+ compare. |
| **What** | `scontrol update` TimeLimit=9d on **5445430** (ok) and **5445276** (denied on RUNNING). |
| **Result** | λ015=9d. r2 remains 3d → ends **Aug 19 14:03 PDT** unless admin extends. |
| **Decision** | Do not scancel r2. If wall hits, submit resume on same tag with 9d rather than kill. |


## λ_slide=0.15 ablation (2026-08-16)

| | |
|--|--|
| **Why** | Live ep7 best early PANDA at λ≈0.12; full 0.3 correlates with later val soft. Test target λ=0.15 with standard warmup. |
| **What** | Job **5445430** tag `opt3_omar6_lambda015`, 2×H200, 3-day, Omar-6 locked stack (`live=64`, LoRA in AdamW, decoder ckpt), `LAMBDA_SLIDE=0.15`, warmup **on**. Cold start. Duplicate **5445429** cancel when controller up. |
| **Result** | PENDING. Wall raised to **9-00:00:00** via scontrol. Live **5443101** / r2 **5445276** not cancelled. |
| **Decision** | Parallel ablation only; keep locked default λ=0.3. Compare on PANDA+ once named epochs exist. |

## Per-epoch PANDA / PANDA+ eval + L_slide vs Dice (2026-08-16)

| | |
|--|--|
| **Why** | Train log has Dice/`L_slide` but not PANDA+ Dice, G5 precision, or ISUP match. Need those every epoch, and a plot of whether `L_slide` fights Dice/`L_seg`. |
| **What** | CPU watcher `scripts/slurm_watch_epoch_eval.sh` (`AUTO_SUBMIT=1`, new-epochs-only). Each new `epoch_*.pth` → 1× L40S `scripts/slurm_eval_opt3_epoch.sh`: PANDA ISUP + PANDA+ Dice + PANDA+ ISUP. PANDA val Dice stays the train-log 20k-patch number. Classifier `scripts/classify_lslide_vs_dice.py`. |
| **Result** | Watcher **5445285** R. ep10–27: 7 together, 7 mixed, 2 flat, **2 fighting (ep12, ep23)**. corr(ΔL_slide, ΔDice)=**−0.18**. Ep15 train-log step is both-improved, not the fight signature; G5 gaming still only visible on PANDA+ (ep7 0.569 / 62.5% vs ep15 0.496 / 52.1%). Commit `ee7fb97`. |
| **Decision** | Auto-eval from the next new named ckpt. Do not backfill ep19–27. Teacher packs stay detect-only. Do not scancel **5443101** / **5445276**. |

## Omar-6 locked recipe + decoder-chunk checkpoint (2026-08-16)

| | |
|--|--|
| **Why** | Omar 5b needs grads on 64 patches, not 4. Cat-then-one-backward kept 64 FPN graphs and OOMd. Live 5443101 silently used micro=4. |
| **What** | `torch.utils.checkpoint` per live decoder chunk (chunk=4). Fail-closed: abort if live_n==micro or LoRA not in AdamW. Default tag `opt3_omar6_locked`; refuse live tag. Cursor rule `.cursor/rules/opt3-omar6-recipe.mdc`. |
| **Result** | Tests OMAR6_WIRING_OK. Job **5445276** PENDING (Priority), tag `opt3_omar6_locked`. Live **5443101** not cancelled. Commit `5a23882`. |
| **Decision** | This is the default until the user says otherwise. |

## Omar-6 default-recipe audit (2026-08-16)

| | |
|--|--|
| **Why** | Lock a future default against Omar’s 6 + later add-ons, not against leftover flags. |
| **What** | Scored live **5443101** vs disk `c40efb7` vs dead r2 **5445233**. Wrote `outputs/docs/opt3_this_run/OMAR6_DEFAULT_RECIPE.md`. |
| **Result** | Lockable now: n&lt;5 skip, λ_slide warmup, GN, freeze+LoRA (disk wiring), proj outside no_grad, absolute-area soft ISUP, grouped split, α=0.1+benign, min_area=0, no cap, every-epoch save. **Not lockable:** live=64/chunk=8 (OOM). **Trap:** slurm default tag is the live run + auto-resume. |
| **Decision** | User sets the default. Do not resubmit r2 until FPN memory is bounded. Do not scancel live. |

## Omar-6 freeze+LoRA is one script (2026-08-16)

| | |
|--|--|
| **Why** | Rule 4 sounded like we might have launched the old freeze-5 / unfreeze-at-ep6 trainer. |
| **What** | Opt3 has a single entry: `scripts/slurm_train_opt3_slidebag.sh` → `src/train_uni2_opt3_slidebag.py` (defaults freeze=100, lora=True). `src/train_uni2_upernet.py` freeze=5 is a different non-Opt3 trainer and was not used. |
| **Result** | Live **5443101** and r2 **5445233** both passed `--freeze-backbone-epochs 100 --lora`. UNI2 base frozen; LoRA QKV + decoder should train. Failure mode was LoRA not stepped (optimizer/`no_grad`), not the wrong script. **5445233** started 12:29 `cp097`, then CUDA OOM 12:57 at UPerNet FPN (~slide 56). |
| **Decision** | Do not scancel live. Do not treat r2 as “still pending.” Next r2 must use `c40efb7` and a tighter live-chunk/FPN bound. |

## Omar-6 LoRA / proj / soft-ISUP wiring (2026-08-16)

| | |
|--|--|
| **Why** | Agreed recipe was LoRA + proj outside `no_grad` + live=64/ckpt + absolute-area soft ISUP. Flags were on; LoRA was not in the optimizer and sat under `no_grad`; proj was inside that block; soft ISUP renormalized by cancer-only total. n&lt;5 hard skip was already correct. |
| **What** | Split backbone maps vs 1×1 proj. Skip backbone `no_grad` when LoRA is trainable. Put LoRA A/B in AdamW at the main LR while UNI2 stays frozen. Soft ISUP uses absolute p3/p4/p5; epoch log `soft_hard_isup_agree`. |
| **Result** | `scripts/test_omar6_wiring.py` OMAR6_WIRING_OK (1% vs 90% same ratio no longer share ISUP 1–5 logits). Live **5443101** unchanged. PENDING **5445233** will load this from disk at start. |
| **Decision** | Do not scancel live. Do not resubmit r2 unless it starts on old code (it should not — Python is read at runtime). |

## Live 5443101 launch-config diagnostic (2026-08-16)

| | |
|--|--|
| **Why** | Need the config the live process loaded at 13:37 PDT Aug 15, not current disk trainer. |
| **What** | Read-only parse of `logs/train_opt3_slidebag_5443101.out` + `training_log.csv` + val subsample. Wrote `outputs/docs/opt3_this_run/scorecard_lr_warmup.csv` (LR + λ_slide state beside Dice/L_slide for every epoch). |
| **Result** | LR = CosineAnnealingLR T_max=100, eta_min=1e-6, **no LR warmup**; ep19 = 9.14e-5 (91% of 1e-4). `--lambda-slide-warmup` is λ_slide only (0 ep1–5, full 0.3 from ep10). Pixel micro=4; ISUP live still **4** in-memory despite `--live-patches 64`. Val cancer Dice uses a **fixed** 20k-patch / 472-slide subset (seed 42 of 55,516). Ep7 PANDA cancer Dice 0.608 vs PANDA+ 0.587; PANDA+ ISUP 30/48 (62.5%); in-domain PANDA ISUP never scored. |
| **Decision** | ep18→ep19 Dice jump is a model swing, not val-set noise. Do not restart 5443101. |

## Omar-6 r2 live=64+ckpt resubmit (2026-08-16)

| | |
|--|--|
| **Why** | **5444968** OOMd on a single 64-patch UPerNet FPN cat. Aug 1 Opt3 used live=64 **with** `--grad-checkpoint`. |
| **What** | Backbone `set_grad_checkpointing(True)`. Live ISUP still n=64; forward in chunks of 8 so FPN never sees 64 at once; cat keeps grads on all 64. New tag `opt3_omar6_r2_live64ckpt`. |
| **Result** | Job **5445233** PENDING (Priority). Live **5443101** not cancelled. Commit `a7f2d23` on origin. |
| **Decision** | This is the r2 recipe. Do not run live=64 without ckpt/chunking. |

## Worktree wipe + origin push (2026-08-16)

| | |
|--|--|
| **Why** | ~120 tracked Opt3 sources vanished from disk at 12:04 PDT. Git still had them at `5030ee7`, but those 10 commits were not on origin. |
| **What** | Restored deleted tracked files from HEAD. Pushing `main` to origin. Checkpoints stay gitignored (`outputs/checkpoints/`, ~4.7G each). |
| **Result** | Trainer/scripts/docs back on disk. Named live ckpts ep5/10/15/19–26 still present. Live **5443101** never stopped. |
| **Decision** | Commit+push sources without being asked. Never git-add `*.pth`. Weights live on the HPC filesystem; git holds code and path/metric logs. |

## Omar-6 r2 FAILED CUDA OOM (2026-08-16)

| | |
|--|--|
| **Why** | Fresh tag to hunt a keepable early peak with every-epoch save. |
| **What** | **5444968** started 11:32 PDT on `cp097` (2×H200, live=64, same Omar-6 recipe, current code). |
| **Result** | FAILED 85s, exit 1:0. `OutOfMemoryError` at first `model(imgs_g)` FPN concat; 129G already allocated, needed +43G. Header-only `training_log.csv`. Live job uses old in-memory trainer and is still R. |
| **Decision** | Do not resubmit r2 until the live-bag / full-bag feat-chunk memory is reduced. Do not scancel **5443101**. |

## Omar-6 r2 wall cut to 3 days (2026-08-15)

| | |
|--|--|
| **Why** | 30-day r2 **5444968** was `ReqNodeNotAvail` on every H200 node; a 3-day early-peak hunt is the point of this tag. |
| **What** | `scontrol update JobId=5444968 TimeLimit=3-00:00:00`. Live **5443101** not changed. |
| **Result** | Reason became `None` (schedulable). Priority 17097→16964. Estimated start still Tue Aug 18 17:37. |
| **Decision** | Keep live job. Do not scancel to force a swap. Revisit if r2 is still PD after the Mon `cp097` hey3 free. |

## Detect-only watcher + teacher selector (2026-08-15)

| | |
|--|--|
| **Why** | Need a standing detect/select loop that does not cache baseline epochs or require terminal babysitting. |
| **What** | Replaced old watcher **5444967** with config-driven **5444985**, `AUTO_SUBMIT=0`, file log `outputs/pseudo_label/watcher_detect.log`. Selector `scripts/select_teacher_epoch.py`: val cancer ≥0.579, L_slide ≤ value 3 epochs prior, G5 precision within 0.03 of ep7 0.569370, and a surviving `epoch_XXX_*.pth`. Watcher re-runs the selector on every new DETECT. Ep15 pack stays on **5444966**; **5444986** (`afterok`) will move it to `validation_only/teacher_ep015/` and run referee with `--allow-validation-only`. Referee now refuses `VALIDATION_ONLY` packs by default and surfaces G5-swap vs original-mask G5 share (no block). |
| **Result** | Baseline frozen **ep5/10/15/19** (not enqueued). Selector: **NO_CANDIDATE**. Latest ep19 val cancer 0.545 (gap 0.034); L_slide 1.351 falling ok; G5 precision missing. Ep7 HISTORICAL_ONLY (weights gone). Ep15 keepable but G5 precision 0.496 is outside 0.03 of 0.569. Tests ALL_PASS. Commit `4519890`. |
| **Decision** | Do not cache a production teacher until the selector prints `CANDIDATE`. Give the live run through **ep22** before changing λ_slide. Eyeball G5-bias numbers on the ep15 validation correction before picking a gate. |

## Watcher + corrected-label wiring (2026-08-15)

| | |
|--|--|
| **Why** | Between-round loop was tools-only: watcher hardcoded one tag and train could not read referee h5. |
| **What** | Multi-target JSON watcher (new-epochs-only, single L40S queue, AUTO_SUBMIT default off). `RefereeCorrectedPatchDataset` + `apply_pixel_ignore`. Stub tests pass. Train Slurm / live watcher **not** switched yet. |
| **Result** | `scripts/test_corrected_label_source.py` ALL_PASS. |
| **Decision** | Replaced **5444967** with **5444985**. Keep AUTO_SUBMIT off until a selector candidate exists. |

## Omar-6 r2 fresh train queued (2026-08-15)

| | |
|--|--|
| **Why** | Live run lost named ep6/7 and is dropping; want a keepable early peak with every-epoch save. |
| **What** | New tag `opt3_omar6_r2_everyep`, cold start, same Omar-6 recipe. Job **5444968** 2×H200 `gpu`+`normal` 30d. Cancelled pending same-tag 4× **5443098** only. Live **5443101** untouched. |
| **Result** | **5444968** PENDING (Priority). Separate ckpt dir. |
| **Decision** | Two trains OK. Never write r2 into the live tag dir. Pick teacher by PANDA+ once r2 has named epochs. |

## Between-round teacher pack + three-way referee (2026-08-15)

| | |
|--|--|
| **Why** | Correction needs preds+confidence, not just weights. Ep7 had neither maps nor a surviving ckpt. |
| **What** | Per epoch: immutable `epoch_*.pth` + `teacher_<tag>_epXXX/` (`preds` uint8, `maxprob` float16, `clinical_isup.csv`, `pack_config.json` with λ_slide). Referee: agree→keep; disagree low-conf→ignore; disagree high-conf + illegal G3–5→nearest allowed. ISUP-0 skip. G5-swap share gate vs teacher G5 pred prior. Corrected h5 dated, never overwritten. Round N+1 is a scratch train on that target; accept on PANDA+. |
| **Result** | Code + Slurm landed. First cache = Omar-6 **ep15** on L40S (train keeps running). Watcher can submit a pack when sidecar writes a new epoch. |
| **Decision** | Do not wait for a scratch retrain to start the pack. Do not cache from dropping `latest.pth`. |

## Omar-6 pixel referee (proposed 2026-08-15)

| | |
|--|--|
| **Why** | Between-round pseudo labels should only rewrite pixels the teacher is unsure about and that ISUP forbids. |
| **What** | Teacher softmax+argmax vs expert mask. Flag if pred≠mask AND maxprob &lt; τ AND pred ∈ {3,4,5} ∉ clinical {primary, secondary}. Target = nearest allowed cancer grade. High-conf and legal-grade disagreements stay. ISUP-0 skip. |
| **Result** | Cannot run on ep7: no `epoch_007` / overwritten `best.pth`. Named files: ep5 / ep10 / ep15 only. Part2 B (older run): pred≠mask is **80% high-conf**, so this gate is conservative. |
| **Decision** | Lock this rule. Need a named teacher (ep15 stand-in or a later saved epoch that beats PANDA+) + a τ (default candidate 0.7 from Part2). Do not use dropping ep16–18. |

## Active defaults (current)

| Item | Value |
|------|--------|
| Architecture | UNI2-h + UPerNet |
| Input | raw 512² (ImageNet norm) |
| Loss | 0.5 CE + 0.5 soft Dice; adj soft **α=0.1** on **benign↔G3↔G4↔G5** (Omar 2026-08-11); optional legacy g45 / cancer-only via flags |
| Classes | 0 bg, 1 stroma, 2 benign, 3 G3, 4 G4, 5 G5 |
| **Live split** | Grouped fusion (2026-08-11): **4683** slides; train **3739** / val **472** / test **472**; groups = ledger twins ∪ UNI2 mutual-NN ∪ (rank2–5 + IoU≥0.29); `max-eval-group=2` (larger → train); **0** confirmed twin cross-split leaks |
| Eval scoring | Prefer **per-group** average on val/test (twins share one vote) |
| **Teacher / external report** | `…/h200x4/epoch_042_cancer_0.7420.pth` (PANDA+ cancer **0.554**) |
| PANDA+ protocol | `gt≥2` only; benign/G3/G4/G5; `--panda-plus-eval` |
| ISUP diagnostic `min_area_pct` | **0.0** (Omar 2026-08-06: remove 5% gate; any positive grade counts) |
| Opt3 tiny-bag gate | **hard skip L_slide/L_grade if n&lt;5** (Omar; was soft n/64 / planned 32) |
| Phase 3 ISUP-0 | **skip** |
| Handoff README | `/common/omarmlab/members/anh/panda_project/README.md` |
| Checkpoint prune | **off** (`--keep-checkpoints 0`); **every epoch** writes immutable `epoch_XXX_cancer_Y.pth` (never `--save-every 5`) |
| **Opt3 recipe** | **Omar-6 locked**: tag `opt3_omar6_grouped_soft01`; α=0.1 benign↔G3–G5; min_area=0; n≥5 skip; LoRA+GN; live=64; λ_slide warmup |

---

## Omar-6 — never wash out epoch ckpts (2026-08-15)

| | |
|--|--|
| **Why** | Resume reset in-memory best to -1; ep16 0.521 overwrote ep7 `best.pth`. Named snapshots were only every 5 epochs. |
| **What** | Every epoch writes immutable `epoch_XXX_cancer_Y.pth`. Resume restores best from `training_log.csv`. `latest.pth` atomic. Prune is a no-op. Live-run sidecar **5444924** (`scripts/slurm_preserve_opt3_ckpts.sh`) copies finished epochs off `latest.pth`. Rule: `.cursor/rules/opt3-checkpoints.mdc`. |
| **Result** | Live **5443101** still uses in-memory old saver; sidecar covers ep18+. Ep7/16/17 unique files cannot be recovered. |
| **Decision** | Keep this policy. Do not cancel 5443101 to load the new saver. |

## Omar-6 recipe + every-epoch ckpt locked (2026-08-15)

| | |
|--|--|
| **Why** | Disk wipes dropped uncommitted eval/train; every-5 saver lost ep6/ep7. |
| **What** | Locked Omar-6: α=0.1 benign↔G3–G5, min_area=0, n≥5, LoRA+GN, live=64, λ_slide warmup, save every epoch. Commit-on-run rule. |
| **Result** | Wiped `src/train/{losses,grade_head,slide_bag_dataset,...}` restored from HEAD. Omar-6 trainer + `--save-every 1` + commit-on-run committed. Live **5443101** still old in-memory saver; sidecar copies finished epochs. |
| **Decision** | This is the only Opt3 recipe. Never `--save-every 5`. Commit immediately before/after every submit. |

## Omar-6 Opt3 — ep15 PANDA+ (2026-08-15)

| | |
|--|--|
| **Why** | Ep7 weights in `best.pth` (val cancer 0.608; PANDA+ cancer 0.587; gland ISUP 77.1/62.5) were overwritten after resume reset the in-memory best. Best remaining snapshot is ep15 (val 0.579). |
| **What** | Same PANDA+ protocol as ep7: `gt≥2` Dice (**5444920**) + gland ISUP thr=0 pred-on-labeled (**5444921**), 1×L40S. Ckpt `epoch_015_cancer_0.5791.pth`. |
| **Result** | First submit 5444918/19 failed on missing src after wipe; resubmitted **5444920/21** 19:14 PDT. Live train **5443101** untouched. |
| **Decision** | Treat ep15 as the reportable Omar-6 snapshot until a later epoch beats 0.608 *and* is written to `epoch_*.pth`. |


## Checkpoint leaderboard

| ID | Stem | PANDA val cancer | PANDA+ cancer | Role |
|----|------|-----------------:|--------------:|------|
| **A** | `h200x4/epoch_042_cancer_0.7420` | **0.742** | **0.554** | Best external teacher |
| B | `10k_g45soft_fp32/epoch_072_cancer_0.7160` | 0.716 | 0.552 | Near-tie PANDA+ |
| C | `40k_bf16_from742/epoch_029_cancer_0.7521` | 0.752 | 0.537 | Val overfit |
| **R4** | `40k_g45soft_bf16/epoch_061_cancer_0.7394` | **0.739** | **0.528** | Fair 40k+g45; external below A |
| R4-ep35 | `40k_g45soft_bf16/epoch_035_cancer_0.7222` | 0.722 | **0.539** | Best R4 PANDA+ |
| R5 | `h200x4_10k_adj015` | — | — | Failed job `5260610`; resubmit after code restore |
| Opt3-ep12 | `opt3_slidebag/epoch_012…0.6190` (**pruned**) | 0.619 | **0.551** | Best Opt3 external so far; ISUP-in-domain 52.6% |
| Opt3-ep21 | `opt3_slidebag/epoch_021…0.6356` | **0.636** | 0.546 | Val/ISUP-in-domain better; **PANDA+ worse** (Model-C pattern) |

---

## Ablation honesty

A/B/C were sequential engineering runs (confounded). Cleaner grid:

| | adj 0.15 only | adj 0.15 + g45 0.22 |
|--|--|--|
| **40k** | A-like (messy history) | **R4** |
| **10k** | **R5** (pending resubmit) | **B** |

---

## Opt3 — grouped split + soft α=0.1+benign (2026-08-11; queue fix 2026-08-12)

| | |
|--|--|
| **Why** | Retrain under clean grouped split; Omar soft labels (benign+G3–G5, α=0.1); keep separate pix / L_slide / L_grade logs for ablation |
| **What** | `scripts/slurm_train_opt3_slidebag.sh` tag `opt3_grouped_soft01_benign`; `--min-slide-patches 5`; `--min-area-pct 0`; `--include-benign-soft`; λ_slide=λ_grade=0.3 |
| **Result** | First queue on **preemptable** (5412995→5413005/5416539): preempted, no ckpt. 2026-08-12 `gpu`+`normal` **5428051** 4×H200 `cp098`: ran **2h30m**, then **FAILED** — NCCL ALLREDUCE timeout 600s (rank1 desynced; ranks 0/2/3 waited). No ep1 / no `latest.pth`. DDP parity restored; resubmitted **5430882** 4×H200 `gpu` (ETA ~Thu 01:26 PDT). |
| **Decision** | Stay on `gpu`+normal. Omar soft chain already correct (G3 0.05/0.90/0.05). Prefer PANDA+ over val; score val/test per group. |

---

## Split — grouped fusion applied (2026-08-11)


| | |
|--|--|
| **Why** | Drop-era live set (3,831) still leaked risk; user signed off: keep all related slides together, large groups → train |
| **What** | `src/make_grouped_splits.py --use-fusion --max-eval-group 2 --apply`. Groups = ledger twins ∪ UNI2 mutual-NN ∪ (rank 2–5 + silhouette IoU ≥ 0.29). Patch CSVs rebuilt from `panda_*_pre_dedupe.csv`. |
| **Result** | Live train **3739** / val **472** / test **472** (4683 total). Val/test max group size **2**; train max **113**. Confirmed twin cross-split leaks **0**/850. Grade shares within ~0.4% of 80/10/10. |
| **Decision** | Live default is grouped fusion. Drop-era backups tagged `pre_grouped_fusion_iou0.29_rank2to5_maxeval2_2026-08-11`. Re-train/eval on this split; score val/test **per group**. |

---

## R4 — 40k + g45 soft (done)

| | |
|--|--|
| **Why** | Isolate g45 soft vs teacher A without 10k confound |
| **What** | 4×H200; 40k/ep; adj 0.15 + g45 0.22; BF16; job **5247423** |
| **Result** | Early stop ep81 (patience 20); best val **0.7394 @ ep61** |
| **PANDA+** | ep35 **0.539**; ep61 **0.528** |
| **Decision** | Do not promote over A (0.554) |

---

## R5 — 10k + adj 0.15 only (failed once)

| | |
|--|--|
| **Why** | Complete 10k × adj-only cell of the ablation grid |
| **What** | Job **5260610**; script `scripts/slurm_train_uni2_10k_adj015_h200x4.sh` |
| **Result** | Started on cp098 Jul 23 → **FAILED** (~9s) on corrupt/syntax-broken `train_uni2_upernet.py` then on disk |
| **Decision** | Code restored from git 2026-07-26; resubmit when ready |

---

## Code wipe / restore (2026-07-26)

| | |
|--|--|
| **Why** | Working tree deleted critical UNI2 train/eval sources (~5k LOC) while still in git |
| **What** | `git checkout HEAD -- src/ …`; restore Phase 3 script; add README |
| **Result** | Runnable sources back on disk |
| **Decision** | Commit immediately; prefer push so wipe cannot recur |

---

## Phase 3 / ISUP-in-loss (open with Omar)

1. Offline Rule B/C soft `.npy` from 5% diagnostic (`scripts/phase3_apply_corrections.py`)
2. Wire corrected soft targets into training dataset/loss
3. Optional: 10-ep ISUP diagnostic + alert loop (no silent mutations)
4. Re-check per-ISUP precision/recall after retrain (not match rate alone)

---

## Opt3 slide-bag (in progress) — Model-C pattern + PANDA+ ISUP

| | |
|--|--|
| **Why** | Dual ISUP losses; check whether clinical-match gains track external seg |
| **What** | Tag `pseudo_r1_opt3_slidebag`; soft gate `min(1,n/64)`; live=64; GN+LoRA |
| **Result** | ep12: val 0.619 / PANDA+ **0.551** / ISUP(train) 52.6%. ep21: val 0.636 / PANDA+ **0.546** / ISUP(train) 57.3%. **Same divergence as Model C: internal↑, PANDA+↓.** |
| **Decision** | Prefer **ep12** for external claims (ckpt currently pruned — keep Dice number). Next tag: **hard skip n&lt;32** for L_slide/L_grade (Policy A). First PANDA+ ISUP jobs **5367373** (Teacher A) / **5367374** (ep21) vs `train.csv` clinical on 48 matched slides. |

### Augmentation (2026-08-09)

| | |
|--|--|
| **Why** | Match Teacher A train aug; Opt3 bags need slide-consistent HED/flip/rot |
| **What** | Shared `augmentations.py`; Opt3 `--augment` default ON; UPerNet patch-level; preview picks patches by **mask non-bg + RGB variance** (not pen `tissue_pct`) |
| **Result** | Early previews were glass/ink (pen score). Mask-verified: `outputs/augmentation_examples/slide_and_patch_preview/two_slides_REAL_tissue_before_after.png` |
| **Decision** | Keep current HED±0.05 / flip / rot90; use mask-gated previews only |

---


## Part1 — min_area_pct calibration (mask vs clinical, PANDA train)

| | |
|--|--|
| **Why** | Threshold used by derive_grade / Rules / L_slide never validated vs clinical |
| **What** | Offline sweep on cached raw-mask pixel counts (3746 train slides); NOT model preds |
| **Result** | match@thr: 0.01→87.8%, 0.02→85.7%, 0.03→84.0%, **0.05→81.5%**, 0.07→79.6%, 0.10→76.5%, 0.15→72.8%. Best=0.01. ISUP0/1 flat ~100%; ISUP2/3/5 drop as thr rises. Max adjacent Δ≈3.7pp. Extended: still monotonic to 0.001→90.6%. Flips 0.05→0.01: 222 wrong→right, 0 right→wrong. |
| **Decision** | (superseded 2026-08-06) Omar removes area gate → **thr=0**. Prior caution against locking 0.01 still noted historically. |

## Part2 — ep21 confidence + LSE precheck (2026-08-05)

| | |
|--|--|
| **Why** | Gate confidence-gating / LSE before wiring into train; ep12 differing-pixel analysis blocked (ckpt pruned) |
| **What** | **5368457** ep21-only A/B conf on clinical-mismatch + pred≠mask; **5368439** ep21 conf + LSE r∈{2,4,8,16,32} on 120 divergent slides. Both `--mem=64G`. |
| **Result** | 64G OOM → fixed. **5368614**/ **5368615** COMPLETED @128G+stream. Conf: cancer low-conf **~50%**. A (ISUP-mismatch G3–5): low-conf **51%** / high **49%**. B (pred≠mask): low-conf **20%** / high **80%**. LSE: plain G5 **22.6%** → r=8 **18.9%** (Δ≈−3.7pp); plateaus after r≈8. |
| **Decision** | Mixed gate: A borderline; B confidently wrong vs expert. Conf-gating not auto-wired. |

## Omar protocol update (2026-08-06) — thr + tiny-bag gate

| | |
|--|--|
| **Why** | Omar: remove 5% area rule; only skip dual-ISUP on very small bags |
| **What** | `derive_grade` default `min_area_pct=0.0`. Opt3 `--min-slide-patches 5` hard-skips `L_slide`/`L_grade` when n&lt;5 (pixel loss still runs). Soft `L_slide` never used the 5% gate for grads. |
| **Result** | Code in `isup_diagnostic.py`, `grade_head.py`, `train_uni2_opt3_slidebag.py`. |
| **Decision** | Next runs / diagnostics use thr=0 and n&lt;5 skip. Re-score ISUP tables at thr=0 before citing new match rates. Prefer PANDA+ for ckpt pick. |

## v33 patch-filter audit (2026-08-06) — tissue gate is stain-biased

| | |
|--|--|
| **Why** | Slide `39f36065811d` (ISUP 5) reaches training with only **3** patches; need the cause before blaming bag size / L_slide |
| **What** | Recovered the v33 rule from 99,764 logged decisions (decision tree, 99.5% agreement) — generator `src/patch_filtering_panda.py` is **missing from repo and git history** (Jul-30 wipe). Order: pen/dark >2% → drop; `tissue_pct` &lt;20 → `low_tissue`; ≥20 &amp; cancer ≥1% → `rescued_cancer`; ≥20 &amp; G4/G5 near cutoff → `rescued_g45_near_cutoff`; ≥40, or ≥20 with mask tissue ≥40 → plain keep. `tissue_pct` fitted to `100*(1-mean(all(RGB>210) \| blank))`, no blur (mean abs err **0.32pp**). Corpus scan → `outputs/docs/lowtissue_drop_scan_v33.csv`. |
| **Result** | On that slide the gate reproduces all 128 decisions exactly (max dropped tissue 19.69% vs min kept 20.83%); **cancer plays no role** — 73 dropped tiles contain cancer, max **35.7%** vs best kept **27.6%**. Root cause is stain intensity, not tissue amount: tile mean RGB **223–230**, so pale eosinophilic stroma clears the 210 "white" cutoff and only dark nuclei count. Corpus (4731 slides / 776,709 tiles / **71.6%** kept): ≥50% `low_tissue` on **24** slides, ≥80% on **1**; **3446 slides (72.8%)** lose ≥1 tile with ≥1% cancer, **32,465** tiles total; dropped tile beats every kept tile on cancer for **3** slides. Both worst cases (`39f36065811d`, `6875cac943f8`) are ISUP 5. |
| **Decision** | Cancer rescue **is live** (v33 is the only tag; `DEFAULT_RULES_TAG="v33"`, splits built from it) — without it this slide would contribute 0 patches. Do **not** treat sparse-bag ISUP mismatch as purely a model/loss problem. Filter cannot be re-run until the generator is rewritten; if rewritten, replace the fixed RGB-210 cutoff with a stain-robust measure (Otsu on optical density) and re-check the 24 worst slides. |

## Open next

1. Re-score train / PANDA+ ISUP at `min_area_pct=0`  
2. Next Opt3 tag with n&lt;5 skip + thr=0  
3. Keep reporting val + PANDA+ + ISUP (not val-best alone)  
4. Commit prune keep-all + Omar thr/gate change  
5. Decide whether to rewrite `patch_filtering_panda.py` with a stain-robust tissue measure  
6. **Slide duplicate scan job `5382339`** — review pairs/clusters; if cross-split hits, dedupe before next train  


## Rescan-alive safe page1 (2026-08-09)

| | |
|--|--|
| **Why** | Second-pass survivors still show high-IoU safe pairs |
| **What** | User: P1+P3 twins; P2,P4–P10 not twins keep both |
| **Result** | Dropped `bfc5…`,`773d…`. Alive **3857** (3091/383/383). |
| **Decision** | Apply; log `slide_duplicates_rescan_alive/user_rescan_safe_page1_*` |


## Twin ledger — canonical dedupe truth (2026-08-09)

| | |
|--|--|
| **Why** | Confirmed twins kept reappearing: the Friday C35-55 bulk restore resurrected round-1 drops, and the reconcile classified them "SOFT / do-not-drop", so leak counts read 0 while the slides were alive |
| **What** | Proposal A applied (drop 15 / keep 7, user not-twin-of-KEEP marks propagated across twin pairs). Froze `CANONICAL_twin_drops.csv` + `CANONICAL_not_twins.csv` with precedence tiers: auto(1) < pair review(2) < multi review(3) < rescan era(4) |
| **Result** | Slides **3838** (3072/383/383). Ledger: 845 drops / 390 not-twins / 100 adjudicated not-twin **pairs**, verify **clean**. 845 matches the 4683→3838 delta exactly |
| **Decision** | `src/twin_ledger.py` is the source of truth. Run `--rebuild` after any review; `rescan_twins_alive.py` aborts if the ledger and splits disagree |

## Clean scan #3 — pair-level suppression (2026-08-09, job 5393362)

| | |
|--|--|
| **Why** | Scan #2 re-showed pairs the user had already judged not-twins, so the galleries could not be trusted as a to-do list |
| **What** | Added `CANONICAL_not_twin_pairs.csv` (the adjudicated **edge**, not the slide) and cut those edges *before* clustering, so a rejected edge can no longer glue a cluster together. Suppression is per-edge on purpose: a slide that is not the twin of A can still be the twin of B — that is how 68a↔8b126 was caught, and slide-level exclusion would have hidden it |
| **Result** | 96 judged edges cut. Safe **2** (both NEW), multi **35** clusters / 209 slides, lower **1833** (NEW 1789 / MIXED 43 / KEPT_NOT_TWIN 1). Out: `outputs/docs/slide_duplicates_scan3_clean/` |
| **Decision** | `slide_duplicates_scan3_clean` is the review folder going forward; `slide_duplicates_rescan_alive` is superseded. Log every not-twin call as a pair so it never comes back |

## Multi-cluster inner twins — the real dedupe gap (2026-08-09)

| | |
|--|--|
| **Why** | User read old gallery C1/C2/C3 and named 9 twin pairs still alive. Detection was never the problem: all 9 sit on IoU 0.715–0.859 edges the scan found |
| **What** | The C1–35 multi review adjudicated **whole clusters** (207 `keep_safe` vs 3 `drop_user_twin`), so a cluster judged "safe to keep" kept every twin pair inside it. Old C1: 26 members, 0 dropped, ever |
| **Result** | 7 drops applied (2 pairs were already resolved). Slides **3831** (3065/383/383); ledger 852 drops / 385 not-twins / 102 pairs, clean. 5 of the 7 had been marked `keep_safe` — tier 4 correctly overrode tier 3 |
| **Decision** | Cluster-level "keep" is not a dedupe decision. Every multi-cluster needs a pairwise pass; 32 of the 35 scan3 clusters (~183 slides) are still unreviewed at pair level |

## Grouped split replaces dropping (2026-08-10, job 5394252)

| | |
|--|--|
| **Why** | Dropping duplicates cost 852 slides and still resurrected pairs weekly. Grouping keeps every slide and just confines duplicates to one split, so a false merge is free and only a missed twin is expensive |
| **What** | Full 4683 UNI2 embed (67 min). Edges = 841 ledger twin pairs ∪ UNI2 rank-1 neighbours, minus 112 adjudicated not-twin pairs; connected components → groups; groups assigned whole to train/val/test, greedy large-first against per-grade quotas |
| **Result (signal)** | Absolute cosine useless at scale: median top-1 **0.977** vs twin median 0.976 in the 315-slide smoke pool; min over all 4683 is 0.833. Rank is clean: twins median rank **1** (80.1% rank-1, 89.8% ≤5) vs not-twins median rank **1282** (1.8% rank-1). Mutual-NN: 1605 pairs, 574 confirmed twins, **1** confirmed not-twin |
| **Result (split)** | **1474 groups**, largest 45, 0 confirmed twins split. train 3738 / val 472 / test 473, every ISUP grade within 0.3% of 80/10/10, 0 groups straddling splits. Rank-2 → largest group 4467; mutual-top3 → 1744; both collapse, rank-1 is the only usable setting |
| **Result (eval-set quality)** | First assignment was backwards: greedy filled val/test first while they sat furthest below quota, so the *largest* clusters landed there — test's biggest group was 38 slides, 8% of the set from one specimen. Pinning groups >2 to train gives train 3737 / val 472 (236 groups) / test 474 (237 groups), largest eval group 2, grades within 0.3%, 0 leaks |
| **Decision** | Adopt rank-1 grouping with `--max-eval-group 2`; restore all 4683 slides. Proposal lives in `outputs/docs/slide_groups/grouped_split_rank1.csv` and is **not** applied to `outputs/splits/` pending user sign-off. Residual 1: the ~10% of twin relationships at rank 2–5 are not caught and cannot be without chain collapse. Residual 2: test still holds 107 confirmed twin pairs internally (≤367 distinct specimens in 474 slides) — **report val/test metrics per group, not per slide**, so each specimen gets one vote |

## Dedupe signal search — shape → cheap descriptors → UNI2 (2026-08-09)

| | |
|--|--|
| **Why** | User found true twins at silhouette IoU 0.30–0.32, including a 3-cut serial block. Shape can't be the signal, and reviewing every flagged pair by hand is thousands of comparisons |
| **What** | Three signals benchmarked on 841 confirmed twins / 106 confirmed not-twins: grade-agnostic full 4683² shape IoU (job 5393604), phash + tissue colour-histogram (5393691), UNI2-h mean-pooled patch tokens (5393822, 315 slides) |
| **Result** | AUC: UNI2 **0.973**, colour-hist 0.924, shape 0.808, phash 0.766. MAX-fusion 0.871 (worse than hist alone). AUC is the wrong metric — operationally colour-hist at 80% recall puts **4,664/4,683 slides in one blob** (19 isolated); 11M pairs vs ~1k twins needs FPR ≲1e-5. UNI2 gets **39/44** twin pairs as mutual nearest neighbours and 3 of 4 serial cuts at rank 1 |
| **Result (limits)** | Absolute cosine does not separate: twin min 0.828 < not-twin max 0.942. Margin (top1−top2) ranks better but also fails — `a0f29ff1`↔`a576f47b`, both ISUP 0 benign, scored twin-like and the user overruled it |
| **Decision** | Stop trying to threshold anything into drops. Restore all 852 dropped slides to train and select **val/test from the most isolated slides per ISUP grade** (job 5394252, `src/embed_all_uni2.py`). Asymmetry does the work: a false group costs nothing (both slides train), only a missed twin split across train/val is expensive. Judge the run on isolated-slide count per grade, not AUC |

## Slide near-duplicate audit (2026-08-07)

| | |
|--|--|
| **Why** | User spotted same cores under different `image_id` (e.g. `48440a60`/`4889d110`) — leakage / overcount risk |
| **What** | Shape+metadata scan on all `radboud_clean` slides (`src/scan_slide_duplicates.py`). Silhouette IoU (flip/rot) within same gleason; cluster + train/val/test mapping. Job **5382352** (5382339 failed SyntaxError). Safe auto-dedupe: size-2 pairs IoU≥0.70 + same ISUP/gleason. Lower-IoU (0.30–0.70) + multi clusters held for visual review. |
| **Result** | Scan OK but silhouette alone over-clusters thin cores. **Safe drop 545** twins → 4683→**4135** slides (backups `*_pre_dedupe.csv`). User reviewing `galleries/lower_iou/` + `multi_clusters/`. So far **14** lower-IoU pairs marked not-same (`user_marked_not_same_lower_iou.csv`). |
| **Decision** | **Safe pairs locked by user (2026-08-07):** true twins → drop one (keep more patches); already applied, verified absent from splits. User “not same” lower-IoU: **keep both**, force both into **train** to block cross-split leakage. User not-same lower-IoU now **86** pairs; keep both + force train. Twin keep/drop policy updated: **prefer cleaner** (v33 pen/dark) over raw patch count when clearly dirtier; P238 drop `62202…` keep `ddbeff…`; **20** safe-pair flips. Splits 3346/384/404. Lower-IoU not-same **87** pairs (keep+force train). Multi-clusters: greedy drop **141** direct IoU≥0.70 redundancies (prefer cleaner); residual multi≥2 forced train. Splits 3252/359/381. |  



### Suspects forced train (2026-08-07 16:00)
| | |
|--|--|
| **Why** | Prevent leakage if any near-twin still kept across splits |
| **What** | Force all still-alive IDs from safe pairs + lower-IoU + multi clusters into **train** |
| **Result** | 1302 suspects → train; slides **3413/254/251** (87.1/6.5/6.4%); patches 407k/31k/29k |
| **Decision** | Accept train-heavy split for leakage safety; do not put suspects in val/test |


### Not-twins gallery corrections (2026-08-09 13:41)
| | |
|--|--|
| **Why** | User: P44 is twin; page2 `3858`/`b61a` twins; suspect `f90`/`aaaf` |
| **What** | Drop P44 dirty twin (`69a682`). Confirmed `3858`–`b61a` IoU **0.867** (gallery showed weaker `3858`–`f90` 0.662); `b61a` already removed in multi. Rendered side-by-sides under `galleries/cross_pair_checks/`. `f90`–`aaaf` silhouette IoU **0.160** (no pair edge); `aaaf` already multi-dropped. |
| **Result** | Splits **3142/254/251** (3647). Not-same list **86** (−P44). |
| **Decision** | Cross-pair high-IoU can hide behind wrong mutual-NN gallery partners; keep visual audit for residual lookalikes. Await user call on `f90`/`aaaf` (visual similar, shape IoU low). |


### Multi C23 restore a8a951/e8d79 (2026-08-09 13:46)
| | |
|--|--|
| **Why** | User: twins page87 P1033/P1034 not real twins; a8a951 & e8d79 safe |
| **What** | Gallery artifact: same KEEP shown once per DROP in multi-cluster. Restored both IDs to train from pre-drop backups. |
| **Result** | +2 slides → **3144/254/251** (3649). KEEP `2102cdae…` unchanged; `6549ae…` still dropped. |
| **Decision** | Trust visual: weak chain members ≠ twins. |


### Restore multi false-twins P1022/P1024–1032 (2026-08-09 13:51)
| | |
|--|--|
| **Why** | User: drop-sides not twins / safe (page86–87); f90 vs aaaf not twins |
| **What** | Restored 10 IDs to train; left P1023 `fb01a0…` dropped; no aaaf restore (multi-drop separate) |
| **Result** | +10 slides → **3154/254/251** (3659); +1465 patches |
| **Decision** | Multi keep-vs-each-drop gallery over-flags lookalikes; trust user safe calls |


### Safe not-twins P476/P480/P511 (2026-08-09 14:10)
| | |
|--|--|
| **Why** | User: these safe-IoU pairs are not twins |
| **What** | Restore drop sides; force both IDs train |
| **Result** | +3 slides → **3157/254/251** (3662) |
| **Decision** | Keep as not-same safe exceptions |


### Multi C1-C35 Friday restore for re-review (2026-08-09 14:24)
| | |
|--|--|
| **Why** | User did not review multi pages 1–35; `_after` gallery is post-drop and confusing |
| **What** | Restore all dead C1–C35 members from Friday `*_pre_multi_cluster_dedupe`; force train; review via `galleries/multi_clusters/` |
| **Result** | +166 slides; C36–C55 left as decided |
| **Decision** | Re-review C1–C35 from Friday gallery before re-applying drops |


### Split rebalance 80/10/10 suspects-in-train (2026-08-09 15:35)
| | |
|--|--|
| **Why** | Train was ~87% after forcing suspects; want 80/10/10 without putting near-dup risk in val/test |
| **What** | Lock 2669 suspects (curated + alive pair edges) in train; fill val/test from clean only (ISUP-stratified) |
| **Result** | **3064/383/383** (80/10/10); patches 364k/43k/45k; suspect leak 0 |
| **Decision** | Keep this split for next train; rescan twins still pending if desired |


### Restore multi safe C35-55 (2026-08-09 15:42)
| | |
|--|--|
| **Why** | User: C35–55 mostly safe (like Friday C1–35); only named drops should stay removed |
| **What** | Restored 28 dead C35–55 members to train; kept explicit drops (C36 850b, C39 non-6f50, C42 971, C49 dbf09, C51 a48b, C54 b8aff) + Friday twins |
| **Result** | 3858 slides (**3092/383/383**); C1–35 still only 3 twin drops |
| **Decision** | Optional re-run 80/10/10 rebalance after this restore |

### Alive twin rescan (job 5392157, 2026-08-09 15:53)
| | |
|--|--|
| **Why** | Catch remaining dups; include prior not-twins |
| **What** | Exclusive safe/multi/lower partition + galleries under slide_duplicates_rescan_alive |
| **Result** | Running/queued on defq |
| **Decision** | Detect-only until review |
