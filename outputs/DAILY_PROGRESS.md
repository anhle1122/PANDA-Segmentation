# Daily progress — PANDA Gleason project

**How to use:** Open this file each day (also mirrored at `docs/DAILY_PROGRESS.md`). Newest date at the top.  
Agents auto-append when training/eval/protocol work lands (see `.cursor/rules/daily-progress.mdc`).

Format per bullet: `- HH:MM TZ | What | Why | Result / next`

---

## 2026-07-21

- **21:56 PDT** | R4 train check `5247423` | Daily status | Through **ep39**; best val cancer **ep35 = 0.722** (was 0.707); ep39 = 0.718; still below teacher A 0.742; ~23 min/ep, running on cp097
- **14:40 PDT** | Wrote detailed tech narrative | Document QC→pen→stain(unused)→EfficientNet→ISUP@5%/3% in digestible form | `TECH_NARRATIVE.md` (+ `docs/`)
- **13:54 PDT** | Expanded presentation outline | Need ablation (A/B/C/R4, 40k vs 10k, soft 0.85/0.15), ISUP findings + per-grade fix, and 10-ep ISUP-in-loss audit plan | Updated `PRESENTATION_OUTLINE.md` (+ `docs/` mirror)
- **13:37 PDT** | Created presentation outline + this daily log | Need slide structure + Cursor-visible day-by-day tracking | See `PRESENTATION_OUTLINE.md`
- **13:33 PDT** | PANDA+ eval on R4 best so far `epoch_015_cancer_0.7072` (job `5247845`) | Check if higher val beats teacher A on external | PANDA+ cancer **0.540**, G5 **0.523** (worse than A 0.554 / 0.528 and ep6 0.554) → val↑ ≠ PANDA+↑ yet
- **~13:16 PDT** | R4 train `5247423` through ep18 | Fair A/B: 40k + g45 soft from scratch | Best val cancer still **ep15 = 0.707**; ep18 = 0.697; job still running on 4×H200

### Open tonight / tomorrow
- [ ] Let R4 run; re-eval PANDA+ when a new best beats 0.707 (or at a clear plateau)
- [ ] Restart Phase 3 label correction from **5%** `diagnostic_report.csv` (not v2)
- [ ] Design/implement every-10-epoch ISUP diagnostic + match-rate alert during train
- [ ] Resume Phase 1 aux cls head after Phase 3 confirm

---

## Template (copy for new days)

```md
## YYYY-MM-DD

- **HH:MM TZ** | What | Why | Result / job id / paths
```
