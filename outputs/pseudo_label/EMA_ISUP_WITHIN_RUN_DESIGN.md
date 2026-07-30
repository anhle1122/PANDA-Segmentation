# EMA + ISUP within-run refresh (DESIGN REJECTED)

**Rejected 2026-07-29 per Dr. Omar.** Do not implement Option B (mid-run self-relabel on the same weights).

Authoritative protocol: `outputs/pseudo_label/OMAR_ROUND_PROTOCOL.md`

Omar: round-based (A) only; pixel mask regenerates once per round; ISUP honesty is a **live grade head / grade loss** every batch — not EMA-teacher periodic mask refresh. Mid-run relabeling trusts early mistakes and is hard to tune.

The notes below are archived for context only.

---

## What EMA is (and is not) — archived

- **Is:** exponential moving average of the **student's weights**, updated every training batch:
  `θ_teacher ← α · θ_teacher + (1−α) · θ_student` with α ≈ 0.999.
- **Is not:** averaging predictions/outputs.

## Why it was proposed

Make mid-run ISUP checks “safe” by checking a smoothed teacher instead of the raw student, refreshing corrected targets every N epochs inside one continuous run.

## Why Omar rejected it

A model relabeling itself mid-run on the same weights starts trusting its own early mistakes before it's any good; headache to tune. Round-based is more stable. Continuous honesty should come from a **slide-level ISUP grade loss** next to seg, not from regenerating the pixel mask mid-run.
