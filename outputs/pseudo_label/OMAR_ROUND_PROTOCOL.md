# Dr. Omar protocol (2026-07-29) — supersedes mid-run EMA design

Status: **authoritative guidance**. Round-based Option A, with corrections below.
`EMA_ISUP_WITHIN_RUN_DESIGN.md` (Option B mid-run self-relabel) is **REJECTED**.

## What Omar confirmed

- Build **round-based** iteration (Option A), **not** continuous mid-run target refresh on the same weights (Option B).
- Mid-run self-relabel trusts early mistakes and is hard to tune.
- Keep to **1–2 rounds**.

## Where the earlier “Option A” description was wrong

The ISUP check is **not** “once at the start of each run only.”

| What | Cadence |
|------|---------|
| **Pixel mask / cleaned seg target** | Regenerated **once per round** (between Model N and Model N+1) |
| **ISUP grade head** | Trains **alongside** the seg head **every batch, every round** — continuous honesty at slide/grade level |

So the loop is: discrete rounds for **pixel targets**, but a live **grade loss** keeps the model honest within each round.

## Losses (do not use fighting dual pixel losses)

Omar clarification (2026-07-29):

| Head | Losses |
|------|--------|
| **Seg head** | Regular **CE + Dice** on the pixel answer key, **plus a slide-level loss** (keeps the paint consistent with slide grade / composition — not a second fighting pixel pseudo map) |
| **Grade head** | **Grade loss only** (predict slide ISUP vs clinician) |

- Still **not** the old 0.7 seg + 0.3 pixel-pseudo split.
- Exact form of seg’s “slide-level loss” (e.g. derived grade/proportions from current preds vs metadata) is an implementation detail to pin down with Omar if needed.
- Full gate + ISUP referee pixel cleanup remains **between rounds**.

## Open question — slide-level loss vs grade head (ask Omar)

Possible training pattern Anh heard:

1. Batch so **all patches of a slide sit in one batch** (slide-complete batches).
2. Seg head paints every patch in that batch.
3. **Aggregate** those preds → derive ISUP (same style as offline diagnostic).
4. Compare derived ISUP to clinician ISUP → a loss.

**Unclear which head that loss belongs to:**

| Option | Meaning |
|--------|---------|
| **A — Seg’s slide-level loss** | Derived-ISUP-from-paint vs clinician is the *extra* term on the seg path; grade head is a separate classifier with its own grade loss |
| **B — Grade head** | That aggregate-from-seg → ISUP *is* the grade head (grade head reads seg output / pooled paint, not a totally independent feature classifier); seg then only has CE+Dice |
| **C — Both somehow** | Unlikely / redundant unless they differ (e.g. soft proportions vs hard derived ISUP) |

Do **not** implement until Omar picks A/B/C. Also confirm: true “all patches of a slide in one batch” may be huge (memory) — may need “enough patches per slide” or gradient accumulation per slide instead.

## Round 1 vs Round 2+ (high level)

- **Round 1:** start from the **original Radboud mask** (must start somewhere). Current `5307779` is this Rules-edited mask baseline (no grade head yet).
- **Round 2:** do **not** replace the mask with raw previous-model preds everywhere. Build one **cleaned** target from last round’s output, then train Model 2 on that.

## Pixel cleaning order (Round 2+ target build) — order matters

For each pixel, using last-round prediction vs original Radboud label:

1. **Agree** with original label → **keep original** (even at moderate confidence).
2. **Disagree + low confidence** → **ignore** (no loss on that pixel).
3. **Disagree + confident** only → **ISUP referee**: if predicted class is forbidden for that slide’s metadata, swap to nearest allowed grade; else take the prediction.
4. Rule 1/2/3-style swaps must run **after** the confidence gate (never swap a low-confidence guess).
5. **Adjacency soft cleanup** last (optional end step).

Gate first, then swap — so Rules only touch confident pixels.

## Implication for current work

- Finish measuring Round 1 Rules-only baseline (`5307779`) — still valid as Round 1 start.
- Do **not** implement EMA within-run refresh.
- Next design work (after Round 1): grade head + slide-level ISUP loss; Round 2 cleaned-target builder in Omar’s order; single seg loss on that map.


## Option 3 chosen (2026-07-30)

Implementing **Option 3**: seg CE+Dice + derived-ISUP-from-seg slide loss **and** separate feature grade head (λ=0.3/0.3). Slide-bag loader over existing patches; train job `5322322` / tag `pseudo_r1_opt3_slidebag`.
