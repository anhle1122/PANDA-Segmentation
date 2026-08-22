# Section 4 — pixel-level wmfix vs ep14 referee

Job **5514638** finished 2026-08-21 20:18 PDT. **No training started.**

wmfix pixel = teacher-A argmax ∈ that slide’s `flag_pred_classes` (same flag as Round 1 train).
Referee pixel = corrected `target !=` original mask on shared `(x,y)` patches.

## Headline

| | pixels |
|---|---:|
| wmfix flagged (442 slides) | 3,954,077,293 |
| referee swap (all 2,487 swap slides) | 3,069,107,759 |
| referee swap on shared patches (350 comparable slides) | 663,541,177 |
| **overlap** | **130,976,321** |
| wmfix only | 3,823,100,972 |
| referee only on shared patches | 532,564,856 |

- Of wmfix flagged pixels, **3.3%** are also referee swaps (4.2% on the 350 slides that exist in both packs).
- Of all referee swaps, **4.3%** hit a wmfix flag.
- Of referee swaps *on those shared patches*, **19.7%** hit a wmfix flag.

This is low on purpose. The systems are not supposed to rewrite the same pixels.

## Does this look broken, or like different real problems?

**Different real problems — not a red flag.** Three independent cuts say the same thing:

1. **Most referee volume is on slides wmfix never touched.** 3.07B − 0.66B = **2.41B swaps (78%)** sit outside the wmfix-correcting set. That is the Test B pattern: ep14 is high-conf-but-wrong on a much larger slide set than teacher-A’s old Rules 1–3.
2. **Even on overlapping slides, 80% of referee pixels are not wmfix flags.** Same slide, different locations / different teacher.
3. **Rule 1 (legal-grade soft ties) is almost untouched.** 0.44% of those wmfix pixels are also referee swaps. That is the design: legal-grade fights keep the expert mask.

## By wmfix rule (350 slides with both H5s)

| Rule | slides | wmfix px | overlap | % of wmfix also referee | % of shared-ref also wmfix |
|---|---:|---:|---:|---:|---:|
| rule1_soft_tie (legal fight) | 50 | 744,105,273 | 3,300,860 | **0.44%** | 41.8% of a tiny referee volume |
| rule2_adjacent_invented | 7 | 94,110,013 | 1,901,138 | 2.02% | 53.2% |
| rule3_invented_default | 293 | 2,257,006,283 | 125,774,323 | 5.57% | 19.3% |

Rule 3 is where the small overlap lives (96% of overlap pixels). Still only 5.6% of those wmfix flags — expected, because wmfix rewrites *every* teacher-A pixel of a flagged class, while the referee only swaps high-conf illegal G3/G4/G5 from a *different* teacher (ep14).

## Coverage caveats (not blockers)

- **92 / 442** wmfix-correcting slides have no ep14 teacher pack and no referee H5. They are **not** in `panda_train.csv` (not ISUP-0 skips). Their wmfix pixels are wmfix-only by construction. Teacher-A srcpred exists for all 442.
- Patch sets differ: 41,360 shared `(x,y)` patches on the 350 comparable slides.
- 315 / 350 comparable slides have at least one overlapping pixel; median per-slide overlap/wmfix is 4.9%.

## Verdict

Pixel-level overlap is small and structured the way the spec requires. The new referee is not silently replaying wmfix, and it is not fighting Rule 1 legal-grade order. **Still do not start Round N+1 until this section is reviewed.**
