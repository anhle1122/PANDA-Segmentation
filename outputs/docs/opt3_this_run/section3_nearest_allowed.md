# Section 3 — nearest-allowed destinations and ties

Source: `correction_manifest.csv` from job 5513153. No H5 re-read. No training.

Tie-break: `min(allowed, key=lambda a: (abs(a-pred), a))` — **lower grade wins a distance tie**.

## Where swaps come from / go

| | pixels | share of 3.07B swaps |
|---|---:|---:|
| from G3 | 2,052,863,235 | 66.89% |
| from G4 | 228,449,221 | 7.44% |
| from G5 | 787,795,303 | 25.67% |
| to G3 | 219,789,089 | 7.16% |
| to G4 | 2,680,476,196 | 87.34% |
| to G5 | 168,842,474 | 5.50% |

## Primary vs secondary destination

- To clinical **primary**: 1,129,026,094 (36.79%)
- To clinical **secondary**: 343,122,393 (11.18%)
- To the only allowed grade (P=S, e.g. 4+4): 1,596,959,272 (52.03%)

## The 3+5 / G4 tie

On 3+5 slides allowed={3,5}. Pred G4 is equidistant; rule picks **G3**.
- Swap slides with allowed 3|5: **86**
- Of those, slides with any G4→G3 tie swap: **86**
- Tie pixels: **145,332,814** (4.74% of all swaps)

## By allowed set

| allowed | slides | n_swap | from G3 | from G4 | from G5 | to G3 | to G4 | to G5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 490 | 1,353,660,523 | 969,420,317 | 0 | 384,240,206 | 0 | 1,353,660,523 | 0 |
| 4|5 | 563 | 937,557,899 | 937,557,899 | 0 | 0 | 0 | 937,557,899 | 0 |
| 3|4 | 877 | 389,257,774 | 0 | 0 | 389,257,774 | 0 | 389,257,774 | 0 |
| 5 | 72 | 168,842,474 | 145,885,019 | 22,957,455 | 0 | 0 | 0 | 168,842,474 |
| 3|5 | 86 | 145,332,814 | 0 | 145,332,814 | 0 | 145,332,814 | 0 | 0 |
| 3 | 399 | 74,456,275 | 0 | 60,158,952 | 14,297,323 | 74,456,275 | 0 | 0 |

## Examples

- **largest_swap** `c5764fe96b2725f1b48f2db0cf20cacb` gleason 4+5 allowed 4|5 n_swap=14,766,360 from G3/G4/G5=14,766,360/0/0 to G3/G4/G5=0/14,766,360/0
- **isup_3plus5_g4_tie** `e573d5bd029fefceb5463707eb41a03a` gleason 3+5 allowed 3|5 n_swap=7,301,558 from G3/G4/G5=0/7,301,558/0 to G3/G4/G5=7,301,558/0/0
- **most_from_g5** `6b3066448fce3a65fbe2bda2b8e6b364` gleason 4+3 allowed 3|4 n_swap=9,290,698 from G3/G4/G5=0/0/9,290,698 to G3/G4/G5=0/9,290,698/0
- **most_from_g3** `c5764fe96b2725f1b48f2db0cf20cacb` gleason 4+5 allowed 4|5 n_swap=14,766,360 from G3/G4/G5=14,766,360/0/0 to G3/G4/G5=0/14,766,360/0

No training started.
