# Why S3 G3 median (0.43–0.48) ≪ pooled (0.72–0.74)

## Short answer
**Pooled** = all G3 pixels in one pile → slides with **lots of G3** dominate.  
**Median** = one Dice per slide → a slide with a **tiny** G3 patch counts the same as a slide full of G3.

S3 is split: some specimens ~0.9+ G3 Dice, others ~0.01–0.05. The median sits in the middle; the pile follows the large-G3 slides.

## Numbers (ZS S3 Test; SICAP same pattern)

| Summary | G3 Dice |
|---------|--------:|
| Per-slide median | 0.428 |
| Per-slide mean | 0.518 |
| **Pooled** | **0.724** |

Per-slide distribution: **12/67** Dice &lt;0.2, **27/67** Dice &gt;0.8 (bimodal).

| Specimen | Mean G3 Dice (across scanners) |
|---------:|-------------------------------:|
| 5 | 0.985 |
| 2 | 0.929 |
| 7 | 0.934 |
| 3 | 0.887 |
| 12 | 0.853 |
| 11 | 0.496 |
| 4 | 0.408 |
| 1 | 0.334 |
| 6 | 0.273 |
| 9 | 0.245 |
| 10 | 0.042 |
| 8 | 0.009 |

Specimen **8** alone is six slides near Dice 0.01 — they pull the **median** down hard but add little mass to **pooled** if their G3 regions are small. Specimens 2/3/5/7/12 with high Dice carry most of the G3 pixels (pooled recall G3 = **0.889**).

## For Omar
Not inconsistent tables. Median says “many S3 slides are hard/small-G3.” Pooled says “where most G3 pixels live, we do OK (~0.72).”
