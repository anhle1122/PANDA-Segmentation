#!/usr/bin/env python3
"""Score every duplicate signal against the pairs the user already adjudicated.

Positives  = 845 confirmed twin drops + the hard serial-cut pairs shape ranked last.
Negatives  = 106 pairs the user explicitly cleared as not-twins.

For the split we only care about recall: a false positive sends a slide to
train, which costs nothing, while a miss puts the same biopsy on both sides of
the split. So the operating point is "highest recall we can buy" and the FPR
column is just the price in training data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from patch_utils import PROJECT

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"


def load_labels() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    conf = pd.read_csv(DUP / "galleries" / "confirmed_twins_keep_drop" / "all_confirmed_twins_keep_drop.csv", dtype=str)
    pos = {tuple(sorted([str(r.keep_id), str(r.drop_id)])) for r in conf.itertuples(index=False)}
    hard = pd.read_csv(G / "benchmark_hard_positives.csv", dtype=str)
    pos |= {tuple(sorted([r.id_a, r.id_b])) for r in hard.itertuples(index=False)}
    neg_df = pd.read_csv(DUP / "CANONICAL_not_twin_pairs.csv", dtype=str)
    neg = {tuple(sorted([r.id_a, r.id_b])) for r in neg_df.itertuples(index=False)}
    return sorted(pos - neg), sorted(neg)


def main() -> None:
    sim = np.load(G / "content_similarity.npz", allow_pickle=True)
    ids = [str(x) for x in sim["ids"]]
    idx = {i: n for n, i in enumerate(ids)}

    shape = np.zeros((len(ids), len(ids)), dtype=np.float32)
    e = pd.read_csv(G / "all_pairs_grade_agnostic.csv", dtype={"image_id_a": str, "image_id_b": str})
    a = e.image_id_a.map(idx).to_numpy()
    b = e.image_id_b.map(idx).to_numpy()
    shape[a, b] = e.shape_iou.to_numpy()
    shape[b, a] = e.shape_iou.to_numpy()

    signals = {"shape_iou": shape, "phash": sim["phash"], "colorhist": sim["hist"]}
    pos, neg = load_labels()
    keep = lambda ps: [(x, y) for x, y in ps if x in idx and y in idx]
    pos, neg = keep(pos), keep(neg)
    print(f"benchmark: {len(pos)} twin pairs, {len(neg)} not-twin pairs\n")

    def vals(m, ps):
        return np.array([m[idx[x], idx[y]] for x, y in ps])

    hard = pd.read_csv(G / "benchmark_hard_positives.csv", dtype=str)
    rows, report = [], {}
    for name, m in signals.items():
        p, n = vals(m, pos), vals(m, neg)
        # threshold that keeps every known not-twin out, and recall there
        strict = float(n.max())
        rec_at_strict = float((p > strict).mean())
        # threshold for 99% recall, and what it costs in false positives
        t99 = float(np.percentile(p, 1))
        fpr_at_99 = float((n >= t99).mean())
        auc = float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
        hard_scores = [float(m[idx[r.id_a], idx[r.id_b]]) for r in hard.itertuples(index=False)]
        rows.append(
            {
                "signal": name,
                "AUC": round(auc, 3),
                "recall@0_FP": round(rec_at_strict, 3),
                "FPR@99_recall": round(fpr_at_99, 3),
                "hard_pos_min": round(min(hard_scores), 3),
                "neg_median": round(float(np.median(n)), 3),
            }
        )
        report[name] = rows[-1]

    print(pd.DataFrame(rows).to_string(index=False))

    print("\nthe 4 hard serial-cut pairs, per signal (negative median in brackets):")
    for r in hard.itertuples(index=False):
        line = f"  {r.id_a[:8]}+{r.id_b[:8]}"
        for name, m in signals.items():
            n = vals(m, neg)
            pct = float((n < m[idx[r.id_a], idx[r.id_b]]).mean() * 100)
            line += f"  {name}={m[idx[r.id_a], idx[r.id_b]]:.3f} (beats {pct:4.0f}% of negatives)"
        print(line)

    (G / "fingerprint_scores.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
