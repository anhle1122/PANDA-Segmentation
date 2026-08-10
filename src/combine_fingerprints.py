#!/usr/bin/env python3
"""Fuse the duplicate signals and measure the fused score on the benchmark.

Each signal is first turned into a percentile against a random sample of all
~11M pairs, so "0.9 colourhist" and "0.9 phash" mean the same thing (both are
rarer than 90% of random pairs) and can be compared. Then:

  max-rule  score = max percentile over signals -- a pair is suspicious if ANY
            signal finds it unusual. This is the union, which is what we want:
            shape catches re-scans, content catches serial cuts.
  logistic  cross-validated, to check whether a learned weighting beats the
            max-rule. Reported with 5-fold CV since there are only 106 negatives.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from patch_utils import PROJECT
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
RNG = np.random.default_rng(0)


def main() -> None:
    sim = np.load(G / "content_similarity.npz", allow_pickle=True)
    ids = [str(x) for x in sim["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    n = len(ids)

    shape = np.zeros((n, n), dtype=np.float32)
    e = pd.read_csv(G / "all_pairs_grade_agnostic.csv", dtype={"image_id_a": str, "image_id_b": str})
    a = e.image_id_a.map(idx).to_numpy()
    b = e.image_id_b.map(idx).to_numpy()
    shape[a, b] = shape[b, a] = e.shape_iou.to_numpy()

    signals = {"shape": shape, "phash": sim["phash"], "hist": sim["hist"]}

    # reference distribution: 2M random pairs
    ri = RNG.integers(0, n, 2_000_000)
    rj = RNG.integers(0, n, 2_000_000)
    ok = ri != rj
    ref = {k: np.sort(m[ri[ok], rj[ok]]) for k, m in signals.items()}

    def pct(name, v):
        return np.searchsorted(ref[name], v) / len(ref[name])

    pos, neg = load_labels()
    pos = [(x, y) for x, y in pos if x in idx and y in idx]
    neg = [(x, y) for x, y in neg if x in idx and y in idx]

    def feats(pairs):
        out = []
        for name, m in signals.items():
            raw = np.array([m[idx[x], idx[y]] for x, y in pairs])
            out.append(pct(name, raw))
        return np.column_stack(out)

    Xp, Xn = feats(pos), feats(neg)
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]

    def report(name, sp, sn):
        auc = float((sp[:, None] > sn[None, :]).mean())
        strict = float(sn.max())
        rows = []
        for r in (0.99, 0.97, 0.95, 0.90):
            t = float(np.percentile(sp, 100 * (1 - r)))
            rows.append((r, round(float((sn >= t).mean()), 3)))
        print(f"\n{name}: AUC {auc:.3f} | recall at zero false positives {float((sp>strict).mean()):.3f}")
        for r, f in rows:
            print(f"    recall {r:.0%}  ->  flags {f:.1%} of known-hard negatives")
        return auc

    per_signal = {}
    for i, name in enumerate(signals):
        per_signal[name] = report(f"{name} (percentile)", Xp[:, i], Xn[:, i])

    auc_max = report("MAX-rule fusion", Xp.max(1), Xn.max(1))

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        oof[te] = lr.decision_function(X[te])
    auc_lr = report("logistic fusion (5-fold CV)", oof[y == 1], oof[y == 0])
    lr_full = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    print("    weights:", dict(zip(signals, lr_full.coef_[0].round(2))))

    hard = pd.read_csv(G / "benchmark_hard_positives.csv", dtype=str)
    hp = feats([(r.id_a, r.id_b) for r in hard.itertuples(index=False)])
    print("\nhard serial-cut pairs under MAX-rule:")
    for r, f in zip(hard.itertuples(index=False), hp):
        print(f"  {r.id_a[:8]}+{r.id_b[:8]}  shape={f[0]:.3f} phash={f[1]:.3f} hist={f[2]:.3f}  -> MAX {f.max():.4f}")

    (G / "fusion_scores.json").write_text(
        json.dumps({"per_signal_auc": per_signal, "auc_max_rule": auc_max, "auc_logistic_cv": auc_lr}, indent=2)
    )


if __name__ == "__main__":
    main()
