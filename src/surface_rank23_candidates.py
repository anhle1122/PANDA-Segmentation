#!/usr/bin/env python3
"""Surface a bounded review list for UNI2 rank-2 / rank-3 twin candidates.

Do NOT add these as grouping edges wholesale — that chain-collapses the graph.
Instead: list pairs where the partner sits at rank 2 or 3, apply cheap filters
that true twins almost always pass, and leave the rest for human adjudication.

Default filters (all must hold):
  * best directional rank is 2 or 3 (partner not already rank-1)
  * same ISUP grade (all 841 confirmed twins share grade)
  * not already an adjudicated twin / not-twin edge
  * not already in the same connected group under the rank-1+ledger grouping
  * cosine above a floor (default 0.94) so we skip obvious strangers

A mutual-within-top-K flag is recorded but not required — some true twins are
asymmetric at rank 2/3.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from patch_utils import PROJECT
from render_dedupe_galleries import GUTTER, TH, TW, font, font_b, thumb
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
PER_PAGE = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="2,3")
    ap.add_argument("--min-cos", type=float, default=0.94)
    ap.add_argument("--render-n", type=int, default=40, help="0 = CSV only")
    ap.add_argument("--out-name", default="rank23_candidates")
    args = ap.parse_args()
    want = {int(x) for x in args.ranks.split(",")}
    out = G / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    sim = d["vecs"] @ d["vecs"].T
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)
    n = len(ids)

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    groups = pd.read_csv(G / "grouped_split_rank1.csv", dtype={"image_id": str}).set_index("image_id")
    pos, neg = load_labels()
    known = {tuple(sorted(p)) for p in pos} | {tuple(sorted(p)) for p in neg}

    rows = []
    seen = set()
    for i in range(n):
        for r in want:
            j = int(order[i, r - 1])
            key = tuple(sorted([ids[i], ids[j]]))
            if key in seen or key in known:
                continue
            seen.add(key)
            a, b = key
            ia, ib = idx[a], idx[b]
            # best of the two directional ranks
            ra = int(np.where(order[ia] == ib)[0][0]) + 1
            rb = int(np.where(order[ib] == ia)[0][0]) + 1
            best = min(ra, rb)
            if best not in want:
                continue
            cos = float(sim[ia, ib])
            if cos < args.min_cos:
                continue
            ga, gb = int(meta.isup_grade[a]), int(meta.isup_grade[b])
            if ga != gb:
                continue
            if groups.group_id[a] == groups.group_id[b]:
                continue  # already co-grouped via rank-1 / ledger
            # mutual within top-3?
            top3_a = set(map(int, order[ia, :3]))
            top3_b = set(map(int, order[ib, :3]))
            mutual3 = ia in top3_b and ib in top3_a
            rows.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "cos": round(cos, 4),
                    "rank_a_to_b": ra,
                    "rank_b_to_a": rb,
                    "best_rank": best,
                    "mutual_top3": mutual3,
                    "isup": ga,
                    "split_a": groups.split[a],
                    "split_b": groups.split[b],
                    "cross_split": groups.split[a] != groups.split[b],
                    "group_a": int(groups.group_id[a]),
                    "group_b": int(groups.group_id[b]),
                    "verdict": "",
                }
            )

    df = pd.DataFrame(rows).sort_values(["cross_split", "cos"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(out / "rank23_candidates.csv", index=False)
    report = {
        "n_candidates": len(df),
        "ranks": sorted(want),
        "min_cos": args.min_cos,
        "cross_split": int(df.cross_split.sum()) if len(df) else 0,
        "mutual_top3": int(df.mutual_top3.sum()) if len(df) else 0,
        "by_best_rank": df.best_rank.value_counts().sort_index().to_dict() if len(df) else {},
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not args.render_n or df.empty:
        return

    # Prefer cross-split first (those are the only expensive misses), then strongest.
    sel = df.head(args.render_n)
    fB, fS, fT, fP = font_b(22), font(15), font(12), font_b(26)
    pages = (len(sel) + PER_PAGE - 1) // PER_PAGE
    for pi in range(pages):
        chunk = sel.iloc[pi * PER_PAGE : (pi + 1) * PER_PAGE]
        sheet = Image.new("RGB", (GUTTER + 2 * (TW + 16) + 340, 96 + len(chunk) * (TH + 56)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        dr.text((12, 8), f"Rank-2/3 candidates  page {pi+1}/{pages}  (showing {len(sel)} of {len(df)})", fill=(10, 10, 10), font=fB)
        dr.text((12, 38), "Same ISUP, not already co-grouped, cos>={:.2f}. Cross-split pairs first — those are the only expensive misses.".format(args.min_cos), fill=(70, 70, 70), font=fS)
        dr.text((12, 58), "Mark TWIN / NOT. Do NOT auto-add all of these as edges — review only.", fill=(70, 70, 70), font=fS)
        y = 96
        for k, r in enumerate(chunk.itertuples(index=False)):
            num = pi * PER_PAGE + k + 1
            dr.rectangle((8, y + 22, GUTTER - 8, y + 78), fill=(20, 20, 20))
            dr.text((14, y + 30), f"R{num}", fill=(255, 230, 0), font=fP)
            tag = "CROSS" if r.cross_split else "same"
            dr.text((10, y + 86), tag, fill=(180, 30, 30) if r.cross_split else (80, 80, 80), font=fT)
            x0 = GUTTER
            sheet.paste(thumb(r.id_a), (x0 + 16, y + 22))
            sheet.paste(thumb(r.id_b), (x0 + TW + 32, y + 22))
            head = (
                f"R{num}  cos={r.cos:.4f}  ranks {r.rank_a_to_b}/{r.rank_b_to_a}  "
                f"ISUP {r.isup}  {r.split_a}↔{r.split_b}"
                + ("  MUTUAL-top3" if r.mutual_top3 else "")
            )
            dr.text((x0 + 16, y), head, fill=(150, 30, 30) if r.cross_split else (20, 20, 20), font=fS)
            dr.text((x0 + 16, y + TH + 26), r.id_a[:16] + "…", fill=(40, 40, 40), font=fT)
            dr.text((x0 + TW + 32, y + TH + 26), r.id_b[:16] + "…", fill=(40, 40, 40), font=fT)
            y += TH + 56
        sheet.save(out / f"rank23_page_{pi+1:03d}.png", optimize=True)
    print(f"wrote {pages} page(s) to {out}")


if __name__ == "__main__":
    main()
