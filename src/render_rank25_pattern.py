#!/usr/bin/env python3
"""Render rank-2..5 pairs so we can look for patterns.

Two galleries:
  A) Confirmed twins whose partner is NOT rank-1 (the known misses) — ground truth
     for "what does a real twin look like when UNI2 ranks it 2-5?"
  B) Unknown candidates at best-rank 2-5 (same ISUP, not ledger twin/not-twin),
     strongest cosine first — what the residual might look like.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from patch_utils import PROJECT
from render_dedupe_galleries import GUTTER, TH, TW, font, font_b, thumb
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
PER = 8


def render(pages_df: pd.DataFrame, out: "Path", title: str, blurb: str) -> None:
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fB, fS, fT, fP = font_b(20), font(14), font(11), font_b(24)
    n = len(pages_df)
    pages = max(1, (n + PER - 1) // PER)
    for pi in range(pages):
        chunk = pages_df.iloc[pi * PER : (pi + 1) * PER]
        H = 90 + len(chunk) * (TH + 56)
        sheet = Image.new("RGB", (GUTTER + 2 * (TW + 16) + 360, H), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        dr.text((12, 8), f"{title}  page {pi+1}/{pages}  ({n} pairs)", fill=(10, 10, 10), font=fB)
        dr.text((12, 36), blurb[:110], fill=(70, 70, 70), font=fS)
        dr.text((12, 56), blurb[110:220], fill=(70, 70, 70), font=fS)
        y = 90
        for k, r in enumerate(chunk.itertuples(index=False)):
            num = pi * PER + k + 1
            dr.rectangle((8, y + 22, GUTTER - 8, y + 78), fill=(20, 20, 20))
            dr.text((14, y + 32), f"R{num}", fill=(255, 230, 0), font=fP)
            sheet.paste(thumb(r.id_a), (GUTTER + 16, y + 22))
            sheet.paste(thumb(r.id_b), (GUTTER + TW + 32, y + 22))
            tag = getattr(r, "tag", "")
            head = (
                f"R{num}  cos={r.cos:.4f}  best_rank={r.best_rank}  "
                f"ranks {r.rank_ab}/{r.rank_ba}  ISUP {r.isup_a}/{r.isup_b}  {tag}"
            )
            col = (20, 20, 20) if int(r.isup_a) == int(r.isup_b) else (160, 30, 30)
            dr.text((GUTTER + 16, y), head, fill=col, font=fS)
            dr.text((GUTTER + 16, y + TH + 26), str(r.id_a)[:16] + "…", fill=(40, 40, 40), font=fT)
            dr.text((GUTTER + TW + 32, y + TH + 26), str(r.id_b)[:16] + "…", fill=(40, 40, 40), font=fT)
            y += TH + 56
        sheet.save(out / f"page_{pi+1:03d}.png", optimize=True)
    pages_df.to_csv(out / "pairs.csv", index=False)
    print(f"wrote {pages} page(s), {n} pairs → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rank", type=int, default=5)
    ap.add_argument("--unknown-n", type=int, default=40)
    ap.add_argument("--min-cos-unknown", type=float, default=0.92)
    args = ap.parse_args()

    emb = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in emb["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    sim = emb["vecs"] @ emb["vecs"].T
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)
    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")
    pos, neg = load_labels()
    posset = {tuple(sorted(p)) for p in pos}
    negset = {tuple(sorted(p)) for p in neg}

    def ranks(a, b):
        i, j = idx[a], idx[b]
        ra = int(np.where(order[i] == j)[0][0]) + 1
        rb = int(np.where(order[j] == i)[0][0]) + 1
        return ra, rb, min(ra, rb), float(sim[i, j])

    # --- A: confirmed twins at best_rank 2..max ---
    conf_rows = []
    for a, b in sorted(posset):
        if a not in idx or b not in idx:
            continue
        ra, rb, best, cos = ranks(a, b)
        if 2 <= best <= args.max_rank:
            conf_rows.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "cos": cos,
                    "rank_ab": ra,
                    "rank_ba": rb,
                    "best_rank": best,
                    "isup_a": int(meta.isup_grade[a]),
                    "isup_b": int(meta.isup_grade[b]),
                    "tag": "CONFIRMED TWIN",
                }
            )
    conf = pd.DataFrame(conf_rows).sort_values(["best_rank", "cos"], ascending=[True, False])
    print(f"confirmed twins at rank 2-{args.max_rank}: {len(conf)}")
    if len(conf):
        print(conf.best_rank.value_counts().sort_index().to_string())
        render(
            conf,
            G / "rank25_confirmed_misses",
            "Confirmed twins at UNI2 rank 2-5 (known misses)",
            "These ARE twins by your eye, but partner is not rank-1. Look for shared patterns "
            "(serial cuts, stain shift, orientation, scant tissue).",
        )

    # --- B: unknown candidates best_rank 2..max ---
    # Collect one edge per unordered pair: for each slide, look at neighbors ranked 2..max
    seen = set()
    unk_rows = []
    for i, sid in enumerate(ids):
        for r in range(1, args.max_rank):  # positions 1..max-1 = ranks 2..max
            j = int(order[i, r])
            key = tuple(sorted([sid, ids[j]]))
            if key in seen or key in posset or key in negset:
                continue
            seen.add(key)
            a, b = key
            ra, rb, best, cos = ranks(a, b)
            if not (2 <= best <= args.max_rank):
                continue
            if cos < args.min_cos_unknown:
                continue
            ga, gb = int(meta.isup_grade[a]), int(meta.isup_grade[b])
            if ga != gb:
                continue
            unk_rows.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "cos": cos,
                    "rank_ab": ra,
                    "rank_ba": rb,
                    "best_rank": best,
                    "isup_a": ga,
                    "isup_b": gb,
                    "tag": f"unknown r{best}",
                }
            )
    unk = pd.DataFrame(unk_rows).sort_values(["best_rank", "cos"], ascending=[True, False])
    # take top unknown_n/2 from rank2, rest spread across 3-5
    parts = []
    per = max(1, args.unknown_n // (args.max_rank - 1))
    for br in range(2, args.max_rank + 1):
        parts.append(unk[unk.best_rank == br].head(per))
    sel = pd.concat(parts).head(args.unknown_n).reset_index(drop=True)
    print(f"unknown candidates available: {len(unk)}; rendering {len(sel)}")
    render(
        sel,
        G / "rank25_unknown_sample",
        "Unknown rank 2-5 candidates (same ISUP, not in ledger)",
        "Not auto-merged. Compare to the confirmed-miss gallery: same look → likely residual twins; "
        "only generic resemblance → leave alone.",
    )


if __name__ == "__main__":
    main()
