#!/usr/bin/env python3
"""Render a sample of the group merges nobody has adjudicated, for eyeballing.

These are mutual nearest neighbours that are neither confirmed twins nor
confirmed not-twins. Nothing is being dropped -- a merge only forces the two
slides onto the same side of the split -- so this is a sanity check on whether
the grouping is behaving, not a decision queue.

Weakest merges come first on purpose: mutual pairs with the lowest cosine are
where the model is least sure, so if those look like genuine matches the rest
almost certainly do too.
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
OUT = G / "spotcheck_merges"
PER_PAGE = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-weakest", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("any", "train", "val", "test"), default="any")
    ap.add_argument("--all", action="store_true", help="render every pair, strongest first, instead of a sample")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    out_dir = OUT if args.out_name is None else OUT.parent / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    sim = d["vecs"] @ d["vecs"].T
    np.fill_diagonal(sim, -1.0)
    nn = sim.argmax(1)

    pos, neg = load_labels()
    known = {tuple(sorted(p)) for p in pos} | {tuple(sorted(p)) for p in neg}

    rows = []
    for i in range(len(ids)):
        j = int(nn[i])
        if nn[j] != i or j < i:
            continue
        key = tuple(sorted([ids[i], ids[j]]))
        if key in known:
            continue
        rows.append({"id_a": key[0], "id_b": key[1], "cos": float(sim[i, j])})
    df = pd.DataFrame(rows).sort_values("cos").reset_index(drop=True)
    print(f"{len(df)} unadjudicated mutual-NN merges; cos {df.cos.min():.4f}–{df.cos.max():.4f}")

    split = pd.read_csv(G / "grouped_split_rank1.csv", dtype={"image_id": str}).set_index("image_id")
    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str}).set_index("image_id")

    if args.split != "any":
        keep = split.split == args.split
        df = df[df.id_a.map(keep).fillna(False) & df.id_b.map(keep).fillna(False)].reset_index(drop=True)
        print(f"{len(df)} of those sit entirely inside {args.split}")

    if args.all:
        # Strongest first: the likeliest true duplicates lead, so review can stop
        # once the pairs stop looking related.
        sel = df.sort_values("cos", ascending=False).reset_index(drop=True)
        sel["bucket"] = "all"
    else:
        weak = df.head(args.n_weakest)
        rest = df.iloc[args.n_weakest :].sample(
            min(args.n - args.n_weakest, len(df) - args.n_weakest), random_state=args.seed
        )
        sel = pd.concat([weak, rest]).reset_index(drop=True)
        sel["bucket"] = ["weakest"] * len(weak) + ["random"] * len(rest)

    fB, fS, fT, fP = font_b(22), font(15), font(12), font_b(26)
    pages = (len(sel) + PER_PAGE - 1) // PER_PAGE
    for pi in range(pages):
        chunk = sel.iloc[pi * PER_PAGE : (pi + 1) * PER_PAGE]
        sheet = Image.new("RGB", (GUTTER + 2 * (TW + 16) + 300, 96 + len(chunk) * (TH + 56)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        scope = "" if args.split == "any" else f" inside {args.split.upper()}"
        dr.text((12, 8), f"Unadjudicated pairs{scope}  page {pi+1}/{pages}  ({len(sel)} total)", fill=(10, 10, 10), font=fB)
        dr.text(
            (12, 38),
            "Mutual nearest neighbours the ledger has never seen. Nothing is dropped - the question",
            fill=(70, 70, 70),
            font=fS,
        )
        order = "strongest match first" if args.all else "weakest evidence first"
        dr.text((12, 58), f"is only whether these are the SAME specimen. Sorted {order}.", fill=(70, 70, 70), font=fS)
        y = 96
        for k, r in enumerate(chunk.itertuples(index=False)):
            num = pi * PER_PAGE + k + 1
            dr.rectangle((8, y + 22, GUTTER - 8, y + 78), fill=(20, 20, 20))
            dr.text((14, y + 30), f"M{num}", fill=(255, 230, 0), font=fP)
            dr.text((14, y + 86), r.bucket, fill=(80, 80, 80), font=fT)
            x0 = GUTTER
            sheet.paste(thumb(r.id_a), (x0 + 16, y + 22))
            sheet.paste(thumb(r.id_b), (x0 + TW + 32, y + 22))
            ga, gb = int(meta.isup_grade[r.id_a]), int(meta.isup_grade[r.id_b])
            sp = str(split.split[r.id_a])
            gs = int(split.group_size[r.id_a])
            head = f"M{num}  cos={r.cos:.4f}  ISUP {ga} vs {gb}  ->  both in {sp.upper()}  (group of {gs})"
            dr.text((x0 + 16, y), head, fill=(150, 30, 30) if ga != gb else (20, 20, 20), font=fS)
            dr.text((x0 + 16, y + TH + 26), r.id_a[:16] + "…", fill=(40, 40, 40), font=fT)
            dr.text((x0 + TW + 32, y + TH + 26), r.id_b[:16] + "…", fill=(40, 40, 40), font=fT)
            y += TH + 56
        sheet.save(out_dir / f"merges_page_{pi+1:03d}.png", optimize=True)

    sel["isup_a"] = sel.id_a.map(meta.isup_grade)
    sel["isup_b"] = sel.id_b.map(meta.isup_grade)
    sel["split"] = sel.id_a.map(split.split)
    sel["verdict"] = ""
    sel.to_csv(out_dir / "spotcheck_pairs.csv", index=False)
    n_mismatch = int((sel.isup_a != sel.isup_b).sum())
    print(f"wrote {pages} page(s) to {out_dir}")
    print(f"grade mismatch within sampled merges: {n_mismatch}/{len(sel)}")


if __name__ == "__main__":
    main()
