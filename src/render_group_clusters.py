#!/usr/bin/env python3
"""Render slide groups of three or more members, one page per group.

Grade purity is printed on every page because it is the honest quality signal:
all 841 confirmed twin pairs share an ISUP grade, so a group spanning several
grades is chain-collapse (A resembles B, B resembles C) rather than one
specimen cut several times.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from patch_utils import PROJECT
from render_dedupe_galleries import font, font_b, thumb
from score_fingerprints import load_labels

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
TW, TH = 170, 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--pure-only", action="store_true", help="only groups where every slide shares one ISUP grade")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-name", default="clusters_3plus")
    args = ap.parse_args()
    out = G / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(G / "grouped_split_rank1.csv", dtype={"image_id": str})
    emb = np.load(G / "uni2_slide_embeddings.npz", allow_pickle=True)
    ids = [str(x) for x in emb["ids"]]
    idx = {i: n for n, i in enumerate(ids)}
    vecs = emb["vecs"]
    pos, _ = load_labels()
    posset = {tuple(sorted(p)) for p in pos}

    sizes = d.groupby("group_id").size()
    pick = sizes[(sizes >= args.min_size) & (sizes <= args.max_size)].sort_values(ascending=False)
    fB, fS, fT = font_b(19), font(13), font(11)

    rows_out = []
    n_done = 0
    for gid in pick.index:
        sub = d[d.group_id == gid].sort_values("isup_grade")
        mems = sub.image_id.tolist()
        grades = sub.isup_grade.tolist()
        pure = len(set(grades)) == 1
        if args.pure_only and not pure:
            continue
        n_done += 1
        if args.limit and n_done > args.limit:
            break

        v = vecs[[idx[m] for m in mems]]
        sim = v @ v.T
        np.fill_diagonal(sim, np.nan)
        lo = float(np.nanmin(sim))
        n_conf = sum(
            1
            for a in range(len(mems))
            for b in range(a + 1, len(mems))
            if tuple(sorted([mems[a], mems[b]])) in posset
        )

        cols = min(8, len(mems))
        nrow = (len(mems) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (TW + 10) + 24, 76 + nrow * (TH + 42)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        dr.text(
            (12, 8),
            f"Group {gid}  -  {len(mems)} slides  -  {'SAME ISUP grade' if pure else 'MIXED ISUP grades'}"
            f"  -  {sub.split.iloc[0]}",
            fill=(20, 20, 20) if pure else (170, 30, 30),
            font=fB,
        )
        dr.text(
            (12, 34),
            f"weakest link in group cos={lo:.4f}   already-confirmed twin pairs inside: {n_conf} of "
            f"{len(mems)*(len(mems)-1)//2}",
            fill=(70, 70, 70),
            font=fS,
        )
        dr.text(
            (12, 52),
            "Mixed grades means the group is a similarity chain, not one specimen. All of these are in train.",
            fill=(110, 110, 110),
            font=fT,
        )
        for i, (sid, gr) in enumerate(zip(mems, grades)):
            x = 12 + (i % cols) * (TW + 10)
            y = 76 + (i // cols) * (TH + 42)
            sheet.paste(thumb(sid, (TW, TH)), (x, y))
            dr.text((x, y + TH + 4), f"ISUP {gr}", fill=(30, 30, 30), font=fS)
            dr.text((x, y + TH + 20), sid[:14], fill=(120, 120, 120), font=fT)
        sheet.save(out / f"group_{int(gid):04d}_n{len(mems)}.png", optimize=True)
        rows_out.append(
            {
                "group_id": gid,
                "n": len(mems),
                "grade_pure": pure,
                "min_cos_in_group": round(lo, 4),
                "confirmed_twin_pairs_inside": n_conf,
                "split": sub.split.iloc[0],
                "image_ids": ";".join(mems),
            }
        )

    df = pd.DataFrame(rows_out)
    df.to_csv(out / "clusters.csv", index=False)

    # Browse order: strongest same-grade groups first, mixed-grade chains last,
    # so the real serial-section sets and the chaining failures do not interleave.
    if not df.empty:
        df = df.sort_values(["grade_pure", "min_cos_in_group"], ascending=[False, False])
        cards = "\n".join(
            f'<div class="c{"" if r.grade_pure else " mix"}">'
            f'<h3>group {r.group_id} &middot; n={r.n} &middot; '
            f'{"same grade" if r.grade_pure else "MIXED grades"} &middot; min cos {r.min_cos_in_group}</h3>'
            f'<a href="group_{int(r.group_id):04d}_n{r.n}.png">'
            f'<img src="group_{int(r.group_id):04d}_n{r.n}.png" loading="lazy"></a></div>'
            for r in df.itertuples(index=False)
        )
        n_pure = int(df.grade_pure.sum())
        (out / "index.html").write_text(
            "<html><head><meta charset='utf-8'><title>slide groups</title><style>"
            "body{background:#111;color:#eee;font:14px system-ui;margin:16px}"
            "h1{font-size:18px}h3{font-size:13px;margin:6px 0;font-weight:600}"
            ".c{border:1px solid #333;padding:8px;margin-bottom:14px}"
            ".mix h3{color:#e58}img{max-width:100%;background:#fff}"
            "</style></head><body>"
            f"<h1>{len(df)} groups &middot; {n_pure} share one ISUP grade &middot; "
            f"{len(df)-n_pure} span several grades (similarity chains, not one specimen)</h1>"
            "<p>Sorted: same-grade groups first, strongest link first. All of these sit in train.</p>"
            f"{cards}</body></html>",
            encoding="utf-8",
        )

    print(f"rendered {len(df)} groups to {out}")
    if not df.empty:
        print(f"grade-pure: {df.grade_pure.mean():.0%} | median weakest-link cosine {df.min_cos_in_group.median():.4f}")
        print(f"groups containing at least one already-confirmed twin pair: {(df.confirmed_twin_pairs_inside>0).mean():.0%}")


if __name__ == "__main__":
    main()
