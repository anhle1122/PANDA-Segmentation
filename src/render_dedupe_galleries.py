#!/usr/bin/env python3
"""Render pair-by-pair galleries for safe twins, multi-clusters, and lower-IoU pairs."""

from __future__ import annotations

from pathlib import Path

import openslide
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from patch_utils import PROJECT

SLIDES = Path("/common/omarmlab/members/anh/panda_data/slides")
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
OUT = DUP / "galleries"

TW, TH = 220, 380
PAIRS_PER_PAGE = 12  # 12 rows → readable


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def font_b(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def thumb(sid: str, box=(TW, TH)) -> Image.Image:
    paths = list(SLIDES.glob(f"{sid[:12]}*.tiff"))
    canvas = Image.new("RGB", box, (235, 235, 235))
    if not paths:
        return canvas
    try:
        sl = openslide.OpenSlide(str(paths[0]))
        im = sl.get_thumbnail(box).convert("RGB")
        sl.close()
        im.thumbnail(box)
        canvas.paste(im, ((box[0] - im.width) // 2, (box[1] - im.height) // 2))
    except Exception:
        pass
    return canvas


GUTTER = 78  # left column for big pair numbers (call out as P47 / pair 47)


def pair_kind_from_flag(flag) -> str:
    """Classify rescan pairs for review badges."""
    if flag is None or (isinstance(flag, float) and pd.isna(flag)):
        return "NEW"
    s = str(flag).strip().lower()
    if not s or s in {"nan", "none", "nat"}:
        return "NEW"
    if s == "both_prior_not_twin":
        return "KEPT_NOT_TWIN"
    if s == "one_prior_not_twin":
        return "MIXED"
    return s.upper()


KIND_STYLE = {
    # fill, text
    "NEW": ((20, 120, 40), (255, 255, 255)),
    "KEPT_NOT_TWIN": ((180, 100, 0), (255, 255, 255)),
    "MIXED": ((30, 80, 160), (255, 255, 255)),
}


def render_pair_pages(df: pd.DataFrame, out_dir: Path, title_prefix: str, id_cols=("image_id_a", "image_id_b")) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fB, fS, fT, fP = font_b(22), font(14), font(12), font_b(28)
    paths = []
    index_rows = []
    n = len(df)
    if "prior_not_twin_flag" in df.columns:
        kinds_all = df["prior_not_twin_flag"].map(pair_kind_from_flag)
        n_new = int((kinds_all == "NEW").sum())
        n_kept = int((kinds_all == "KEPT_NOT_TWIN").sum())
        n_mixed = int((kinds_all == "MIXED").sum())
    else:
        n_new, n_kept, n_mixed = n, 0, 0
    pages = max(1, (n + PAIRS_PER_PAGE - 1) // PAIRS_PER_PAGE)
    for pi in tqdm(range(pages), desc=title_prefix[:24]):
        chunk = df.iloc[pi * PAIRS_PER_PAGE : (pi + 1) * PAIRS_PER_PAGE]
        H = 90 + len(chunk) * (TH + 56)
        W = GUTTER + 2 * (TW + 16) + 320
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        d.text(
            (12, 8),
            f"{title_prefix}  page {pi+1}/{pages}  (P{pi*PAIRS_PER_PAGE+1}–P{pi*PAIRS_PER_PAGE+len(chunk)} of {n})",
            fill=(10, 10, 10),
            font=fB,
        )
        d.text(
            (12, 36),
            f"NEW={n_new} (green)  KEPT_NOT_TWIN={n_kept} (orange)  MIXED={n_mixed} (blue)  |  LEFT=keep  RIGHT=drop",
            fill=(60, 60, 60),
            font=fS,
        )
        d.text(
            (12, 56),
            "KEPT_NOT_TWIN = you marked not-twins before (still alive). NEW = fresh survivor pair. MIXED = one side prior not-twin.",
            fill=(90, 90, 90),
            font=fT,
        )
        y = 90
        for local_i, r in enumerate(chunk.itertuples(index=False)):
            pair_num = pi * PAIRS_PER_PAGE + local_i + 1  # 1-based global
            row = local_i + 1
            rd = r._asdict()
            if "keep_id" in rd and "drop_id" in rd:
                a, b = str(rd["keep_id"]), str(rd["drop_id"])
            else:
                a, b = str(rd[id_cols[0]]), str(rd[id_cols[1]])
            kind = pair_kind_from_flag(rd.get("prior_not_twin_flag"))
            fill, tcol = KIND_STYLE.get(kind, ((80, 80, 80), (255, 255, 255)))
            # pair badge
            d.rectangle((8, y + 22, GUTTER - 8, y + 22 + 56), fill=(20, 20, 20))
            d.text((14, y + 30), f"P{pair_num}", fill=(255, 230, 0), font=fP)
            d.text((14, y + 82), f"row {row}", fill=(80, 80, 80), font=fT)
            # kind badge under row label
            d.rectangle((8, y + 100, GUTTER - 8, y + 122), fill=fill)
            d.text((10, y + 102), kind[:12], fill=tcol, font=fT)
            x0 = GUTTER
            sheet.paste(thumb(a), (x0 + 16, y + 22))
            sheet.paste(thumb(b), (x0 + TW + 32, y + 22))
            iou = float(rd.get("shape_iou", rd.get("max_iou", 0)))
            cross = bool(rd.get("cross_split", False))
            col = (180, 20, 20) if cross else (20, 20, 20)
            gle = rd.get("gleason", "?")
            meta = f"P{pair_num}  [{kind}]  IoU={iou:.3f}  Gleason {gle}"
            if "split_keep" in rd:
                meta += f"  {rd['split_keep']}↔{rd['split_drop']}  patches {rd.get('keep_n_patches','?')}/{rd.get('drop_n_patches','?')}"
            elif "split_a" in rd:
                meta += f"  {rd['split_a']}↔{rd['split_b']}"
            d.text((x0 + 16, y), meta, fill=col, font=fS)
            d.text((x0 + 16, y + TH + 26), f"KEEP {a[:14]}…", fill=(0, 120, 40), font=fT)
            d.text((x0 + TW + 32, y + TH + 26), f"DROP {b[:14]}…", fill=(160, 30, 30), font=fT)
            index_rows.append(
                {
                    "pair_num": pair_num,
                    "page": pi + 1,
                    "row": row,
                    "pair_kind": kind,
                    "prior_not_twin_flag": rd.get("prior_not_twin_flag", ""),
                    "keep_id": a,
                    "drop_id": b,
                    "shape_iou": iou,
                    "gleason": gle,
                    "split_keep": rd.get("split_keep", rd.get("split_a", "")),
                    "split_drop": rd.get("split_drop", rd.get("split_b", "")),
                    "cross_split": cross,
                }
            )
            y += TH + 56
        outp = out_dir / f"page_{pi+1:03d}.png"
        sheet.save(outp, optimize=True)
        paths.append(outp)
    if index_rows:
        idx = pd.DataFrame(index_rows)
        idx.to_csv(out_dir / "pair_index.csv", index=False)
        idx.to_csv(out_dir / "pair_index_annotated.csv", index=False)
        idx[idx.pair_kind == "NEW"].to_csv(out_dir / "pair_index_NEW_only.csv", index=False)
        idx[idx.pair_kind == "KEPT_NOT_TWIN"].to_csv(out_dir / "pair_index_KEPT_NOT_TWIN_only.csv", index=False)
        idx[idx.pair_kind == "MIXED"].to_csv(out_dir / "pair_index_MIXED_only.csv", index=False)
    return paths


def render_multi_pages(clusters: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fB, fS, fT = font_b(20), font(13), font(11)
    paths = []
    # one cluster per page (up to 8 thumbs)
    for r in tqdm(clusters.itertuples(index=False), total=len(clusters), desc="multi"):
        mems = r.image_ids.split(";")
        cols = min(8, len(mems))
        rows = (len(mems) + cols - 1) // cols
        tw, th = 160, 280
        W = cols * (tw + 8) + 24
        H = 80 + rows * (th + 40)
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        n_prior = int(getattr(r, "n_prior_not_twin", 0) or 0)
        kind = "HAS_KEPT_NOT_TWIN" if n_prior > 0 else "NEW"
        d.text(
            (12, 10),
            f"MULTI cluster {r.cluster_id}/{len(clusters)}  [{kind}]  n={r.n_members}  "
            f"prior_not_twin={n_prior}  maxIoU={float(r.max_iou):.3f}",
            fill=(160, 0, 0),
            font=fB,
        )
        d.text(
            (12, 40),
            "NEW=no prior not-twins in cluster. HAS_KEPT_NOT_TWIN=includes slide(s) you kept as not-twins. Not auto-dropped.",
            fill=(50, 50, 50),
            font=fS,
        )
        for i, sid in enumerate(mems):
            x = 12 + (i % cols) * (tw + 8)
            y = 70 + (i // cols) * (th + 40)
            sheet.paste(thumb(sid, (tw, th)), (x, y))
            d.text((x, y + th + 4), sid[:12], fill=(30, 30, 30), font=fT)
        outp = out_dir / f"cluster_{int(r.cluster_id):03d}.png"
        sheet.save(outp, optimize=True)
        paths.append(outp)
    return paths


def write_html(section: str, pages: list[Path], index_path: Path, blurb: str, root_out: Path | None = None) -> None:
    base = root_out or OUT
    rel = [p.relative_to(base).as_posix() for p in pages]
    cards = "\n".join(
        f'<div class="card"><h3>{p.name}</h3><a href="{p.as_posix()}"><img src="{p.as_posix()}" loading="lazy"></a></div>'
        for p in (Path(x) for x in rel)
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{section}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa}}
h1{{margin:0 0 8px}}
.card{{margin:24px 0;background:#fff;padding:12px;border:1px solid #ddd}}
img{{max-width:100%;height:auto;border:1px solid #ccc}}
nav a{{margin-right:12px}}
</style></head><body>
<nav>
<a href="index.html">Home</a>
<a href="safe_pairs/index.html">safe pairs</a>
<a href="multi_clusters/index.html">multi groups</a>
<a href="lower_iou/index.html">Lower IoU pairs</a>
</nav>
<h1>{section}</h1>
<p>{blurb}</p>
{cards}
</body></html>"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(html)


def main(dup_dir: Path | None = None, out_dir: Path | None = None) -> None:
    import argparse

    if dup_dir is None or out_dir is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dup-dir", type=Path, default=DUP)
        ap.add_argument("--out-dir", type=Path, default=None)
        args, _unknown = ap.parse_known_args()
        dup_dir = dup_dir or args.dup_dir
        out_dir = out_dir or args.out_dir or (Path(dup_dir) / "galleries")

    dup_dir = Path(dup_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe = pd.read_csv(dup_dir / "dedupe_safe_pairs_iou70.csv").sort_values("shape_iou", ascending=False)
    multi = pd.read_csv(dup_dir / "dedupe_multi_clusters_iou70.csv")
    lower = pd.read_csv(dup_dir / "dedupe_lower_iou_pairs_30_70.csv").sort_values("shape_iou", ascending=False)
    # invent keep/drop only if missing (rescan already provides them)
    if not {"keep_id", "drop_id"}.issubset(lower.columns):
        counts = {}
        for name in ("panda_train_pre_dedupe.csv", "panda_val_pre_dedupe.csv", "panda_test_pre_dedupe.csv"):
            path = PROJECT / "outputs" / "splits" / name
            if not path.exists():
                path = PROJECT / "outputs" / "splits" / name.replace("_pre_dedupe", "")
            df = pd.read_csv(path, usecols=["image_id"])
            for sid, n in df.groupby("image_id").size().items():
                counts[str(sid)] = counts.get(str(sid), 0) + int(n)
        keep, drop, kn, dn, sk, sd = [], [], [], [], [], []
        for r in lower.itertuples(index=False):
            a, b = r.image_id_a, r.image_id_b
            if counts.get(a, 0) >= counts.get(b, 0):
                keep.append(a); drop.append(b)
                kn.append(counts.get(a, 0)); dn.append(counts.get(b, 0))
                sk.append(r.split_a); sd.append(r.split_b)
            else:
                keep.append(b); drop.append(a)
                kn.append(counts.get(b, 0)); dn.append(counts.get(a, 0))
                sk.append(r.split_b); sd.append(r.split_a)
        lower = lower.assign(
            keep_id=keep,
            drop_id=drop,
            keep_n_patches=kn,
            drop_n_patches=dn,
            split_keep=sk,
            split_drop=sd,
            cross_split=[x != y for x, y in zip(sk, sd)],
        )

    print(f"Rendering safe={len(safe)} multi={len(multi)} lower={len(lower)} → {out_path}")
    p1 = render_pair_pages(safe, out_path / "safe_pairs", f"SAFE twins IoU≥0.70 (n={len(safe)}; review)")
    p2 = render_multi_pages(multi, out_path / "multi_clusters")
    p3 = render_pair_pages(lower, out_path / "lower_iou", f"LOWER IoU 0.30–0.70 exclusive (n={len(lower)})")

    write_html(
        f"{len(safe)} safe clear twins (IoU≥0.70)",
        p1,
        out_path / "safe_pairs" / "index.html",
        "KEEP/DROP labels are proposals only — not auto-applied. Red = cross-split. Prior not-twins are still included.",
        root_out=out_path,
    )
    write_html(
        f"{len(multi)} multi-member clusters (IoU≥0.70 chain)",
        p2,
        out_path / "multi_clusters" / "index.html",
        "Membership-disjoint from safe/lower. Not auto-dropped.",
        root_out=out_path,
    )
    write_html(
        f"Lower IoU pairs (0.30–0.70), n={len(lower)}",
        p3,
        out_path / "lower_iou" / "index.html",
        "Exclusive of any IoU≥0.70 component member. Not auto-dropped.",
        root_out=out_path,
    )
    (out_path / "index.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Dedupe galleries</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px}}a{{font-size:20px;display:block;margin:12px 0}}</style>
</head><body>
<h1>Duplicate review galleries (rescan)</h1>
<p>Detect-only. Safe/lower/multi are membership-disjoint. Prior not-twins included.</p>
<a href="safe_pairs/index.html">1) Safe twins IoU≥0.70 — {len(safe)} pairs</a>
<a href="multi_clusters/index.html">2) Multi-slide groups — {len(multi)} clusters</a>
<a href="lower_iou/index.html">3) Lower IoU pairs (0.30–0.70) — {len(lower)} pairs</a>
</body></html>"""
    )
    print(f"Done → open {out_path/'index.html'}")


if __name__ == "__main__":
    main()
