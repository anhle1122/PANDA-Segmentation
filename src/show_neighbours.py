#!/usr/bin/env python3
"""Render a slide next to its top embedding neighbours, for eyeball verification.

Page 153 paired `eab6d4ea` with two partners the user rejected, while the
embedding's actual top match was a slide that was never on the page. This draws
that comparison so the claim can be checked rather than taken on trust.
"""
from __future__ import annotations

import argparse

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont

from patch_utils import PROJECT

G = PROJECT / "outputs" / "docs" / "slide_groups"
SLIDES = PROJECT / "data" / "slides"
TW, TH = 300, 560
# twins on page 153 sat at margin >= 0.0115; every pair the user rejected was <= 0.0061
MARGIN_TWIN = 0.010


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def thumb(sid: str) -> Image.Image:
    canvas = Image.new("RGB", (TW, TH), (240, 240, 240))
    hits = list(SLIDES.glob(f"{sid[:12]}*.tiff"))
    if not hits:
        return canvas
    sl = openslide.OpenSlide(str(hits[0]))
    im = sl.get_thumbnail((TW, TH)).convert("RGB")
    sl.close()
    im.thumbnail((TW, TH))
    canvas.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(G / "verify_page153.npz"))
    ap.add_argument("--anchor", default="eab6d4eae42fed6ed8ce52aa6b313a1c")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--out", default=str(G / "eab6d4ea_neighbours.png"))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    sim = d["sim"]
    idx = {s: i for i, s in enumerate(ids)}
    a = idx[args.anchor]

    order = np.argsort(-sim[a])
    panels = [(args.anchor, None, "ANCHOR")]
    for j in order[: args.top]:
        panels.append((ids[j], float(sim[a, j]), f"top-{len(panels)}"))
    for sid in args.also:
        full = next((s for s in ids if s.startswith(sid)), None)
        if full:
            panels.append((full, float(sim[a, idx[full]]), "gallery partner"))

    W = TW * len(panels) + 20 * (len(panels) + 1)
    sheet = Image.new("RGB", (W, TH + 156), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((20, 14), f"{args.anchor[:12]} — embedding neighbours vs what the gallery paired it with",
            fill="black", font=font(21, True))

    # Absolute cosine says nothing on its own -- generic cores reach 0.95. What
    # marks a duplicate is the top match standing clear of the runner-up.
    ranked = np.sort(sim[a])[::-1]
    margin = float(ranked[0] - ranked[1])
    is_twin = margin >= MARGIN_TWIN
    dr.text(
        (20, 44),
        f"top-1 {ranked[0]:.4f}  top-2 {ranked[1]:.4f}  MARGIN {margin:+.4f}  ->  "
        + ("HAS A TWIN" if is_twin else "NO TWIN (flat neighbourhood: every match is generic)"),
        fill="#b00000" if is_twin else "#0a6",
        font=font(17, True),
    )

    for k, (sid, cos, tag) in enumerate(panels):
        x = 20 + k * (TW + 20)
        sheet.paste(thumb(sid), (x, 100))
        dr.text((x, 78), tag, fill="#0a6", font=font(16, True))
        label = sid[:12] if cos is None else f"{sid[:12]}  cos={cos:.4f}"
        dr.text((x, TH + 108), label, fill="black", font=font(15, True))
        if cos is not None:
            gap = float(ranked[0] - cos)
            dr.text(
                (x, TH + 128),
                "= top match" if gap == 0 else f"{gap:.4f} below top match",
                fill="#555555",
                font=font(13),
            )

    sheet.save(args.out)
    print("wrote", args.out)
    print(f"\ntop {args.top + 3} neighbours of {args.anchor[:12]}:")
    for j in order[: args.top + 3]:
        print(f"  {ids[j][:12]}  cos={sim[a, j]:.4f}")


if __name__ == "__main__":
    main()
