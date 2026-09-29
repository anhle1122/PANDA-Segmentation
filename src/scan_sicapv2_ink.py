#!/usr/bin/env python3
"""Scan official SICAPv2 patches for pen/ink (HSV heuristic + optional WSISegQC).

Writes:
  outputs/evaluation/sicapv2_ink/patch_ink_scan.csv
  outputs/evaluation/sicapv2_ink/slide_ink_summary.csv
  outputs/evaluation/sicapv2_ink/summary.json
  outputs/evaluation/sicapv2_ink/gallery/  (top flagged patches)

Usage:
  python src/scan_sicapv2_ink.py
  python src/scan_sicapv2_ink.py --with-wsisegqc --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import OUTPUTS  # noqa: E402

DEFAULT_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")
OUT_DIR = OUTPUTS / "evaluation" / "sicapv2_ink"

# HSV heuristics (OpenCV-style H in 0..180). Tuned for common biopsy ink.
# Fractions are of all pixels in the 512 patch.
HSV_THRESH = {
    "blue": dict(h=(90, 130), s_min=40, v_min=40, v_max=230),
    "green": dict(h=(35, 85), s_min=40, v_min=40, v_max=230),
    "black": dict(h=(0, 180), s_min=0, s_max=80, v_min=0, v_max=50),
}
FLAG_FRAC = {"blue": 0.002, "green": 0.002, "black": 0.01, "any": 0.003}


def rgb_to_hsv_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 H×W×3 → HSV uint8 with H in [0,180] (OpenCV-like)."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    rc = (maxc - r) / (maxc - minc + 1e-6)
    gc = (maxc - g) / (maxc - minc + 1e-6)
    bc = (maxc - b) / (maxc - minc + 1e-6)
    h = np.zeros_like(maxc)
    h = np.where((maxc == r) & (maxc > minc), (bc - gc) % 6.0, h)
    h = np.where((maxc == g) & (maxc > minc), 2.0 + rc - bc, h)
    h = np.where((maxc == b) & (maxc > minc), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    hsv = np.stack([h * 180.0, s * 255.0, v * 255.0], axis=-1)
    return np.clip(hsv, 0, 255).astype(np.uint8)


def hsv_ink_fractions(rgb: np.ndarray) -> dict[str, float]:
    hsv = rgb_to_hsv_u8(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = float(h.size)
    out = {}
    for name, t in HSV_THRESH.items():
        h0, h1 = t["h"]
        mask = (h >= h0) & (h <= h1) & (s >= t["s_min"]) & (v >= t["v_min"]) & (v <= t["v_max"])
        if "s_max" in t:
            mask &= s <= t["s_max"]
        out[name] = float(mask.sum()) / n
    out["any"] = float(max(out["blue"], out["green"], out["black"]))
    # union fraction
    hb = (h >= 90) & (h <= 130) & (s >= 40) & (v >= 40) & (v <= 230)
    hg = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 40) & (v <= 230)
    hk = (s <= 80) & (v <= 50)
    out["union"] = float((hb | hg | hk).sum()) / n
    return out


def slide_id_from_stem(stem: str) -> str:
    # 16B0001851_Block_Region_...
    return stem.split("_Block_")[0] if "_Block_" in stem else stem.split("_")[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Scan SICAPv2 patches for ink/pen")
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--with-wsisegqc", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--gallery-n", type=int, default=24)
    p.add_argument("--wsisegqc-min-px", type=int, default=10)
    args = p.parse_args()

    img_dir = args.sicap_root / "images"
    if not img_dir.is_dir():
        raise SystemExit(f"missing {img_dir}")
    paths = sorted(img_dir.glob("*.jpg"))
    if args.max_patches:
        paths = paths[: args.max_patches]
    print(f"scanning {len(paths)} patches under {img_dir}", flush=True)

    wsiseg = None
    pen_weights = None
    if args.with_wsisegqc:
        from pen_wsisegqc import DEFAULT_PEN_PT, infer_patch_pen  # noqa: WPS410

        # pen_wsisegqc resolves PROJECT from its file location (_code_mirror),
        # which does not contain external/. Prefer the real project weights.
        candidates = [
            Path("/common/omarmlab/members/anh/panda_project/external/wsisegqc/models/models/pen.pt"),
            DEFAULT_PEN_PT,
        ]
        pen_weights = next((c for c in candidates if c.is_file()), None)
        if pen_weights is None:
            raise SystemExit(f"pen.pt not found; tried {candidates}")
        wsiseg = infer_patch_pen
        print(f"WSISegQC pen.pt={pen_weights} device={args.device}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = args.out_dir / "gallery"
    gallery.mkdir(exist_ok=True)

    rows = []
    for i, path in enumerate(paths, 1):
        rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        fr = hsv_ink_fractions(rgb)
        flagged = (
            fr["blue"] >= FLAG_FRAC["blue"]
            or fr["green"] >= FLAG_FRAC["green"]
            or fr["black"] >= FLAG_FRAC["black"]
            or fr["union"] >= FLAG_FRAC["any"]
        )
        colors = ",".join(
            c for c in ("green", "blue", "black") if fr[c] >= FLAG_FRAC[c]
        )
        rec = {
            "patch": path.name,
            "slide_id": slide_id_from_stem(path.stem),
            "hsv_blue": round(fr["blue"], 6),
            "hsv_green": round(fr["green"], 6),
            "hsv_black": round(fr["black"], 6),
            "hsv_union": round(fr["union"], 6),
            "hsv_flagged": int(flagged),
            "hsv_colors": colors,
            "wsisegqc_pen_px": "",
            "wsisegqc_pen_pct": "",
            "wsisegqc_flagged": "",
        }
        if wsiseg is not None:
            m = wsiseg(
                rgb,
                device=args.device,
                weights=pen_weights,
                min_pen_px=args.wsisegqc_min_px,
            )
            rec["wsisegqc_pen_px"] = m["pen_px"]
            rec["wsisegqc_pen_pct"] = round(m["pen_pct_patch"], 4)
            rec["wsisegqc_flagged"] = int(m["pen_flagged"])
        rows.append(rec)
        if i % 1000 == 0:
            n_f = sum(r["hsv_flagged"] for r in rows)
            print(f"  {i}/{len(paths)} hsv_flagged={n_f}", flush=True)

    csv_path = args.out_dir / "patch_ink_scan.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # slide-level rollup
    by_slide: dict[str, list] = defaultdict(list)
    for r in rows:
        by_slide[r["slide_id"]].append(r)
    slide_rows = []
    for sid, lst in sorted(by_slide.items()):
        n = len(lst)
        n_hsv = sum(r["hsv_flagged"] for r in lst)
        mean_u = float(np.mean([r["hsv_union"] for r in lst]))
        max_u = float(np.max([r["hsv_union"] for r in lst]))
        n_wsi = sum(int(r["wsisegqc_flagged"] or 0) for r in lst) if wsiseg else ""
        slide_rows.append(
            {
                "slide_id": sid,
                "n_patches": n,
                "n_hsv_flagged": n_hsv,
                "pct_hsv_flagged": round(100.0 * n_hsv / n, 2),
                "mean_hsv_union": round(mean_u, 6),
                "max_hsv_union": round(max_u, 6),
                "n_wsisegqc_flagged": n_wsi,
                "slide_hsv_flagged": int(n_hsv > 0),
            }
        )
    slide_csv = args.out_dir / "slide_ink_summary.csv"
    with slide_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slide_rows[0].keys()))
        w.writeheader()
        w.writerows(slide_rows)

    n_hsv = sum(r["hsv_flagged"] for r in rows)
    n_slide_f = sum(r["slide_hsv_flagged"] for r in slide_rows)
    summary = {
        "n_patches": len(rows),
        "n_slides": len(slide_rows),
        "n_patches_hsv_flagged": n_hsv,
        "pct_patches_hsv_flagged": round(100.0 * n_hsv / max(len(rows), 1), 3),
        "n_slides_with_any_hsv_flag": n_slide_f,
        "pct_slides_with_any_hsv_flag": round(100.0 * n_slide_f / max(len(slide_rows), 1), 2),
        "flag_thresholds": FLAG_FRAC,
        "with_wsisegqc": bool(wsiseg),
        "wsisegqc_patches_flagged": (
            int(sum(int(r["wsisegqc_flagged"] or 0) for r in rows)) if wsiseg else None
        ),
        "top_slides_by_pct_flagged": sorted(
            slide_rows, key=lambda r: (-r["pct_hsv_flagged"], -r["max_hsv_union"])
        )[:15],
    }
    if wsiseg:
        summary["pct_patches_wsisegqc_flagged"] = round(
            100.0 * summary["wsisegqc_patches_flagged"] / max(len(rows), 1), 3
        )

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # gallery: highest union fraction among flagged (or overall)
    ranked = sorted(rows, key=lambda r: -r["hsv_union"])[: args.gallery_n]
    for j, r in enumerate(ranked):
        src = img_dir / r["patch"]
        dst = gallery / f"{j:02d}_u{r['hsv_union']:.4f}_{r['patch']}"
        try:
            Image.open(src).save(dst)
        except OSError:
            pass

    print(json.dumps({k: summary[k] for k in summary if k != "top_slides_by_pct_flagged"}, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {slide_csv}")
    print(f"Wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
