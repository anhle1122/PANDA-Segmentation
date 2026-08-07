#!/usr/bin/env python3
"""Detect near-duplicate WSIs by tissue silhouette shape + metadata.

PANDA/Radboud can contain the same core under different image_ids (re-crop /
re-scan). This scan fingerprints every clean slide, matches crop-robust shape
IoU (with flip/rot), and reports duplicate clusters vs train/val/test splits.

Usage:
  python src/scan_slide_duplicates.py            # full run
  python src/scan_slide_duplicates.py --resume   # reuse fingerprints.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from patch_utils import PROJECT

SLIDES_DIR = Path("/common/omarmlab/members/anh/panda_data/slides")
CLEAN_CSV = PROJECT / "data" / "radboud_clean.csv"
SPLITS_CSV = PROJECT / "outputs" / "splits" / "panda_slide_splits.csv"
OUT_DIR = PROJECT / "outputs" / "docs" / "slide_duplicates"
FP_PATH = OUT_DIR / "fingerprints.npz"
META_PATH = OUT_DIR / "fingerprint_meta.csv"
PAIRS_PATH = OUT_DIR / "duplicate_pairs.csv"
CLUSTERS_PATH = OUT_DIR / "duplicate_clusters.csv"
SUMMARY_PATH = OUT_DIR / "duplicate_summary.json"
CONTACT_PATH = OUT_DIR / "duplicate_clusters_contact.png"

SIL_SIZE = 128
THUMB_LONG = 512
# Empirically: flagged true pairs scored 0.33–0.83; unrelated ~0.05.
# Use 0.28 as review floor; 0.50 as high-confidence.
IOU_REVIEW = 0.28
IOU_HIGH = 0.50
HU_PREFILTER = 8.0  # max L2 on log-Hu for candidate pairs
MAX_CAND_PER_SLIDE = 40


def tissue_silhouette(slide_path: Path, size: int = SIL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    sl = openslide.OpenSlide(str(slide_path))
    dims = sl.dimensions
    thumb = np.array(sl.get_thumbnail((THUMB_LONG, THUMB_LONG)).convert("RGB"))
    sl.close()
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    tissue = ((gray < 235) & (hsv[:, :, 1] > 12)).astype(np.uint8) * 255
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    keep = np.zeros_like(tissue)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        thr = max(50, 0.01 * float(areas.max()))
        for i, ar in enumerate(areas, 1):
            if ar >= thr:
                keep[lab == i] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8), np.array(dims, dtype=np.int64)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = keep[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    sil = np.zeros((size, size), np.uint8)
    sil[(size - nh) // 2 : (size - nh) // 2 + nh, (size - nw) // 2 : (size - nw) // 2 + nw] = rs
    return sil, np.array(dims, dtype=np.int64)


def hu_log(sil: np.ndarray) -> np.ndarray:
    m = cv2.moments(sil)
    hu = cv2.HuMoments(m).flatten()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-12)).astype(np.float32)


def best_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    orients = [
        b,
        np.fliplr(b),
        np.flipud(b),
        np.fliplr(np.flipud(b)),
        np.rot90(b, 1),
        np.rot90(b, 2),
        np.rot90(b, 3),
        np.fliplr(np.rot90(b, 1)),
        np.fliplr(np.rot90(b, 3)),
    ]
    best = 0.0
    aa = int(a.sum())
    for c in orients:
        c = c > 0
        inter = int((a & c).sum())
        union = aa + int(c.sum()) - inter
        if union:
            best = max(best, inter / union)
    return float(best)


def build_fingerprints(meta: pd.DataFrame, resume: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if resume and FP_PATH.exists() and META_PATH.exists():
        data = np.load(FP_PATH)
        sils, hus, dims = data["sils"], data["hus"], data["dims"]
        fp_meta = pd.read_csv(META_PATH)
        done = set(fp_meta["image_id"].astype(str))
        todo = meta[~meta["image_id"].astype(str).isin(done)].copy()
        print(f"Resume: {len(done)} fingerprints on disk, {len(todo)} remaining")
        if len(todo) == 0:
            return sils, hus, dims, fp_meta
        rows = fp_meta.to_dict("records")
        sil_list = [sils[i] for i in range(len(sils))]
        hu_list = [hus[i] for i in range(len(hus))]
        dim_list = [dims[i] for i in range(len(dims))]
    else:
        todo = meta.copy()
        rows, sil_list, hu_list, dim_list = [], [], [], []

    misses = 0
    for r in tqdm(todo.itertuples(index=False), total=len(todo), desc="fingerprints"):
        sid = str(r.image_id)
        paths = list(SLIDES_DIR.glob(f"{sid[:12]}*.tiff"))
        if not paths:
            misses += 1
            continue
        try:
            sil, dims = tissue_silhouette(paths[0])
        except Exception as e:
            print(f"  fail {sid[:12]}: {e}")
            misses += 1
            continue
        sil_list.append(sil)
        hu_list.append(hu_log(sil))
        dim_list.append(dims)
        rows.append(
            {
                "image_id": sid,
                "isup_grade": int(r.isup_grade) if str(r.isup_grade).isdigit() else -1,
                "gleason_score": str(r.gleason_score),
                "split": str(getattr(r, "split", "unknown")),
                "width": int(dims[0]),
                "height": int(dims[1]),
            }
        )
        if len(rows) % 200 == 0:
            _save_fp(sil_list, hu_list, dim_list, rows)

    _save_fp(sil_list, hu_list, dim_list, rows)
    print(f"Fingerprints done: {len(rows)} slides ({misses} missing/failed)")
    sils = np.stack(sil_list).astype(np.uint8)
    hus = np.stack(hu_list).astype(np.float32)
    dims = np.stack(dim_list).astype(np.int64)
    return sils, hus, dims, pd.DataFrame(rows)


def _save_fp(sil_list, hu_list, dim_list, rows) -> None:
    np.savez_compressed(
        FP_PATH,
        sils=np.stack(sil_list).astype(np.uint8),
        hus=np.stack(hu_list).astype(np.float32),
        dims=np.stack(dim_list).astype(np.int64),
    )
    pd.DataFrame(rows).to_csv(META_PATH, index=False)


def match_pairs(sils: np.ndarray, hus: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    n = len(meta)
    # Prefer same gleason; also allow same ISUP if gleason missing/odd
    gleason = meta["gleason_score"].astype(str).tolist()
    isup = meta["isup_grade"].astype(int).tolist()
    ids = meta["image_id"].astype(str).tolist()
    splits = meta["split"].astype(str).tolist()
    widths = meta["width"].astype(int).to_numpy()
    heights = meta["height"].astype(int).to_numpy()
    aspects = widths / np.maximum(heights, 1)

    pairs = []
    for i in tqdm(range(n), desc="match"):
        # candidate mask: same gleason OR (same isup & aspect within 25%)
        same_g = [j for j in range(i + 1, n) if gleason[j] == gleason[i]]
        if len(same_g) > MAX_CAND_PER_SLIDE * 3:
            # prefilter by Hu distance
            d = np.linalg.norm(hus[same_g] - hus[i], axis=1)
            order = np.argsort(d)[:MAX_CAND_PER_SLIDE]
            cands = [same_g[k] for k in order if d[k] <= HU_PREFILTER]
        else:
            d = np.linalg.norm(hus[same_g] - hus[i], axis=1) if same_g else np.array([])
            cands = [same_g[k] for k in range(len(same_g)) if d[k] <= HU_PREFILTER]

        # add a few same-ISUP near-aspect if gleason pool is tiny
        if len(cands) < 5:
            for j in range(i + 1, n):
                if isup[j] != isup[i]:
                    continue
                if abs(aspects[j] - aspects[i]) / max(aspects[i], 1e-6) > 0.35:
                    continue
                if np.linalg.norm(hus[j] - hus[i]) > HU_PREFILTER:
                    continue
                if j not in cands:
                    cands.append(j)
                if len(cands) >= MAX_CAND_PER_SLIDE:
                    break

        for j in cands[:MAX_CAND_PER_SLIDE]:
            iou = best_iou(sils[i], sils[j])
            if iou < IOU_REVIEW:
                continue
            cross = splits[i] != splits[j] and splits[i] != "unknown" and splits[j] != "unknown"
            pairs.append(
                {
                    "image_id_a": ids[i],
                    "image_id_b": ids[j],
                    "split_a": splits[i],
                    "split_b": splits[j],
                    "isup_a": isup[i],
                    "isup_b": isup[j],
                    "gleason_a": gleason[i],
                    "gleason_b": gleason[j],
                    "dims_a": f"{widths[i]}x{heights[i]}",
                    "dims_b": f"{widths[j]}x{heights[j]}",
                    "shape_iou": round(iou, 4),
                    "hu_l2": round(float(np.linalg.norm(hus[i] - hus[j])), 4),
                    "same_gleason": gleason[i] == gleason[j],
                    "same_isup": isup[i] == isup[j],
                    "cross_split": cross,
                    "confidence": "high" if iou >= IOU_HIGH else "review",
                }
            )
    return pd.DataFrame(pairs)


def cluster_pairs(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in pairs.itertuples(index=False):
        union(r.image_id_a, r.image_id_b)

    groups: dict[str, list[str]] = {}
    for sid in set(pairs["image_id_a"]) | set(pairs["image_id_b"]):
        groups.setdefault(find(sid), []).append(sid)

    split_map = dict(zip(meta["image_id"].astype(str), meta["split"].astype(str)))
    isup_map = dict(zip(meta["image_id"].astype(str), meta["isup_grade"]))
    gle_map = dict(zip(meta["image_id"].astype(str), meta["gleason_score"].astype(str)))
    dim_map = {
        str(r.image_id): f"{int(r.width)}x{int(r.height)}" for r in meta.itertuples(index=False)
    }

    # max iou inside cluster
    iou_lookup = {}
    for r in pairs.itertuples(index=False):
        iou_lookup[(r.image_id_a, r.image_id_b)] = r.shape_iou
        iou_lookup[(r.image_id_b, r.image_id_a)] = r.shape_iou

    rows = []
    for ci, (_root, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        members = sorted(members)
        splits = sorted({split_map.get(m, "?") for m in members})
        cross = len(set(splits) - {"unknown", "?"}) > 1
        # representative max pairwise iou
        max_iou = 0.0
        for a_i, a in enumerate(members):
            for b in members[a_i + 1 :]:
                max_iou = max(max_iou, iou_lookup.get((a, b), 0.0))
        rows.append(
            {
                "cluster_id": ci,
                "n_members": len(members),
                "image_ids": ";".join(members),
                "splits": ";".join(splits),
                "cross_split": cross,
                "isup_grades": ";".join(str(isup_map.get(m, "?")) for m in members),
                "gleason_scores": ";".join(str(gle_map.get(m, "?")) for m in members),
                "dims": ";".join(dim_map.get(m, "?") for m in members),
                "max_shape_iou": round(max_iou, 4),
                "confidence": "high" if max_iou >= IOU_HIGH else "review",
            }
        )
    return pd.DataFrame(rows)


def render_contact(clusters: pd.DataFrame, max_clusters: int = 24) -> None:
    if len(clusters) == 0:
        return
    try:
        font_b = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        font_s = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_b = font_s = ImageFont.load_default()

    show = clusters.head(max_clusters)
    thumb_w, thumb_h = 140, 220
    max_members = int(show["n_members"].max())
    cols = min(max_members, 6)
    sheet = Image.new(
        "RGB",
        (cols * (thumb_w + 8) + 220, len(show) * (thumb_h + 36) + 60),
        (255, 255, 255),
    )
    d = ImageDraw.Draw(sheet)
    d.text(
        (12, 10),
        "Duplicate clusters (shape IoU) — red label = cross-split leakage",
        fill=(10, 10, 10),
        font=font_b,
    )
    for ri, r in enumerate(show.itertuples(index=False)):
        y = 50 + ri * (thumb_h + 36)
        members = r.image_ids.split(";")[:cols]
        color = (180, 20, 20) if r.cross_split else (20, 20, 20)
        d.text(
            (12, y),
            f"C{r.cluster_id} n={r.n_members} IoU={r.max_shape_iou} splits={r.splits}",
            fill=color,
            font=font_s,
        )
        for mi, sid in enumerate(members):
            paths = list(SLIDES_DIR.glob(f"{sid[:12]}*.tiff"))
            if not paths:
                continue
            sl = openslide.OpenSlide(str(paths[0]))
            im = sl.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
            sl.close()
            im.thumbnail((thumb_w, thumb_h))
            canvas = Image.new("RGB", (thumb_w, thumb_h), (240, 240, 240))
            canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(canvas, (200 + mi * (thumb_w + 8), y + 18))
            d.text(
                (200 + mi * (thumb_w + 8), y + thumb_h + 18),
                sid[:8],
                fill=(60, 60, 60),
                font=font_s,
            )
    sheet.save(CONTACT_PATH)
    print(f"Wrote {CONTACT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iou-review", type=float, default=IOU_REVIEW)
    ap.add_argument("--skip-contact", action="store_true")
    args = ap.parse_args()
    global IOU_REVIEW
    IOU_REVIEW = float(args.iou_review)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(CLEAN_CSV, dtype={"image_id": str})
    splits = pd.read_csv(SPLITS_CSV, dtype={"image_id": str})
    meta = clean.merge(splits[["image_id", "split"]], on="image_id", how="left")
    meta["split"] = meta["split"].fillna("not_in_splits")
    print(f"Clean slides: {len(meta)}  | in splits: {(meta.split != 'not_in_splits').sum()}")

    sils, hus, dims, fp_meta = build_fingerprints(meta, resume=args.resume)
    # align fp_meta split from latest splits file
    smap = dict(zip(meta["image_id"].astype(str), meta["split"].astype(str)))
    fp_meta["split"] = fp_meta["image_id"].map(smap).fillna(fp_meta["split"])

    pairs = match_pairs(sils, hus, fp_meta)
    pairs = pairs.sort_values(["cross_split", "shape_iou"], ascending=[False, False])
    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"Pairs ≥{IOU_REVIEW}: {len(pairs)}  high≥{IOU_HIGH}: {(pairs.confidence=='high').sum() if len(pairs) else 0}")
    print(f"Cross-split pairs: {int(pairs.cross_split.sum()) if len(pairs) else 0}")

    clusters = cluster_pairs(pairs, fp_meta)
    if len(clusters):
        clusters = clusters.sort_values(["cross_split", "n_members", "max_shape_iou"], ascending=[False, False, False])
        clusters.to_csv(CLUSTERS_PATH, index=False)

    # summary stats
    n_slides_in_dup = 0
    if len(clusters):
        n_slides_in_dup = len(set(";".join(clusters["image_ids"]).split(";")))
    split_pair_counts = {}
    if len(pairs):
        for r in pairs.itertuples(index=False):
            key = "↔".join(sorted([r.split_a, r.split_b]))
            split_pair_counts[key] = split_pair_counts.get(key, 0) + 1

    summary = {
        "n_fingerprinted": int(len(fp_meta)),
        "iou_review": IOU_REVIEW,
        "iou_high": IOU_HIGH,
        "n_pairs_review": int(len(pairs)),
        "n_pairs_high": int((pairs.confidence == "high").sum()) if len(pairs) else 0,
        "n_pairs_cross_split": int(pairs.cross_split.sum()) if len(pairs) else 0,
        "n_clusters": int(len(clusters)),
        "n_clusters_cross_split": int(clusters.cross_split.sum()) if len(clusters) else 0,
        "n_slides_in_any_duplicate_cluster": n_slides_in_dup,
        "pair_counts_by_split_combo": split_pair_counts,
        "known_user_pairs_recovered": {},
    }
    # check user-flagged pairs recovered
    flagged = [
        ("48440a60", "4889d110"),
        ("dd11c914", "72e64850"),
        ("e707ef8c", "507ac341"),
    ]
    if len(pairs):
        for a, b in flagged:
            hit = pairs[
                (pairs.image_id_a.str.startswith(a) & pairs.image_id_b.str.startswith(b))
                | (pairs.image_id_a.str.startswith(b) & pairs.image_id_b.str.startswith(a))
            ]
            summary["known_user_pairs_recovered"][f"{a}/{b}"] = (
                float(hit.shape_iou.iloc[0]) if len(hit) else None
            )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.skip_contact and len(clusters):
        render_contact(clusters)


if __name__ == "__main__":
    main()
