#!/usr/bin/env python3
"""Before/after ISUP class mix, duplicate leakage into val/test, and n>=5 mask-ISUP check.

Writes JSON/CSV under outputs/docs/slide_duplicates/audit_after_dedupe/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from isup_diagnostic import derive_grade, gleason_to_isup
from patch_utils import PROJECT

DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
SPLITS = PROJECT / "outputs" / "splits"
H5_DIR = PROJECT / "outputs" / "kept_extract" / "raw"
OUT = DUP / "audit_after_dedupe"
META = PROJECT / "data" / "train.csv"


def pct_table(s: pd.Series) -> dict:
    vc = s.value_counts().sort_index()
    n = int(vc.sum())
    return {
        "n": n,
        "counts": {str(int(k)): int(v) for k, v in vc.items()},
        "pct": {str(int(k)): round(100.0 * float(v) / n, 2) for k, v in vc.items()} if n else {},
    }


def load_suspect_edges() -> pd.DataFrame:
    """Undirected suspect pairs with IoU when available."""
    rows = []
    safe = pd.read_csv(DUP / "dedupe_safe_pairs_iou70.csv")
    for r in safe.itertuples(index=False):
        rows.append(
            dict(
                a=str(r.keep_id),
                b=str(r.drop_id),
                source="safe_iou70",
                shape_iou=float(r.shape_iou),
            )
        )
    lower = pd.read_csv(DUP / "dedupe_lower_iou_pairs_30_70.csv")
    for r in lower.itertuples(index=False):
        rows.append(
            dict(
                a=str(r.image_id_a),
                b=str(r.image_id_b),
                source="lower_iou",
                shape_iou=float(r.shape_iou),
            )
        )
    pairs = pd.read_csv(DUP / "dedupe_pairs_shape_isup.csv")
    multi = pd.read_csv(DUP / "dedupe_multi_clusters_iou70.csv")
    multi_ids = set()
    for s in multi.image_ids:
        multi_ids.update(str(s).split(";"))
    # direct multi edges at IoU>=0.70
    for r in pairs.itertuples(index=False):
        a, b = str(r.image_id_a), str(r.image_id_b)
        if a in multi_ids and b in multi_ids and float(r.shape_iou) >= 0.70:
            rows.append(dict(a=a, b=b, source="multi_edge_iou70", shape_iou=float(r.shape_iou)))
    return pd.DataFrame(rows)


def leakage_report(sp: pd.DataFrame, edges: pd.DataFrame, patch_counts: dict[str, int], tag: str) -> dict:
    split = sp.set_index("image_id")["split"].to_dict()
    alive = set(split)
    cross = []
    for r in edges.itertuples(index=False):
        if r.a not in alive or r.b not in alive:
            continue
        sa, sb = split[r.a], split[r.b]
        if sa == sb:
            continue
        if {sa, sb} == {"train", "val"} or {sa, sb} == {"train", "test"} or {sa, sb} == {"val", "test"}:
            cross.append(
                dict(
                    a=r.a,
                    b=r.b,
                    split_a=sa,
                    split_b=sb,
                    source=r.source,
                    shape_iou=float(r.shape_iou),
                )
            )
    cdf = pd.DataFrame(cross)
    val_ids = set(sp.loc[sp.split == "val", "image_id"])
    test_ids = set(sp.loc[sp.split == "test", "image_id"])
    train_ids = set(sp.loc[sp.split == "train", "image_id"])

    leaked_val = set()
    leaked_test = set()
    for r in cross:
        sides = {r["split_a"]: r["a"], r["split_b"]: r["b"]}
        if "train" in sides and "val" in sides:
            leaked_val.add(sides["val"])
        if "train" in sides and "test" in sides:
            leaked_test.add(sides["test"])
        if "val" in sides and "test" in sides:
            # val↔test also contamination between eval sets
            leaked_val.add(sides["val"])
            leaked_test.add(sides["test"])

    def patch_frac(ids: set[str], universe: set[str]) -> tuple[int, int, float]:
        p_ids = sum(patch_counts.get(i, 0) for i in ids)
        p_u = sum(patch_counts.get(i, 0) for i in universe)
        return p_ids, p_u, (100.0 * p_ids / p_u if p_u else 0.0)

    pv, pvall, pv_pct = patch_frac(leaked_val, val_ids)
    pt, ptall, pt_pct = patch_frac(leaked_test, test_ids)

    return {
        "tag": tag,
        "n_cross_split_edges": int(len(cdf)),
        "n_val_slides": int(len(val_ids)),
        "n_test_slides": int(len(test_ids)),
        "n_leaked_val_slides": int(len(leaked_val)),
        "n_leaked_test_slides": int(len(leaked_test)),
        "pct_val_slides_leaked": round(100.0 * len(leaked_val) / len(val_ids), 2) if val_ids else 0.0,
        "pct_test_slides_leaked": round(100.0 * len(leaked_test) / len(test_ids), 2) if test_ids else 0.0,
        "pct_val_patches_leaked": round(pv_pct, 2),
        "pct_test_patches_leaked": round(pt_pct, 2),
        "leaked_val_patches": int(pv),
        "leaked_test_patches": int(pt),
        "mean_iou_cross_edges": round(float(cdf.shape_iou.mean()), 4) if len(cdf) else None,
        "by_source": cdf.groupby("source").size().astype(int).to_dict() if len(cdf) else {},
        "leaked_val_ids": sorted(leaked_val),
        "leaked_test_ids": sorted(leaked_test),
    }


def dice_leak_proxy(leak: dict, val_cancer_dice: float, train_like_dice: float) -> dict:
    """Approximate how much leaked val patches inflate reported cancer Dice.

    contaminated_dice ≈ (1-f)*true + f*train_like
    => true ≈ (reported - f*train_like)/(1-f)
    Using train_like ≈ reported val as upper (optimistic) and 0.85 as pessimistic twin match.
    """
    f = leak["pct_val_patches_leaked"] / 100.0
    out = {"val_cancer_dice_reported": val_cancer_dice, "f_val_patches_leaked": round(f, 4)}
    for name, d_twin in [("twin_equals_reported", val_cancer_dice), ("twin_dice_0.85", 0.85), ("twin_dice_0.90", 0.90)]:
        if f >= 1:
            true = None
        else:
            true = (val_cancer_dice - f * d_twin) / (1 - f)
        out[name] = {
            "assumed_twin_dice": d_twin,
            "implied_clean_val_dice": None if true is None else round(true, 4),
            "approx_inflation_points": None if true is None else round(val_cancer_dice - true, 4),
        }
    return out


def metadata_isup_from_gleason(gleason: str) -> int:
    g = str(gleason).strip().lower()
    if g in {"negative", "0+0", "0", "nan", ""}:
        return 0
    a, b = g.split("+")
    return gleason_to_isup(int(a), int(b))


def h5_path(sid: str) -> Path | None:
    p = H5_DIR / f"{sid}_kept_raw.h5"
    if p.exists():
        return p
    hits = list(H5_DIR.glob(f"{sid[:12]}*_kept_raw.h5"))
    return hits[0] if hits else None


def mask_isup_for_split(
    split_csv: Path,
    *,
    min_patches: int,
    min_area_pct: float,
) -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    meta = pd.read_csv(META).set_index("image_id")
    # group coords
    groups = {sid: g[["x", "y"]].to_numpy() for sid, g in split.groupby("image_id")}
    rows = []
    missing_h5 = 0
    for sid, coords in tqdm(groups.items(), desc="mask-isup H5"):
        if len(coords) < min_patches:
            continue
        path = h5_path(str(sid))
        if path is None:
            missing_h5 += 1
            continue
        want = {(int(x), int(y)) for x, y in coords}
        counts = np.zeros(6, dtype=np.int64)
        n_used = 0
        with h5py.File(path, "r") as f:
            h5_coords = f["coords"][:]
            masks = f["masks"]
            for i, (x, y) in enumerate(h5_coords):
                key = (int(x), int(y))
                if key not in want:
                    continue
                m = np.asarray(masks[i])
                m = np.clip(m, 0, 5).astype(np.int64)
                counts += np.bincount(m.ravel(), minlength=6)[:6]
                n_used += 1
        if n_used < min_patches:
            continue
        gleason, derived = derive_grade(counts, min_area_pct=min_area_pct)
        meta_g = str(meta.loc[sid, "gleason_score"]) if sid in meta.index else ""
        meta_i = metadata_isup_from_gleason(meta_g) if meta_g else -1
        rows.append(
            dict(
                slide_id=str(sid),
                n_patches_in_split=int(len(coords)),
                n_patches_used=int(n_used),
                metadata_gleason=meta_g,
                metadata_isup=int(meta_i),
                mask_derived_gleason=gleason,
                mask_derived_isup=int(derived),
                match=bool(int(derived) == int(meta_i)),
                min_area_pct=float(min_area_pct),
                pix0=int(counts[0]),
                pix1=int(counts[1]),
                pix2=int(counts[2]),
                pix3=int(counts[3]),
                pix4=int(counts[4]),
                pix5=int(counts[5]),
            )
        )
    out = pd.DataFrame(rows)
    out.attrs["missing_h5"] = missing_h5
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-isup", action="store_true")
    ap.add_argument("--min-patches", type=int, default=5)
    ap.add_argument("--min-area-pct", type=float, default=0.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(SPLITS / "panda_slide_splits_pre_dedupe.csv", dtype={"image_id": str})
    after = pd.read_csv(SPLITS / "panda_slide_splits.csv", dtype={"image_id": str})
    before["isup_grade"] = before["isup_grade"].astype(int)
    after["isup_grade"] = after["isup_grade"].astype(int)

    # patch counts
    def patch_counts_from(prefix: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for split in ("train", "val", "test"):
            path = SPLITS / f"panda_{split}{prefix}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["image_id"])
            for sid, n in df.groupby("image_id").size().items():
                out[str(sid)] = out.get(str(sid), 0) + int(n)
        return out

    pc_before = patch_counts_from("_pre_dedupe")
    pc_after = patch_counts_from("")

    # --- class mix ISUP 0-5 ---
    class_report = {
        "before_pre_dedupe_slides": {
            "overall": pct_table(before["isup_grade"]),
            "by_split": {s: pct_table(before.loc[before.split == s, "isup_grade"]) for s in ("train", "val", "test")},
        },
        "after_current_slides": {
            "overall": pct_table(after["isup_grade"]),
            "by_split": {s: pct_table(after.loc[after.split == s, "isup_grade"]) for s in ("train", "val", "test")},
        },
    }
    # patch-weighted ISUP (each patch inherits slide ISUP)
    def patch_weighted(sp: pd.DataFrame, pc: dict[str, int]) -> dict:
        rows = []
        for r in sp.itertuples(index=False):
            n = pc.get(str(r.image_id), 0)
            if n:
                rows.append((int(r.isup_grade), n))
        if not rows:
            return pct_table(pd.Series(dtype=int))
        labels = np.repeat([a for a, _ in rows], [b for _, b in rows])
        return pct_table(pd.Series(labels))

    class_report["before_pre_dedupe_patch_weighted"] = patch_weighted(before, pc_before)
    class_report["after_current_patch_weighted"] = patch_weighted(after, pc_after)

    # delta overall slide pct
    b_pct = class_report["before_pre_dedupe_slides"]["overall"]["pct"]
    a_pct = class_report["after_current_slides"]["overall"]["pct"]
    class_report["delta_slide_pct_points"] = {
        k: round(float(a_pct.get(k, 0)) - float(b_pct.get(k, 0)), 2) for k in sorted(set(b_pct) | set(a_pct))
    }

    (OUT / "isup_class_mix_before_after.json").write_text(json.dumps(class_report, indent=2))

    # --- leakage ---
    edges = load_suspect_edges()
    leak_before = leakage_report(before, edges, pc_before, "pre_dedupe")
    leak_after = leakage_report(after, edges, pc_after, "current_after_suspects_train")
    # drop huge id lists from printed summary copy
    leak_before_ids = {
        "leaked_val_ids": leak_before.pop("leaked_val_ids"),
        "leaked_test_ids": leak_before.pop("leaked_test_ids"),
    }
    leak_after_ids = {
        "leaked_val_ids": leak_after.pop("leaked_val_ids"),
        "leaked_test_ids": leak_after.pop("leaked_test_ids"),
    }
    pd.Series(leak_before_ids["leaked_val_ids"]).to_csv(OUT / "leaked_val_ids_before.csv", index=False, header=["image_id"])
    pd.Series(leak_before_ids["leaked_test_ids"]).to_csv(OUT / "leaked_test_ids_before.csv", index=False, header=["image_id"])

    # reported val dice from baseline (era of leaked splits)
    val_cancer_dice = 0.702824
    dice_proxy = dice_leak_proxy(leak_before, val_cancer_dice, train_like_dice=0.85)

    leak_bundle = {
        "before": leak_before,
        "after": leak_after,
        "dice_inflation_proxy_before": dice_proxy,
        "note": (
            "No per-slide Dice exists for the leaked-era val set; proxy estimates clean val Dice "
            "if leaked patches scored like near-duplicates (assumed twin Dice)."
        ),
    }
    (OUT / "leakage_before_after.json").write_text(json.dumps(leak_bundle, indent=2))

    # --- ISUP sufficiency n>=5 on current train ---
    isup_summary = None
    if not args.skip_isup:
        df = mask_isup_for_split(
            SPLITS / "panda_train.csv",
            min_patches=args.min_patches,
            min_area_pct=args.min_area_pct,
        )
        df.to_csv(OUT / "mask_isup_train_nge5_thr0.csv", index=False)
        # also all train slides n>=1 for context
        n_train_slides = after.loc[after.split == "train", "image_id"].nunique()
        n_ge5 = int((pd.read_csv(SPLITS / "panda_train.csv").groupby("image_id").size() >= args.min_patches).sum())
        isup_summary = {
            "min_patches": args.min_patches,
            "min_area_pct": args.min_area_pct,
            "n_train_slides_total": int(n_train_slides),
            "n_train_slides_nge5": int(n_ge5),
            "n_scored": int(len(df)),
            "n_match": int(df["match"].sum()) if len(df) else 0,
            "match_rate": float(df["match"].mean()) if len(df) else None,
            "per_clinical_isup_match_rate": {
                str(k): round(float(v), 4) for k, v in df.groupby("metadata_isup")["match"].mean().items()
            }
            if len(df)
            else {},
            "pixel_class_pct_among_scored": {},
        }
        if len(df):
            pix = df[[f"pix{i}" for i in range(6)]].sum()
            tot = float(pix.sum())
            isup_summary["pixel_class_pct_among_scored"] = {
                f"G{i}": round(100.0 * float(pix[f"pix{i}"]) / tot, 2) for i in range(6)
            }
        (OUT / "mask_isup_train_nge5_thr0_summary.json").write_text(json.dumps(isup_summary, indent=2))

    summary = {
        "class_mix_path": str(OUT / "isup_class_mix_before_after.json"),
        "leakage_path": str(OUT / "leakage_before_after.json"),
        "isup_path": str(OUT / "mask_isup_train_nge5_thr0_summary.json") if isup_summary else None,
        "slide_counts_before": before.split.value_counts().to_dict(),
        "slide_counts_after": after.split.value_counts().to_dict(),
        "class_delta_slide_pct_points": class_report["delta_slide_pct_points"],
        "leakage_before_pct_val_slides": leak_before["pct_val_slides_leaked"],
        "leakage_before_pct_test_slides": leak_before["pct_test_slides_leaked"],
        "leakage_after_pct_val_slides": leak_after["pct_val_slides_leaked"],
        "leakage_after_pct_test_slides": leak_after["pct_test_slides_leaked"],
        "dice_proxy": dice_proxy,
        "isup_summary": isup_summary,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
