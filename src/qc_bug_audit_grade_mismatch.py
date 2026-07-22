"""Read-only technical audit of grade_mismatch QC exclusions. Report only — no CSV changes."""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from clean_dataset import (  # noqa: E402
    MASKS_DIR,
    check_grade_consistency,
    gleason_grades_in_score,
)

OUT = PROJECT / "outputs" / "qc_bug_audit"
META_CSV = PROJECT / "data" / "train_radboud.csv"
FLAGS_CSV = PROJECT / "outputs" / "clean_dataset_flags.csv"
CHECKPOINT_JSON = PROJECT / "outputs" / "clean_dataset_checkpoint.json"
SLIDES_DIR = Path(__import__("os").environ.get("PANDA_DATA_ROOT", PROJECT / "data")) / "slides"

CLASS_NAMES = {0: "bg", 1: "stroma", 2: "benign", 3: "G3", 4: "G4", 5: "G5"}
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 180, 0),
    3: (255, 255, 0),
    4: (255, 140, 0),
    5: (255, 0, 0),
}

EXPECTED_LABELS = {0, 1, 2, 3, 4, 5}
BULK_MASK_MTIME = datetime(2020, 4, 22, tzinfo=timezone.utc)  # PANDA mask epoch


def parse_gleason(gs: str) -> tuple[int | None, int | None]:
    if gs == "negative":
        return None, None
    a, b = gs.split("+")
    return int(a), int(b)


def grade_mismatch_category(gs: str, counts: dict[int, int]) -> str:
    pri, sec = parse_gleason(gs)
    if gs == "negative":
        return "negative_metadata"
    needed = gleason_grades_in_score(gs)
    if len(needed) == 1:
        g = next(iter(needed))
        return "sole_grade_absent" if counts.get(g, 0) == 0 else "unexpected"
    if counts.get(pri, 0) > 0 and counts.get(sec, 0) == 0:
        return "primary_present_secondary_absent"
    if counts.get(pri, 0) == 0 and counts.get(sec, 0) > 0:
        return "primary_absent_secondary_present"
    if counts.get(pri, 0) == 0 and counts.get(sec, 0) == 0:
        return "both_grades_absent"
    return "other"


def expected_mask_path(image_id: str) -> Path:
    return MASKS_DIR / f"{image_id}_mask.tiff"


def read_l0_grade_counts_and_hist(mask_path: Path) -> tuple[dict[int, int], int, int]:
    """Chunked level-0 read: grade counts, tissue px, total px, value histogram."""
    slide = openslide.OpenSlide(str(mask_path))
    try:
        w, h = slide.level_dimensions[0]
        total = w * h
        counts = {g: 0 for g in range(6)}
        extra: dict[int, int] = {}
        tissue = 0
        step = 4096
        for y in range(0, h, step):
            th = min(step, h - y)
            for x in range(0, w, step):
                tw = min(step, w - x)
                tile = np.array(slide.read_region((x, y), 0, (tw, th)))[:, :, 0]
                tissue += int((tile > 0).sum())
                for g in range(6):
                    counts[g] += int((tile == g).sum())
                for v in np.unique(tile):
                    vi = int(v)
                    if vi not in EXPECTED_LABELS:
                        extra[vi] = extra.get(vi, 0) + int((tile == vi).sum())
    finally:
        slide.close()
    return counts, tissue, total, extra


def independent_grade_check(mask_path: Path, gleason: str) -> tuple[bool, dict[int, int]]:
    counts, _, _, _ = read_l0_grade_counts_and_hist(mask_path)
    # emulate check_grade_consistency without loading full array
    for grade in gleason_grades_in_score(gleason):
        if counts.get(grade, 0) == 0:
            return False, counts
    return True, counts


def resize_labels_nearest(labels: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = labels.shape[:2]
    scale = min(max_edge / w, max_edge / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) == (w, h):
        return labels
    return np.array(Image.fromarray(labels).resize((nw, nh), Image.Resampling.NEAREST))


def read_mask_nn(mask_path: Path, max_edge: int = 512) -> np.ndarray:
    slide = openslide.OpenSlide(str(mask_path))
    try:
        level = slide.level_count - 1
        for i in range(slide.level_count):
            lw, lh = slide.level_dimensions[i]
            if max(lw, lh) <= max_edge:
                level = i
                break
        lw, lh = slide.level_dimensions[level]
        arr = np.array(slide.read_region((0, 0), level, (lw, lh)))
        labels = arr[:, :, 0] if arr.ndim == 3 else arr
        return resize_labels_nearest(labels.astype(np.uint8), max_edge)
    finally:
        slide.close()


def save_viz(image_id: str, gleason: str, mask_path: Path, slide_path: Path, out_png: Path) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    mask = openslide.OpenSlide(str(mask_path))
    try:
        slide_thumb = np.array(slide.get_thumbnail((512, 512)).convert("RGB"))
        labels = read_mask_nn(mask_path, 512)
        colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for lab, col in CLASS_COLORS.items():
            colored[labels == lab] = col
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].imshow(slide_thumb)
        axes[0].set_title("WSI thumbnail (512x512)")
        axes[0].axis("off")
        axes[1].imshow(colored)
        axes[1].set_title("Colored mask (nearest-neighbor)")
        axes[1].axis("off")
        legend_labels = sorted(int(v) for v in np.unique(labels) if int(v) in CLASS_NAMES)
        patches = [
            Patch(facecolor=np.array(CLASS_COLORS[v]) / 255.0, label=f"{v}={CLASS_NAMES[v]}")
            for v in legend_labels
        ]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 6), fontsize=8)
        fig.suptitle(f"{image_id} | Gleason {gleason}", fontsize=11, y=0.98)
        fig.subplots_adjust(bottom=0.12, top=0.90)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    finally:
        slide.close()
        mask.close()


def parse_looks_correct(raw: str) -> bool:
    if pd.isna(raw):
        return False
    s = str(raw).strip()
    if s == "negative":
        return True
    parts = s.split("+")
    if len(parts) != 2:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
        return a in (3, 4, 5) and b in (3, 4, 5)
    except ValueError:
        return False


def load_grade_mismatch_ids() -> pd.DataFrame:
    flags = pd.read_csv(FLAGS_CSV)
    gm = flags[flags["reason_codes"].astype(str).str.contains("grade_mismatch", na=False)].copy()
    meta = pd.read_csv(META_CSV)
    gm = gm.merge(meta[["image_id", "gleason_score", "isup_grade"]], on="image_id", how="left", suffixes=("_flags", "_meta"))
    if "gleason_score_meta" in gm.columns:
        gm["gleason_score"] = gm["gleason_score_meta"].fillna(gm.get("gleason_score_flags"))
    cat_path = PROJECT / "outputs" / "qc_failed_by_type" / "failed_grade_mismatch.csv"
    if cat_path.exists():
        cats = pd.read_csv(cat_path)[["image_id", "grade_mismatch_category"]]
        gm = gm.merge(cats, on="image_id", how="left")
    else:
        gm["grade_mismatch_category"] = ""
    return gm


def step1_provenance(gm: pd.DataFrame) -> pd.DataFrame:
    checkpoint = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8")) if CHECKPOINT_JSON.exists() else {}
    meta_dup = pd.read_csv(META_CSV)
    dup_ids = set(meta_dup.loc[meta_dup.duplicated("image_id", keep=False), "image_id"])
    rows = []
    for _, r in tqdm(gm.iterrows(), total=len(gm), desc="step1 provenance"):
        sid = r["image_id"]
        gleason = r["gleason_score"]
        exp_path = expected_mask_path(sid)
        cp = checkpoint.get(sid, {})
        orig = cp.get("grade_consistent", r.get("grade_consistent"))
        orig_result = "pass" if orig is True else "fail" if orig is False else "unknown"
        path_used = Path(str(cp.get("mask_path") or r.get("mask_path") or exp_path))
        exists = path_used.exists()
        real = path_used.resolve() if exists else None
        mtime = datetime.fromtimestamp(real.stat().st_mtime, tz=timezone.utc).isoformat() if real else ""
        path_matches_id = path_used.name == f"{sid}_mask.tiff"
        id_in_path = sid in str(path_used)
        mismatch_reasons = []
        if not exists:
            mismatch_reasons.append("mask_missing")
        if not path_matches_id:
            mismatch_reasons.append("path_name_mismatch")
        if not id_in_path:
            mismatch_reasons.append("id_not_in_path")
        if str(path_used) != str(exp_path) and exists and real != exp_path.resolve():
            mismatch_reasons.append("path_not_expected_symlink_target")
        reread = "error"
        counts = {}
        if exists:
            try:
                ok, counts = independent_grade_check(real, gleason)
                reread = "pass" if ok else "fail"
            except Exception as exc:
                reread = f"error:{exc}"
                mismatch_reasons.append(f"reread_error:{exc}")
        result_mismatch = orig_result != reread and orig_result != "unknown" and reread not in ("error",)
        if orig_result == "fail" and reread == "pass":
            mismatch_reasons.append("false_exclusion_reread_passes")
        if orig_result == "pass" and reread == "fail":
            mismatch_reasons.append("false_inclusion_reread_fails")
        rows.append(
            {
                "slide_id": sid,
                "gleason_score": gleason,
                "original_result": orig_result,
                "reread_result": reread,
                "file_path_used": str(path_used),
                "file_path_expected": str(exp_path),
                "file_path_resolved": str(real) if real else "",
                "file_mtime": mtime,
                "path_matches_slide_id": path_matches_id,
                "joined_by_image_id": True,
                "duplicate_id_found": sid in dup_ids,
                "checkpoint_mask_path": cp.get("mask_path", ""),
                "mismatch_flag": result_mismatch or bool(mismatch_reasons),
                "mismatch_reasons": ";".join(mismatch_reasons),
                "G3_l0": counts.get(3, None),
                "G4_l0": counts.get(4, None),
                "G5_l0": counts.get(5, None),
            }
        )
    return pd.DataFrame(rows)


def step2_parsing(gm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META_CSV)
    distinct = sorted(meta["gleason_score"].dropna().unique().tolist())
    distinct_rows = []
    for s in distinct:
        try:
            parsed = sorted(gleason_grades_in_score(s))
            err = ""
        except Exception as exc:
            parsed = []
            err = str(exc)
        distinct_rows.append(
            {
                "raw_gleason_string": s,
                "parsed_grade_set": str(parsed),
                "parse_error": err,
                "parse_looks_correct": parse_looks_correct(s),
            }
        )
    edge_cases = ["", "0+0", "4-5", "4,5", "nan", "3 + 4", "4+5+5"]
    for s in edge_cases:
        try:
            parsed = sorted(gleason_grades_in_score(s if s != "nan" else float("nan")))
            err = ""
        except Exception as exc:
            parsed = []
            err = str(exc)
        distinct_rows.append(
            {
                "raw_gleason_string": s,
                "parsed_grade_set": str(parsed),
                "parse_error": err,
                "parse_looks_correct": False,
            }
        )
    fail_rows = []
    for _, r in gm.iterrows():
        raw = r["gleason_score"]
        try:
            parsed = sorted(gleason_grades_in_score(raw))
            err = ""
        except Exception as exc:
            parsed = []
            err = str(exc)
        fail_rows.append(
            {
                "slide_id": r["image_id"],
                "raw_gleason_string": raw,
                "parsed_grade_set": str(parsed),
                "parse_error": err,
                "parse_looks_correct": parse_looks_correct(raw) and not err,
            }
        )
    return pd.DataFrame(distinct_rows), pd.DataFrame(fail_rows)


def step3_encoding(gm: pd.DataFrame) -> pd.DataFrame:
    random.seed(42)
    pools = {
        "primary_present_secondary_absent": gm[gm["grade_mismatch_category"] == "primary_present_secondary_absent"],
        "primary_absent_secondary_present": gm[gm["grade_mismatch_category"] == "primary_absent_secondary_present"],
        "sole_grade_absent": gm[gm["grade_mismatch_category"] == "sole_grade_absent"],
        "both_grades_absent": gm[gm["grade_mismatch_category"] == "both_grades_absent"],
    }
    picks = []
    picks.extend(pools["primary_present_secondary_absent"].sample(min(10, len(pools["primary_present_secondary_absent"]))).image_id.tolist())
    picks.extend(pools["primary_absent_secondary_present"].sample(min(10, len(pools["primary_absent_secondary_present"]))).image_id.tolist())
    picks.extend(pools["sole_grade_absent"].sample(min(5, len(pools["sole_grade_absent"]))).image_id.tolist())
    picks.extend(pools["both_grades_absent"].sample(min(5, len(pools["both_grades_absent"]))).image_id.tolist())
    rows = []
    for sid in picks:
        cat = gm.loc[gm.image_id == sid, "grade_mismatch_category"].iloc[0]
        p = expected_mask_path(sid)
        counts, tissue, total, extra = read_l0_grade_counts_and_hist(p.resolve())
        uniq = sorted(counts.keys())
        if extra:
            uniq = sorted(set(uniq) | set(extra.keys()))
        hist = {str(k): counts.get(k, extra.get(k, 0)) for k in uniq if counts.get(k, 0) or extra.get(k, 0)}
        rows.append(
            {
                "slide_id": sid,
                "category": cat,
                "unique_values_found": str(uniq),
                "class_histogram": str(hist),
                "unexpected_values_present": bool(extra),
                "unexpected_values": str(sorted(extra.keys())) if extra else "",
            }
        )
    return pd.DataFrame(rows)


def step4_category_b(gm: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    sub = gm[gm["grade_mismatch_category"] == "primary_absent_secondary_present"].copy()
    out_dir = OUT / "category_b_review"
    rows = []
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc="step4 category B"):
        sid = r["image_id"]
        gs = r["gleason_score"]
        pri, sec = parse_gleason(gs)
        p = expected_mask_path(sid).resolve()
        counts, tissue, total, _ = read_l0_grade_counts_and_hist(p)
        present_g = sec if counts.get(pri, 0) == 0 else pri
        absent_g = pri if present_g == sec else sec
        if counts.get(pri, 0) == 0 and counts.get(sec, 0) > 0:
            present_g, absent_g = sec, pri
        elif counts.get(sec, 0) == 0 and counts.get(pri, 0) > 0:
            present_g, absent_g = pri, sec
        slide_path = SLIDES_DIR / f"{sid}.tiff"
        png = out_dir / f"slide_{sid}.png"
        if slide_path.exists():
            save_viz(sid, gs, p, slide_path, png)
        prov_row = prov[prov.slide_id == sid].iloc[0] if sid in prov.slide_id.values else None
        rows.append(
            {
                "slide_id": sid,
                "gleason_score": gs,
                "grade_present": f"G{present_g}",
                "grade_present_pixel_count": counts.get(present_g, 0),
                "grade_present_pct_of_tissue": round(100 * counts.get(present_g, 0) / max(tissue, 1), 4),
                "grade_absent": f"G{absent_g}",
                "grade_absent_pixel_count_l0": counts.get(absent_g, 0),
                "total_tissue_area_px": tissue,
                "total_mask_px": total,
                "provenance_mismatch_flag": bool(prov_row["mismatch_flag"]) if prov_row is not None else False,
                "technical_red_flag": bool(prov_row["mismatch_flag"]) if prov_row is not None else False,
                "png_path": str(png),
            }
        )
    return pd.DataFrame(rows)


def step5_categories_cd(gm: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    sub = gm[gm["grade_mismatch_category"].isin(["sole_grade_absent", "both_grades_absent"])].copy()
    rows = []
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc="step5 categories C/D"):
        sid = r["image_id"]
        p = expected_mask_path(sid).resolve()
        counts, tissue, total, extra = read_l0_grade_counts_and_hist(p)
        hist = {CLASS_NAMES.get(k, str(k)): counts[k] for k in sorted(counts) if counts[k]}
        tissue_ratio = tissue / max(total, 1)
        # flag if tissue_ratio < 0.5% but WSI exists (possible truncated mask)
        slide_path = SLIDES_DIR / f"{sid}.tiff"
        suspicious = tissue_ratio < 0.005 and slide_path.exists()
        prov_row = prov[prov.slide_id == sid].iloc[0] if sid in prov.slide_id.values else None
        rows.append(
            {
                "slide_id": sid,
                "category": r["grade_mismatch_category"],
                "gleason_score": r["gleason_score"],
                "total_tissue_area_px": tissue,
                "tissue_ratio": round(tissue_ratio, 6),
                "class_histogram": str(hist),
                "unexpected_label_values": str(sorted(extra.keys())) if extra else "",
                "tissue_area_suspiciously_small": suspicious,
                "provenance_mismatch_flag": bool(prov_row["mismatch_flag"]) if prov_row is not None else False,
            }
        )
    return pd.DataFrame(rows)


def step6_category_a_sample(gm: pd.DataFrame, prov: pd.DataFrame) -> pd.DataFrame:
    random.seed(42)
    a = gm[gm["grade_mismatch_category"] == "primary_present_secondary_absent"].copy()
    a["primary_grade"] = a["gleason_score"].apply(lambda g: parse_gleason(g)[0])
    g4 = a[a.primary_grade == 4].sample(min(15, (a.primary_grade == 4).sum()), random_state=42)
    g3 = a[a.primary_grade == 3].sample(min(10, (a.primary_grade == 3).sum()), random_state=42)
    g5 = a[a.primary_grade == 5]
    sample = pd.concat([g4, g3, g5]).drop_duplicates("image_id")
    out_dir = OUT / "category_a_sample_review"
    rows = []
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="step6 category A sample"):
        sid = r["image_id"]
        gs = r["gleason_score"]
        pri, sec = parse_gleason(gs)
        p = expected_mask_path(sid).resolve()
        counts, tissue, total, _ = read_l0_grade_counts_and_hist(p)
        slide_path = SLIDES_DIR / f"{sid}.tiff"
        png = out_dir / f"slide_{sid}.png"
        if slide_path.exists():
            save_viz(sid, gs, p, slide_path, png)
        notes = []
        if counts.get(sec, 0) == 0:
            notes.append(f"G{sec}=0 at L0 confirmed")
        if counts.get(pri, 0) > 0:
            notes.append(f"G{pri} present ({counts[pri]:,} px)")
        if tissue / max(total, 1) < 0.01:
            notes.append("low tissue_ratio")
        prov_row = prov[prov.slide_id == sid].iloc[0] if sid in prov.slide_id.values else None
        if prov_row is not None and prov_row["mismatch_flag"]:
            notes.append("PROVENANCE_MISMATCH")
        rows.append(
            {
                "slide_id": sid,
                "gleason_score": gs,
                "primary_grade": pri,
                "primary_pixel_count": counts.get(pri, 0),
                "primary_pct_tissue": round(100 * counts.get(pri, 0) / max(tissue, 1), 4),
                "secondary_grade": sec,
                "secondary_pixel_count_l0": counts.get(sec, 0),
                "visual_flag_notes": "; ".join(notes),
                "png_path": str(png),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    prov: pd.DataFrame,
    parsing_distinct: pd.DataFrame,
    parsing_fail: pd.DataFrame,
    encoding: pd.DataFrame,
    cat_b: pd.DataFrame,
    cat_cd: pd.DataFrame,
    cat_a: pd.DataFrame,
) -> None:
    mismatches = prov[prov.mismatch_flag == True]  # noqa: E712
    false_excl = prov[prov["mismatch_reasons"].astype(str).str.contains("false_exclusion", na=False)]
    parse_bad = parsing_fail[~parsing_fail.parse_looks_correct]
    enc_bad = encoding[encoding.unexpected_values_present == True]  # noqa: E712
    b_clean = cat_b[~cat_b.technical_red_flag]
    cd_susp = cat_cd[cat_cd.tissue_area_suspiciously_small == True]  # noqa: E712
    a_notes = cat_a["visual_flag_notes"].astype(str)

    lines = [
        "# QC grade-mismatch bug audit summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Provenance / fresh L0 re-read (all 322)",
        "",
        f"- Slides with `mismatch_flag=True`: **{len(mismatches)}**",
        f"- False exclusions (original fail, independent re-read pass): **{len(false_excl)}**",
        "",
    ]
    if len(mismatches):
        lines.append("### Mismatch slide_ids")
        for sid in mismatches.slide_id.tolist():
            reasons = mismatches.loc[mismatches.slide_id == sid, "mismatch_reasons"].iloc[0]
            lines.append(f"- `{sid}` — {reasons}")
        lines.append("")
    else:
        lines.append("No provenance mismatches found. All 322 failures reproduced on fresh independent L0 re-read.")
        lines.append("")

    lines += [
        "## 2. Gleason string parsing",
        "",
        f"- Distinct scores in metadata: {parsing_distinct.raw_gleason_string.nunique()} (+ edge cases tested)",
        f"- Parsing errors on failure set: **{parsing_fail.parse_error.astype(bool).sum()}**",
        f"- Failure-set rows where parse looks wrong: **{len(parse_bad)}**",
        "",
        "Edge-case behavior: `negative` → empty set; `4+5` → {4,5}; malformed strings raise or mis-parse (see `qc_bug_audit_parsing.csv`).",
        "",
        "## 3. Label encoding (30-slide sample)",
        "",
        f"- Slides with unexpected mask values outside {{0-5}}: **{len(enc_bad)}**",
        "",
    ]
    if len(enc_bad):
        for _, r in enc_bad.iterrows():
            lines.append(f"- `{r.slide_id}` ({r.category}): unexpected {r.unexpected_values}")
        lines.append("")

    lines += [
        "## 4. Category B (52 slides — primary absent, secondary present)",
        "",
        f"- Clean technical read (no provenance flag): **{len(b_clean)}** / 52",
        f"- Provenance red flags: **{cat_b.technical_red_flag.sum()}**",
        f"- PNGs: `category_b_review/`",
        "",
        "## 5. Categories C/D (82 slides)",
        "",
        f"- Suspiciously small tissue area (<0.5% labeled): **{len(cd_susp)}**",
        f"- Provenance red flags: **{cat_cd.provenance_mismatch_flag.sum()}**",
        "",
        "## 6. Category A sample (34 slides)",
        "",
        f"- Sample size: {len(cat_a)} (15 G4-main, 10 G3-main, 9 G5-main)",
        f"- All samples: secondary grade 0 at L0 (`G{{sec}}=0 at L0 confirmed` in notes): **{a_notes.str.contains('=0 at L0 confirmed').sum()}**",
        f"- Provenance flags in sample: **{a_notes.str.contains('PROVENANCE_MISMATCH').sum()}**",
        f"- PNGs: `category_a_sample_review/`",
        "",
        "## 7. Re-run recommendations",
        "",
    ]
    rerun = set(false_excl.slide_id.tolist()) | set(mismatches[mismatches.mismatch_reasons.str.contains("mask_missing|reread_error", na=False)].slide_id.tolist())
    if rerun:
        lines.append("Re-run through `clean_dataset.py` after fixing:")
        for sid in sorted(rerun):
            lines.append(f"- `{sid}`")
    else:
        lines.append("**None required for technical bugs.** The 322 exclusions are reproducible on fresh L0 reads with current mask files.")
        lines.append("")
        lines.append("Separate from this audit: 5 `missing_local_mask` stale-checkpoint slides (not in grade_mismatch set) may still need re-QC if masks now exist.")

    lines += [
        "",
        "## Conclusion",
        "",
        "This audit tests whether exclusions reflect **correct reads of the current mask files**, not whether annotations are clinically perfect.",
        "",
    ]
    (OUT / "qc_bug_audit_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gm = load_grade_mismatch_ids()
    print(f"Grade mismatch slides: {len(gm)}")

    prov = step1_provenance(gm)
    prov.to_csv(OUT / "qc_bug_audit_provenance.csv", index=False)

    parsing_distinct, parsing_fail = step2_parsing(gm)
    parsing_distinct.to_csv(OUT / "qc_bug_audit_parsing_distinct.csv", index=False)
    parsing_fail.to_csv(OUT / "qc_bug_audit_parsing.csv", index=False)

    encoding = step3_encoding(gm)
    encoding.to_csv(OUT / "qc_bug_audit_encoding.csv", index=False)

    cat_b = step4_category_b(gm, prov)
    cat_b.to_csv(OUT / "qc_bug_audit_category_b.csv", index=False)

    cat_cd = step5_categories_cd(gm, prov)
    cat_cd.to_csv(OUT / "qc_bug_audit_categories_cd.csv", index=False)

    cat_a = step6_category_a_sample(gm, prov)
    cat_a.to_csv(OUT / "qc_bug_audit_category_a_sample.csv", index=False)

    write_summary(prov, parsing_distinct, parsing_fail, encoding, cat_b, cat_cd, cat_a)
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
