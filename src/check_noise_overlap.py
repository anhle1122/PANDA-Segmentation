"""Post-hoc cross-reference: our high-loss slides vs 6th-place noise_ratio_10 flags.

DIAGNOSTIC ONLY — this script cannot and does not change training data.

The 6th-place list is another model's opinion, not ground truth. Use this after
training on a sufficiently converged checkpoint (not epoch 1-2) to report
agreement statistics and flag slides for manual review.

Do NOT auto-drop slides based on overlap. Agreement is informative; disagreement
is equally worth logging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"

SIXTH_PLACE_CSV = DATA / "sixth_place_train.csv"
SIXTH_PLACE_FLAGGED_TXT = DATA / "sixth_place_noise_flagged_ids.txt"
LEGACY_EXCLUDED_TXT = DATA / "excluded_slide_ids.txt"
TRAIN_CSV = DATA / "train.csv"
HIGH_LOSS_CSV = OUTPUTS / "high_loss_slides.csv"
SUMMARY_CSV = OUTPUTS / "noise_overlap_summary.csv"
SUMMARY_BY_PROVIDER_CSV = OUTPUTS / "noise_overlap_by_provider.csv"
SLIDES_CSV = OUTPUTS / "noise_overlap_slides.csv"
MANUAL_REVIEW_CSV = OUTPUTS / "manual_review_candidates.csv"
DISAGREEMENT_CSV = OUTPUTS / "noise_overlap_disagreements.csv"

LOSS_COLUMNS = ("mean_loss", "loss", "slide_loss", "val_loss", "loss_rank")


def load_metadata() -> pd.DataFrame:
    if SIXTH_PLACE_CSV.exists():
        return pd.read_csv(SIXTH_PLACE_CSV)
    if TRAIN_CSV.exists():
        return pd.read_csv(TRAIN_CSV)
    raise FileNotFoundError(
        f"Missing metadata. Run src/build_clean_radboud.py to fetch {SIXTH_PLACE_CSV.name}."
    )


def load_sixth_place_flagged(metadata: pd.DataFrame, scope: set[str] | None) -> set[str]:
    if "noise_ratio_10" in metadata.columns:
        flagged = set(metadata.loc[metadata["noise_ratio_10"] == 0, "image_id"].astype(str))
    elif SIXTH_PLACE_FLAGGED_TXT.exists():
        flagged = {
            line.strip()
            for line in SIXTH_PLACE_FLAGGED_TXT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif LEGACY_EXCLUDED_TXT.exists():
        flagged = {
            line.strip()
            for line in LEGACY_EXCLUDED_TXT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        raise FileNotFoundError("No 6th-place reference list found.")
    if scope is not None:
        flagged &= scope
    return flagged


def pick_loss_column(df: pd.DataFrame) -> str:
    for col in LOSS_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"High-loss CSV must include one of {LOSS_COLUMNS}; got {list(df.columns)}"
    )


def resolve_high_loss_csv(epoch: int | None, csv_path: Path) -> Path:
    if epoch is not None:
        epoch_csv = OUTPUTS / "loss_tracking" / f"epoch_{epoch:03d}_per_slide.csv"
        if not epoch_csv.exists():
            raise FileNotFoundError(
                f"Missing {epoch_csv}. Train with --loss-hist-every 1, or run "
                "src/suggest_noise_epoch.py to find a valid epoch."
            )
        return epoch_csv
    return csv_path


def load_our_high_loss(
    csv_path: Path,
    *,
    top_pct: float,
    top_n: int | None,
    scope: set[str] | None,
) -> tuple[set[str], str, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. After training run:\n"
            "  python src/suggest_noise_epoch.py --export"
        )
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        raise ValueError(f"{csv_path} must have an image_id column")

    loss_col = pick_loss_column(df)
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    if scope is not None:
        df = df[df["image_id"].isin(scope)]

    ascending = loss_col == "loss_rank"
    df = df.sort_values(loss_col, ascending=ascending)
    if top_n is not None:
        selected = df.head(top_n)
    else:
        n = max(1, int(round(len(df) * top_pct / 100.0)))
        selected = df.head(n)
    return set(selected["image_id"]), loss_col, selected


def compute_overlap(reference: set[str], candidates: set[str]) -> dict:
    both = reference & candidates
    ref_only = reference - candidates
    cand_only = candidates - reference
    union = reference | candidates

    def pct(num: int, den: int) -> float | None:
        return round(100.0 * num / den, 2) if den else None

    def safe_div(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "sixth_place_flagged": len(reference),
        "our_high_loss_flagged": len(candidates),
        "overlap_count": len(both),
        "sixth_place_only_count": len(ref_only),
        "our_high_loss_only_count": len(cand_only),
        "union_count": len(union),
        "pct_of_sixth_place": pct(len(both), len(reference)),
        "pct_of_our_high_loss": pct(len(both), len(candidates)),
        "jaccard": safe_div(len(both), len(union)),
    }


def stratify_by_provider(
    reference: set[str],
    candidates: set[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if "data_provider" not in metadata.columns:
        return pd.DataFrame()

    rows = []
    meta = metadata.copy()
    meta["image_id"] = meta["image_id"].astype(str)
    for provider in sorted(meta["data_provider"].dropna().unique()):
        provider_ids = set(meta.loc[meta["data_provider"] == provider, "image_id"])
        ref = reference & provider_ids
        ours = candidates & provider_ids
        metrics = compute_overlap(ref, ours)
        metrics["data_provider"] = provider
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_slide_table(
    reference: set[str],
    candidates: set[str],
    metadata: pd.DataFrame,
    candidate_detail: pd.DataFrame | None,
) -> pd.DataFrame:
    all_ids = sorted(reference | candidates)
    meta = metadata.copy()
    meta["image_id"] = meta["image_id"].astype(str)
    meta = meta.drop_duplicates("image_id")

    rows = []
    for image_id in all_ids:
        in_ref = image_id in reference
        in_cand = image_id in candidates
        if in_ref and in_cand:
            bucket = "both_agree"
        elif in_ref:
            bucket = "sixth_place_only"
        elif in_cand:
            bucket = "our_high_loss_only"
        else:
            bucket = "neither"
        rows.append(
            {
                "image_id": image_id,
                "sixth_place_noise_flag": in_ref,
                "our_high_loss_flag": in_cand,
                "overlap_bucket": bucket,
                "manual_review_recommended": in_ref and in_cand,
            }
        )

    out = pd.DataFrame(rows).merge(meta, on="image_id", how="left")
    if candidate_detail is not None and "image_id" in candidate_detail.columns:
        detail = candidate_detail.copy()
        detail["image_id"] = detail["image_id"].astype(str)
        extra_cols = [c for c in detail.columns if c != "image_id"]
        out = out.merge(detail[extra_cols + ["image_id"]], on="image_id", how="left")
    return out


def print_summary(metrics: dict, by_provider: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("NOISE FLAG CROSS-REFERENCE (diagnostic only — no auto-drops)")
    print("=" * 60)
    print(f"6th-place noise_ratio_10 == 0:  {metrics['sixth_place_flagged']}")
    print(f"Our high-loss tail:              {metrics['our_high_loss_flagged']}")
    print(f"Overlap (both flag):             {metrics['overlap_count']}")
    print(f"  % of 6th-place flags:          {metrics['pct_of_sixth_place']}%")
    print(f"  % of our high-loss tail:       {metrics['pct_of_our_high_loss']}%")
    print(f"6th-place only (disagreement):   {metrics['sixth_place_only_count']}")
    print(f"Our high-loss only (disagreement): {metrics['our_high_loss_only_count']}")
    print(f"Jaccard:                         {metrics['jaccard']}")
    print("=" * 60)

    if len(by_provider):
        print("\nStratified by data_provider:")
        for _, row in by_provider.iterrows():
            print(
                f"  {row['data_provider']}: overlap {int(row['overlap_count'])}/"
                f"{int(row['sixth_place_flagged'])} sixth-place & "
                f"{int(row['our_high_loss_flagged'])} ours "
                f"({row['pct_of_sixth_place']}% of their flags, "
                f"{row['pct_of_our_high_loss']}% of ours)"
            )
    print()
    print("Slides in overlap_bucket=both_agree -> manual_review_candidates.csv")
    print("Disagreement cases are logged separately (not discarded).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc overlap: our high-loss slides vs 6th-place noise flags"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Use per-slide losses from outputs/loss_tracking/epoch_NNN_per_slide.csv",
    )
    parser.add_argument(
        "--high-loss-csv",
        type=Path,
        default=HIGH_LOSS_CSV,
        help="Per-slide loss CSV (default: outputs/high_loss_slides.csv after --export)",
    )
    parser.add_argument("--top-pct", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument(
        "--scope-ids",
        type=Path,
        default=None,
        help="Optional CSV/txt of image_ids to restrict comparison scope",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    scope = None
    if args.scope_ids:
        if args.scope_ids.suffix == ".csv":
            scope = set(pd.read_csv(args.scope_ids)["image_id"].astype(str))
        else:
            scope = {
                line.strip()
                for line in args.scope_ids.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    sixth_place = load_sixth_place_flagged(metadata, scope)
    loss_csv = resolve_high_loss_csv(args.epoch, args.high_loss_csv)
    ours, loss_col, candidate_detail = load_our_high_loss(
        loss_csv,
        top_pct=args.top_pct,
        top_n=args.top_n,
        scope=scope,
    )

    metrics = compute_overlap(sixth_place, ours)
    metrics.update(
        {
            "candidate_source": str(loss_csv),
            "loss_column": loss_col,
            "epoch": args.epoch,
            "top_n": args.top_n,
            "top_pct": None if args.top_n else args.top_pct,
        }
    )
    by_provider = stratify_by_provider(sixth_place, ours, metadata)

    slide_table = build_slide_table(sixth_place, ours, metadata, candidate_detail)
    manual_review = slide_table[slide_table["manual_review_recommended"]].copy()
    disagreements = slide_table[
        slide_table["overlap_bucket"].isin(["sixth_place_only", "our_high_loss_only"])
    ].copy()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(SUMMARY_CSV, index=False)
    if len(by_provider):
        by_provider.to_csv(SUMMARY_BY_PROVIDER_CSV, index=False)
    slide_table.to_csv(SLIDES_CSV, index=False)
    manual_review.to_csv(MANUAL_REVIEW_CSV, index=False)
    disagreements.to_csv(DISAGREEMENT_CSV, index=False)

    print_summary(metrics, by_provider)
    print(f"Saved summary           -> {SUMMARY_CSV}")
    if len(by_provider):
        print(f"Saved by-provider table -> {SUMMARY_BY_PROVIDER_CSV}")
    print(f"Saved per-slide table   -> {SLIDES_CSV}")
    print(f"Saved manual review     -> {MANUAL_REVIEW_CSV} ({len(manual_review)} slides)")
    print(f"Saved disagreements     -> {DISAGREEMENT_CSV} ({len(disagreements)} slides)")


if __name__ == "__main__":
    main()
