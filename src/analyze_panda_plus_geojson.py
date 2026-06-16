"""Analyze PANDA-PLUS GeoJSON exports: class counts, polygon areas, benchmark overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_GEOJSON_DIR = PROJECT / "panda_plus_geojson_export-2" / "panda_plus_geojson_export"
OUTPUTS = PROJECT / "outputs"

LABELS = ("Benign", "GP3", "GP4", "GP5")
BENCHMARK_SLIDE_IDS = {"27", "101", "391", "416", "2571", "3010", "3333", "3501", "3524"}


def ring_area(ring: list) -> float:
    """Signed polygon area via shoelace formula (level-0 pixel coordinates)."""
    if len(ring) < 3:
        return 0.0
    xs = np.asarray([pt[0] for pt in ring], dtype=np.float64)
    ys = np.asarray([pt[1] for pt in ring], dtype=np.float64)
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def polygon_area(coords: list) -> float:
    """Area of a GeoJSON Polygon (exterior minus holes)."""
    if not coords:
        return 0.0
    area = abs(ring_area(coords[0]))
    for hole in coords[1:]:
        area -= abs(ring_area(hole))
    return max(area, 0.0)


def geometry_area(geometry: dict) -> float:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return 0.0
    if gtype == "Polygon":
        return polygon_area(coords)
    if gtype == "MultiPolygon":
        return sum(polygon_area(poly) for poly in coords)
    return 0.0


def analyze_geojson(path: Path) -> dict:
    slide_id = path.stem
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    counts = {label: 0 for label in LABELS}
    areas = {label: 0.0 for label in LABELS}
    other_labels: dict[str, int] = {}

    for feature in data.get("features", []):
        label = feature.get("properties", {}).get("label", "UNKNOWN")
        area = geometry_area(feature.get("geometry", {}))

        if label in counts:
            counts[label] += 1
            areas[label] += area
        else:
            other_labels[label] = other_labels.get(label, 0) + 1

    row = {
        "slide_id": slide_id,
        "n_features": sum(counts.values()) + sum(other_labels.values()),
        "is_benchmark_slide": slide_id in BENCHMARK_SLIDE_IDS,
    }
    for label in LABELS:
        row[f"count_{label}"] = counts[label]
        row[f"area_{label}"] = areas[label]
    row["total_area"] = sum(areas.values())
    if other_labels:
        row["other_labels"] = "; ".join(f"{k}:{v}" for k, v in sorted(other_labels.items()))
    return row


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = df["slide_id"].astype(str).tolist()
    x = np.arange(len(slides))
    width = 0.2
    colors = {"Benign": "#4daf4a", "GP3": "#377eb8", "GP4": "#ff7f00", "GP5": "#e41a1c"}

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for i, label in enumerate(LABELS):
        axes[0].bar(x + (i - 1.5) * width, df[f"count_{label}"], width=width, label=label, color=colors[label])
    axes[0].set_ylabel("Feature count")
    axes[0].set_title("PANDA-PLUS GeoJSON: class counts per slide (n=50)")
    axes[0].legend(loc="upper right", ncol=4)
    axes[0].grid(axis="y", alpha=0.3)

    for i, label in enumerate(LABELS):
        axes[1].bar(x + (i - 1.5) * width, df[f"area_{label}"] / 1e6, width=width, label=label, color=colors[label])
    axes[1].set_ylabel("Area (M px^2)")
    axes[1].set_xlabel("Slide ID (PANDA+ internal index)")
    axes[1].set_title("PANDA-PLUS GeoJSON: class area per slide (shoelace, level-0 pixels)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(slides, rotation=90, fontsize=7)

    bench_mask = df["is_benchmark_slide"].to_numpy()
    for ax in axes:
        for idx in np.where(bench_mask)[0]:
            ax.axvspan(idx - 0.5, idx + 0.5, color="gold", alpha=0.15, zorder=0)

    fig.tight_layout()
    fig.savefig(out_dir / "panda_plus_class_distribution.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    total_counts = [df[f"count_{label}"].sum() for label in LABELS]
    total_areas = [df[f"area_{label}"].sum() / 1e6 for label in LABELS]
    x2 = np.arange(len(LABELS))
    ax2.bar(x2 - 0.2, total_counts, width=0.35, label="Count", color="#6baed6")
    ax2_t = ax2.twinx()
    ax2_t.bar(x2 + 0.2, total_areas, width=0.35, label="Area (M px^2)", color="#fd8d3c", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Total feature count")
    ax2_t.set_ylabel("Total area (M px^2)")
    ax2.set_title("PANDA-PLUS GeoJSON: aggregate class distribution (50 slides)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "panda_plus_class_distribution_aggregate.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PANDA-PLUS GeoJSON exports")
    parser.add_argument("--geojson-dir", type=Path, default=DEFAULT_GEOJSON_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUTS / "panda_plus_analysis.csv")
    args = parser.parse_args()

    geojson_files = sorted(args.geojson_dir.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in {args.geojson_dir}")

    rows = [analyze_geojson(path) for path in geojson_files]
    df = pd.DataFrame(rows).sort_values("slide_id", key=lambda s: s.astype(int))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    plot_distributions(df, args.output_csv.parent)

    print(f"Analyzed {len(df)} GeoJSON files from {args.geojson_dir}")
    print(f"Saved summary -> {args.output_csv}")
    print(f"Saved plots   -> {args.output_csv.parent / 'panda_plus_class_distribution.png'}")

    print("\nAggregate totals (50 slides):")
    for label in LABELS:
        print(
            f"  {label:6s}: {int(df[f'count_{label}'].sum()):5d} regions, "
            f"{df[f'area_{label}'].sum() / 1e6:,.1f} M px^2"
        )

    gp5_slides = df[df["count_GP5"] > 0].sort_values("count_GP5", ascending=False)
    print(f"\nSlides with GP5: {len(gp5_slides)} / {len(df)}")
    for _, row in gp5_slides.iterrows():
        bench = " [BENCHMARK]" if row["is_benchmark_slide"] else ""
        print(f"  slide {row['slide_id']:>4s}: {int(row['count_GP5']):3d} GP5 regions{bench}")

    bench_in_export = sorted(BENCHMARK_SLIDE_IDS & {path.stem for path in geojson_files}, key=int)
    bench_missing = sorted(BENCHMARK_SLIDE_IDS - {path.stem for path in geojson_files}, key=int)
    print(f"\nBenchmark IDs in export ({len(bench_in_export)}): {', '.join(bench_in_export) or '(none)'}")
    print(f"Benchmark IDs NOT in export ({len(bench_missing)}): {', '.join(bench_missing) or '(none)'}")


if __name__ == "__main__":
    main()
