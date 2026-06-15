"""Cross-reference PANDA-PLUS slide IDs with radboud_clean.csv and remove overlaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
BENCH_REPO = PROJECT / "panda-plus-bench"
IDS_FILE = DATA / "panda_plus_ids.txt"
BENCH_IDS_FILE = DATA / "panda_plus_bench_slide_ids.txt"
RADBOUD_CLEAN = DATA / "radboud_clean.csv"
INVENTORY_FILE = DATA / "panda_plus_repo_inventory.txt"


def inventory_bench_repo() -> list[str]:
    if not BENCH_REPO.exists():
        return []
    skip = {".git"}
    paths: list[str] = []
    for path in sorted(BENCH_REPO.rglob("*")):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            paths.append(str(path.relative_to(BENCH_REPO)))
    return paths


def load_panda_plus_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def extract_bench_slide_ids() -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []
    parquet = hf_hub_download(
        repo_id="dellacorte/PANDA-PLUS-Bench",
        repo_type="dataset",
        filename="data/baseline-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet, columns=["slide_id"])
    return sorted(df["slide_id"].astype(str).unique().tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", type=Path, default=IDS_FILE)
    args = parser.parse_args()

    repo_files = inventory_bench_repo()
    INVENTORY_FILE.write_text("\n".join(repo_files) + ("\n" if repo_files else ""), encoding="utf-8")

    print("=" * 70)
    print("PANDA-PLUS-Bench repository files")
    print("=" * 70)
    for rel in repo_files:
        print(f"  {rel}")
    print(f"\nTotal files (excluding .git): {len(repo_files)}")

    csv_json = [p for p in repo_files if p.lower().endswith((".csv", ".json", ".txt", ".tsv"))]
    print("\nCSV / JSON / text files in repo:")
    if csv_json:
        for rel in csv_json:
            print(f"  {rel}")
    else:
        print("  (none with slide ID lists)")

    bench_ids = extract_bench_slide_ids()
    if bench_ids:
        BENCH_IDS_FILE.write_text("\n".join(bench_ids) + "\n", encoding="utf-8")
        print(f"\nSaved 9 benchmark internal slide IDs -> {BENCH_IDS_FILE}")
        print("  IDs:", ", ".join(bench_ids))

    panda_plus_ids = load_panda_plus_ids(args.ids_file)
    print(f"\nPANDA-PLUS ID list ({args.ids_file}): {len(panda_plus_ids)} IDs")

    clean = pd.read_csv(RADBOUD_CLEAN)
    before = len(clean)
    overlap = sorted(set(clean["image_id"]) & set(panda_plus_ids))
    print(f"\nRadboud clean (before PANDA-PLUS exclusion): {before}")
    print(f"Overlap with PANDA-PLUS list:                  {len(overlap)}")

    if panda_plus_ids:
        updated = clean[~clean["image_id"].isin(panda_plus_ids)].copy()
        updated.to_csv(RADBOUD_CLEAN, index=False)
        print(f"Radboud clean (after PANDA-PLUS exclusion):    {len(updated)}")
        print(f"Saved -> {RADBOUD_CLEAN}")
    else:
        print("\nNo PANDA-PLUS Kaggle image IDs loaded yet.")
        print("The 546-slide list is not in panda-plus-bench GitHub.")
        print("Request masks from the corresponding author (J Pathol Inform 2025).")

    print(f"\nRepo inventory saved -> {INVENTORY_FILE}")


if __name__ == "__main__":
    main()
