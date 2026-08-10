#!/usr/bin/env python3
"""Apply the user's 2026-08-09 review of old multi-clusters C1/C2/C3.

The C1-35 multi review resolved clusters as a whole ("keep_safe" x207), so twin
pairs sitting *inside* a big cluster were never split apart. These are those
pairs, called by eye off the round-1 gallery pages.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dedupe_slides_shape_isup import META_PATH, SPLITS_DIR, apply_drops, choose_keep_drop, load_patch_counts
from patch_utils import PROJECT

DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"

TWIN_PAIRS = [
    ("multi_C1", "f344628cd3bfd6e1cd80456bca9358ab", "fe6984b9e99ee331bb510c61f9e5ccd4"),
    ("multi_C2", "478261713d5379009e2f1be1ad3deb63", "500b90c3e7eba2e3b25818a8c1834d45"),
    ("multi_C2", "b13961504ea859ff34a150bc19fed335", "b95035c00e6e99f3b4e3cd210700f6ce"),
    ("multi_C2", "d2e8ff5257ef522c1d7bef30c96d06df", "a3794ec31a02fbf429486dd464f83d25"),
    ("multi_C2", "757b927de93c8083c392f2248529d29a", "82c289e2fb53713789599f77cf368bc6"),
    ("multi_C3", "49764620319a820dfd2e6738861b1aa0", "fe025faa4c6e6de2a71f964b56de7c9c"),
    ("multi_C3", "9e0fa510b3e5a89ab765dfa48957a4cc", "a28b628b33608b2cdadff12fc6aa2537"),
]

# scan3 safe pairs the user looked at and cleared
NOT_TWIN_PAIRS = [
    ("scan3_safe_P1", "49a2ed9b6af2a48a788b9dd6066bdb0a", "9a4ba048c62623465d6a7a0d8549a657"),
    ("scan3_safe_P2", "70f71c6fa7ad2baeaff157a299e276a0", "63226114e772943ce7f8017d9f5d4b90"),
]


def main() -> None:
    alive = set(pd.read_csv(SPLITS_DIR / "panda_slide_splits.csv", dtype=str).image_id)
    meta = pd.read_csv(META_PATH)
    meta["image_id"] = meta["image_id"].astype(str)
    counts = load_patch_counts()

    rows = []
    for cluster, a, b in TWIN_PAIRS:
        if a not in alive or b not in alive:
            print(f"skip {a[:12]}/{b[:12]} — already resolved")
            continue
        keep, drop = choose_keep_drop([a, b], counts, meta)
        rows.append(
            {
                "cluster": cluster,
                "keep_id": keep,
                "drop_id": drop[0],
                "keep_n_patches": counts.get(keep, 0),
                "drop_n_patches": counts.get(drop[0], 0),
                "action": "drop_twin",
                "reason": "user 2026-08-09: twins inside round-1 multi cluster",
            }
        )
    dec = pd.DataFrame(rows)
    dec.to_csv(DUP / "user_multi_C1_C2_C3_inner_twins_decisions.csv", index=False)
    print(dec.to_string(index=False))

    pd.DataFrame(
        [
            {
                "source": src,
                "keep_id": a,
                "drop_id": b,
                "action": "keep_not_twin",
                "reason": "user 2026-08-09: scan3 safe pair reviewed, not twins",
            }
            for src, a, b in NOT_TWIN_PAIRS
        ]
    ).to_csv(DUP / "user_marked_not_same_scan3_safe.csv", index=False)

    report = apply_drops(set(dec.drop_id), dry_run=False)
    after = pd.read_csv(SPLITS_DIR / "panda_slide_splits.csv", dtype=str)
    report["n_dropped_now"] = len(dec)
    report["split_counts"] = after.split.value_counts().to_dict()
    (DUP / "user_multi_C1_C2_C3_inner_twins_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
