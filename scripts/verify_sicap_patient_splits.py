#!/usr/bin/env python3
"""Verify SICAPv2 patient_id separation across official Test and Val folds.

Exits 0 if all required overlaps are empty; 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")


def slide_id_from_patch(name: str) -> str:
    return Path(name).stem.split("_Block_")[0]


def main() -> int:
    wsi = pd.read_excel(ROOT / "wsi_labels.xlsx")
    sid_to_pid = {str(r.slide_id): int(r.patient_id) for _, r in wsi.iterrows()}

    def slides(rel: str) -> set[str]:
        df = pd.read_excel(ROOT / rel)
        col = "image_name" if "image_name" in df.columns else df.columns[0]
        return {slide_id_from_patch(str(x)) for x in df[col]}

    def patients(rel: str) -> set[int]:
        return {sid_to_pid[s] for s in slides(rel)}

    fail = False

    def check(name: str, overlap: set) -> None:
        nonlocal fail
        ok = len(overlap) == 0
        print(f"{'OK' if ok else 'FAIL'} {name}: overlap={len(overlap)}")
        if not ok:
            fail = True
            print("  e.g.", list(overlap)[:5])

    tt, te = "partition/Test/Train.xlsx", "partition/Test/Test.xlsx"
    check("Test/Train ∩ Test/Test slides", slides(tt) & slides(te))
    check("Test/Train ∩ Test/Test patients", patients(tt) & patients(te))

    for i in range(1, 5):
        tr = f"partition/Validation/Val{i}/Train.xlsx"
        ts = f"partition/Validation/Val{i}/Test.xlsx"
        check(f"Val{i} Train∩Test slides", slides(tr) & slides(ts))
        check(f"Val{i} Train∩Test patients", patients(tr) & patients(ts))

    for i in range(1, 5):
        for j in range(i + 1, 5):
            a = f"partition/Validation/Val{i}/Test.xlsx"
            b = f"partition/Validation/Val{j}/Test.xlsx"
            check(f"Val{i}Test ∩ Val{j}Test slides", slides(a) & slides(b))
            check(f"Val{i}Test ∩ Val{j}Test patients", patients(a) & patients(b))

    val_union = set()
    for i in range(1, 5):
        val_union |= patients(f"partition/Validation/Val{i}/Test.xlsx")
    check("Union ValTest ∩ official Test patients", val_union & patients(te))
    equal = val_union == patients(tt)
    print(f"{'OK' if equal else 'FAIL'} Union ValTest patients == Test/Train patients")
    if not equal:
        fail = True

    print(
        f"counts: TestTrain patients={len(patients(tt))} TestTest={len(patients(te))} "
        f"slides Train/Test={len(slides(tt))}/{len(slides(te))}"
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
