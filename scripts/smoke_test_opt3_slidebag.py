"""CPU smoke for Option 3 slide-bag + dual ISUP loss helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from train.grade_head import (  # noqa: E402
    ISUPGradeHead,
    aggregate_softmax_probs,
    derived_isup_ce_from_seg_probs,
    grade_head_ce,
)
from train.slide_bag_dataset import (  # noqa: E402
    SlideBagPatchDataset,
    load_patch_batch,
    summarize_bags,
)


def main() -> None:
    ds = SlideBagPatchDataset(
        "outputs/splits/panda_train.csv",
        max_patches_per_slide=2,
        lazy=True,
        allow_missing_h5=True,
    )
    stats = summarize_bags(ds)
    assert stats["max_patches"] == 323
    assert stats["min_patches"] == 3
    bag = ds[0]
    assert "patch_indices" in bag and bag["patch_indices"].numel() <= 2
    assert "images" not in bag
    imgs, masks, weights = load_patch_batch(ds.base, bag["patch_indices"])
    assert imgs.shape[0] == bag["patch_indices"].numel()
    print(
        "bag ok",
        bag["image_id"],
        "n",
        int(bag["patch_indices"].numel()),
        "loaded",
        tuple(imgs.shape),
        "isup",
        int(bag["isup"]),
    )

    logits = [torch.randn(2, 6, 8, 8), torch.randn(1, 6, 8, 8)]
    mean_p = aggregate_softmax_probs(logits)
    assert mean_p.shape == (6,)
    loss, hard = derived_isup_ce_from_seg_probs(mean_p, int(bag["isup"]))
    assert loss.ndim == 0 and 0 <= hard <= 5
    print("derived_isup loss", float(loss), "hard", hard)

    head = ISUPGradeHead(32)
    g = head(torch.randn(32))
    gl = grade_head_ce(g, int(bag["isup"]))
    assert gl.ndim == 0
    print("grade_head loss", float(gl))
    print("SMOKE_OPT3_OK")


if __name__ == "__main__":
    main()
