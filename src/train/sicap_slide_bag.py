"""SICAPv2 slide-bag dataset for Opt3-style fine-tune.

Official masks use {0,3,4,5}=NC/G3/G4/G5. Slide ISUP from wsi_labels
(Gleason primary+secondary → ISUP 0–5). Patient CV via Val{fold} Train/Test.xlsx.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from train.augmentations import SlideAugParams, apply_slide_consistent, sample_slide_aug_params

DEFAULT_SICAP_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")


def gleason_to_isup(primary: int, secondary: int) -> int:
    p, s = int(primary), int(secondary)
    if p == 0 and s == 0:
        return 0
    if p == 3 and s == 3:
        return 1
    if p == 3 and s == 4:
        return 2
    if p == 4 and s == 3:
        return 3
    if (p + s) == 8 or (p, s) in {(3, 5), (5, 3), (4, 4)}:
        return 4
    if (p + s) >= 9:
        return 5
    if p == 0 or s == 0:
        return 0
    raise ValueError(f"unmapped Gleason {p}+{s}")


def slide_id_from_patch(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_Block_")[0]


def load_split_names(root: Path, fold: int, split: str) -> list[str]:
    path = root / "partition" / "Validation" / f"Val{fold}" / f"{split}.xlsx"
    df = pd.read_excel(path)
    col = "image_name" if "image_name" in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def load_isup_by_slide(root: Path) -> dict[str, int]:
    wsi = pd.read_excel(root / "wsi_labels.xlsx")
    out: dict[str, int] = {}
    for _, row in wsi.iterrows():
        sid = str(row["slide_id"])
        out[sid] = gleason_to_isup(int(row["Gleason_primary"]), int(row["Gleason_secondary"]))
    return out


class SicapPatchStore:
    """Indexable patch store (JPG + PNG mask) for micro-batch loads."""

    def __init__(
        self,
        root: Path,
        patch_names: list[str],
        *,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.augment = augment
        self._slide_aug: SlideAugParams | None = None
        self.stems: list[str] = []
        skipped = 0
        for name in patch_names:
            stem = Path(name).stem
            img = self.images / f"{stem}.jpg"
            msk = self.masks / f"{stem}.png"
            if not img.is_file() or not msk.is_file():
                skipped += 1
                continue
            self.stems.append(stem)
        self.skipped = skipped
        self.slide_ids = [slide_id_from_patch(s) for s in self.stems]

    def __len__(self) -> int:
        return len(self.stems)

    def set_slide_aug_params(self, params: SlideAugParams | None) -> None:
        self._slide_aug = params

    def clear_open_handles(self) -> None:
        return

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.stems[idx]
        rgb = np.array(Image.open(self.images / f"{stem}.jpg").convert("RGB"), dtype=np.uint8)
        msk = np.array(Image.open(self.masks / f"{stem}.png"))
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = msk.astype(np.int64)
        w = np.ones_like(msk, dtype=np.float32)
        if self.augment and self._slide_aug is not None:
            rgb, msk_o, w_o = apply_slide_consistent(rgb, msk, w, self._slide_aug)
            assert msk_o is not None and w_o is not None
            msk, w = msk_o, w_o
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(msk.astype(np.int64))
        wt = torch.from_numpy(w.astype(np.float32))
        return x, y, wt


def load_patch_batch(
    store: SicapPatchStore, patch_idxs: list[int] | Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(patch_idxs, Tensor):
        patch_idxs = [int(i) for i in patch_idxs.tolist()]
    images, masks, weights = [], [], []
    for pi in patch_idxs:
        image_t, mask_t, weight_t = store[int(pi)]
        images.append(image_t)
        masks.append(mask_t)
        weights.append(weight_t)
    return torch.stack(images, 0), torch.stack(masks, 0), torch.stack(weights, 0)


class SicapSlideBagDataset(Dataset):
    """One item = one slide bag (lazy indices + ISUP)."""

    def __init__(
        self,
        root: Path | str = DEFAULT_SICAP_ROOT,
        *,
        fold: int = 1,
        split: str = "Train",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.fold = int(fold)
        self.split = split
        self.rng = np.random.default_rng(seed)
        names = load_split_names(self.root, self.fold, split)
        self.base = SicapPatchStore(self.root, names, augment=augment)
        isup_map = load_isup_by_slide(self.root)
        groups: dict[str, list[int]] = {}
        for i, sid in enumerate(self.base.slide_ids):
            groups.setdefault(sid, []).append(i)
        self.slide_ids = sorted(groups.keys())
        self.indices_by_slide = {s: groups[s] for s in self.slide_ids}
        missing = [s for s in self.slide_ids if s not in isup_map]
        if missing:
            raise ValueError(f"{len(missing)} SICAP slides lack wsi_labels ISUP e.g. {missing[:3]}")
        self.isup_by_slide = {s: int(isup_map[s]) for s in self.slide_ids}

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        slide_id = self.slide_ids[idx]
        patch_idxs = list(self.indices_by_slide[slide_id])
        return {
            "image_id": slide_id,
            "patch_indices": torch.tensor(patch_idxs, dtype=torch.long),
            "isup": torch.tensor(int(self.isup_by_slide[slide_id]), dtype=torch.long),
        }


class SicapPatchDataset(Dataset):
    """Flat patch dataset for val / class-weight scans."""

    def __init__(self, store: SicapPatchStore) -> None:
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.store[idx]

    def clear_open_handles(self) -> None:
        self.store.clear_open_handles()


def slide_bag_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"SicapSlideBag expects batch_size=1, got {len(batch)}")
    return batch[0]


def summarize_bags(dataset: SicapSlideBagDataset) -> dict:
    sizes = [len(dataset.indices_by_slide[s]) for s in dataset.slide_ids]
    return {
        "n_slides": len(sizes),
        "n_patches": int(sum(sizes)),
        "min_patches": int(min(sizes)) if sizes else 0,
        "median_patches": float(np.median(sizes)) if sizes else 0.0,
        "max_patches": int(max(sizes)) if sizes else 0,
        "mean_patches": float(np.mean(sizes)) if sizes else 0.0,
    }


def estimate_class_weights(store: SicapPatchStore, *, max_patches: int = 2000, seed: int = 42) -> Tensor:
    """Inverse-freq weights over 6 PANDA classes (1,2 expected near-absent)."""
    rng = np.random.default_rng(seed)
    n = len(store)
    idxs = list(range(n))
    if n > max_patches:
        idxs = sorted(rng.choice(n, size=max_patches, replace=False).tolist())
    counts = np.zeros(6, dtype=np.float64)
    for i in idxs:
        _, y, _ = store[i]
        u, c = np.unique(y.numpy(), return_counts=True)
        for lab, cnt in zip(u.tolist(), c.tolist()):
            if 0 <= int(lab) < 6:
                counts[int(lab)] += float(cnt)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)
