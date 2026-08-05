"""Phase 2: compare model-derived slide ISUP against clinical metadata.

This is diagnostic-only: it reads images and a checkpoint, writes a report,
and never modifies masks, manifests, or training labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

torch.backends.cudnn.benchmark = False
if os.environ.get("TORCH_CUDNN_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
    torch.backends.cudnn.enabled = False

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    detect_arch,
    load_model_weights,
)
from patch_utils import NUM_CLASSES, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset, _preprocess_image  # noqa: E402


DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"
DEFAULT_METADATA = PROJECT / "data" / "train.csv"
DEFAULT_OUT = PROJECT / "outputs" / "pseudo_label" / "diagnostic_report.csv"


def gleason_to_isup(primary: int, secondary: int) -> int:
    """Map Gleason patterns to ISUP grade group."""
    score = (int(primary), int(secondary))
    if score in {(0, 0)}:
        return 0
    if score == (3, 3):
        return 1
    if score == (3, 4):
        return 2
    if score == (4, 3):
        return 3
    if score in {(4, 4), (3, 5), (5, 3)}:
        return 4
    if score in {(4, 5), (5, 4), (5, 5)}:
        return 5
    raise ValueError(f"Unsupported Gleason pattern: {primary}+{secondary}")


def derive_grade(
    class_pixel_counts: np.ndarray,
    *,
    min_area_pct: float = 0.05,
) -> tuple[str, int]:
    """Derive Gleason/ISUP from predicted G3/G4/G5 pixel proportions."""
    cancer = np.asarray(class_pixel_counts[3:6], dtype=np.float64)
    total_cancer = float(cancer.sum())
    if total_cancer <= 0:
        return "benign", 0

    retained = [
        (grade, float(cancer[grade - 3] / total_cancer))
        for grade in (3, 4, 5)
        if float(cancer[grade - 3] / total_cancer) >= min_area_pct
    ]
    if not retained:
        return "benign", 0
    retained.sort(key=lambda item: (-item[1], item[0]))
    primary = retained[0][0]
    secondary = retained[1][0] if len(retained) > 1 else primary
    return f"{primary}+{secondary}", gleason_to_isup(primary, secondary)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return 0, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl").lower()
    dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    return local_rank, rank, world_size, device


class SlideInferenceDataset(Dataset):
    """Image-only view of BaselinePatchDataset with stable slide indices."""

    def __init__(
        self,
        split_df: pd.DataFrame,
        slide_to_index: dict[str, int],
        *,
        mode: str,
        allow_missing_h5: bool = True,
    ) -> None:
        # allow_missing_h5=True: PANDA+ coords may not be in train H5 index → WSI fallback
        self.base = BaselinePatchDataset(
            DEFAULT_SPLIT, mode=mode, allow_missing_h5=allow_missing_h5
        )
        self.base.df = split_df.reset_index(drop=True)
        self.df = self.base.df
        self.slide_to_index = slide_to_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        slide_id = str(row["image_id"])
        rgb = self.base._read_image(slide_id, int(row["x"]), int(row["y"]))
        return _preprocess_image(rgb), self.slide_to_index[slide_id]


class RankSlideSampler(Sampler[int]):
    """Shard whole slides across ranks, avoiding DistributedSampler padding."""

    def __init__(self, dataset: SlideInferenceDataset, rank: int, world_size: int) -> None:
        self.indices = [
            i
            for i, slide_id in enumerate(dataset.df["image_id"].astype(str))
            if dataset.slide_to_index[slide_id] % world_size == rank
        ]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def select_smoke_slides(
    split_df: pd.DataFrame,
    metadata: pd.DataFrame,
    max_slides: int | None,
) -> list[str]:
    all_slides = sorted(split_df["image_id"].astype(str).unique())
    if not max_slides or max_slides >= len(all_slides):
        return all_slides
    meta = metadata[metadata["image_id"].astype(str).isin(all_slides)].copy()
    meta["image_id"] = meta["image_id"].astype(str)
    selected: list[str] = []
    # Round-robin across ISUP groups so a smoke test is not all benign.
    groups = {
        int(grade): sorted(group["image_id"].tolist())
        for grade, group in meta.groupby("isup_grade")
    }
    offset = 0
    while len(selected) < max_slides:
        added = False
        for grade in sorted(groups):
            if offset < len(groups[grade]):
                selected.append(groups[grade][offset])
                added = True
                if len(selected) == max_slides:
                    break
        if not added:
            break
        offset += 1
    return selected


def reduce_counts(local_counts: np.ndarray, device: torch.device) -> np.ndarray:
    if not (dist.is_available() and dist.is_initialized()):
        return local_counts
    backend = dist.get_backend()
    reduce_device = device if backend == "nccl" else torch.device("cpu")
    tensor = torch.from_numpy(local_counts).to(reduce_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 ISUP consistency diagnostic")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", default="raw", choices=("raw", "normalized", "normalized_ink_raw"))
    parser.add_argument("--arch", default="auto", choices=("auto", "baseline", "uni2_upernet"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--min-area-pct", type=float, default=0.05)
    parser.add_argument("--max-slides", type=int, default=None, help="Smoke test only")
    parser.add_argument(
        "--allow-missing-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back to WSI when patch coords missing from H5 (needed for PANDA+)",
    )
    parser.add_argument(
        "--max-patches-per-slide",
        type=int,
        default=None,
        help="Smoke test only; makes derived grades approximate",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_area_pct < 1.0:
        parser.error("--min-area-pct must be in [0, 1)")

    _local_rank, rank, world_size, device = setup_distributed()
    split_df = pd.read_csv(args.split, dtype={"image_id": str})
    metadata = pd.read_csv(args.metadata, dtype={"image_id": str})
    if metadata["image_id"].duplicated().any():
        raise ValueError("Metadata has duplicate image_id rows")

    slide_ids = select_smoke_slides(split_df, metadata, args.max_slides)
    split_df = split_df[split_df["image_id"].isin(slide_ids)].copy()
    split_df = split_df.sort_values(["image_id", "y", "x"])
    if args.max_patches_per_slide:
        split_df = split_df.groupby("image_id", sort=False).head(args.max_patches_per_slide)
    slide_ids = sorted(split_df["image_id"].unique())
    slide_to_index = {slide_id: i for i, slide_id in enumerate(slide_ids)}
    patch_counts = split_df.groupby("image_id").size().to_dict()

    dataset = SlideInferenceDataset(
        split_df, slide_to_index, mode=args.mode, allow_missing_h5=args.allow_missing_h5
    )
    sampler = RankSlideSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    resolved_arch = detect_arch(args.checkpoint, args.arch)
    # Match evaluate.run_eval: Opt3 ckpts need GN + LoRA wraps before load.
    _ckpt_peek, peek_keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(peek_keys)
    model = build_eval_model(
        resolved_arch, decode_norm=decode_norm, use_lora=use_lora
    )
    ckpt = load_model_weights(args.checkpoint, model)
    model.to(device).eval()
    if rank == 0 and (_is_opt3 or use_lora):
        print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora}")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    local_counts = np.zeros((len(slide_ids), NUM_CLASSES), dtype=np.int64)

    if rank == 0:
        print(
            f"ISUP diagnostic: slides={len(slide_ids)} patches={len(dataset)} "
            f"world={world_size} checkpoint={args.checkpoint}"
        )
        if args.max_slides or args.max_patches_per_slide:
            print("SMOKE TEST: results are not a full diagnostic")

    with torch.inference_mode():
        for step, (images, slide_indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                out = model(images)
                logits = out[0] if isinstance(out, tuple) else out
                preds = logits.argmax(dim=1)
            for batch_i, slide_index in enumerate(slide_indices.tolist()):
                bincount = torch.bincount(
                    preds[batch_i].reshape(-1), minlength=NUM_CLASSES
                )
                local_counts[slide_index] += bincount.cpu().numpy()
            if rank == 0 and step % 100 == 0:
                print(f"rank0 batches={step}/{len(loader)}", flush=True)

    counts = reduce_counts(local_counts, device)
    if rank == 0:
        metadata_index = metadata.set_index("image_id")
        rows: list[dict] = []
        for slide_id in slide_ids:
            if slide_id not in metadata_index.index:
                raise KeyError(f"Missing metadata for slide {slide_id}")
            idx = slide_to_index[slide_id]
            gleason, derived_isup = derive_grade(counts[idx], min_area_pct=args.min_area_pct)
            meta_row = metadata_index.loc[slide_id]
            metadata_isup = int(meta_row["isup_grade"])
            total_cancer = int(counts[idx, 3:6].sum())
            rows.append(
                {
                    "slide_id": slide_id,
                    "metadata_gleason": str(meta_row["gleason_score"]),
                    "metadata_isup": metadata_isup,
                    "derived_gleason": gleason,
                    "derived_isup": derived_isup,
                    "match": derived_isup == metadata_isup,
                    "n_patches": int(patch_counts[slide_id]),
                    **{f"pred_pixels_{c}": int(counts[idx, c]) for c in range(NUM_CLASSES)},
                    **{
                        f"cancer_frac_g{grade}": (
                            float(counts[idx, grade] / total_cancer) if total_cancer else 0.0
                        )
                        for grade in (3, 4, 5)
                    },
                }
            )
        report = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
        by_grade = (
            report.groupby("metadata_isup")["match"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"sum": "matches", "mean": "match_rate"})
        )
        summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_val_cancer_dice": float(
                ckpt.get("val_cancer_dice", ckpt.get("metrics", {}).get("cancer_dice", np.nan))
            ),
            "split": str(args.split),
            "total_slides": int(len(report)),
            "total_patches": int(len(dataset)),
            "matches": int(report["match"].sum()),
            "mismatches": int((~report["match"]).sum()),
            "match_rate": float(report["match"].mean()),
            "min_area_pct": args.min_area_pct,
            "smoke_test": bool(args.max_slides or args.max_patches_per_slide),
            "by_metadata_isup": {
                str(int(grade)): {
                    "count": int(row["count"]),
                    "matches": int(row["matches"]),
                    "match_rate": float(row["match_rate"]),
                }
                for grade, row in by_grade.iterrows()
            },
        }
        summary_path = args.out.with_name(args.out.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nPhase 2 ISUP diagnostic")
        print(f"Total slides: {len(report)}")
        print(f"Consistent: {report['match'].sum()} ({report['match'].mean():.1%})")
        print(f"Inconsistent: {(~report['match']).sum()}")
        print("\nBy metadata ISUP:")
        print(by_grade.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nReport: {args.out}")
        print(f"Summary: {summary_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
