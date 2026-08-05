"""Evaluate a baseline Gleason checkpoint on a patch split."""

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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if os.environ.get("TORCH_CUDNN_ENABLED", "1").strip() in {"0", "false", "False", "no"}:
    torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import CLASS_NAMES, NUM_CLASSES, OUTPUTS  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402
from train.model import build_model  # noqa: E402
from train.uni2_upernet import build_uni2_upernet  # noqa: E402
from train.lora_vit import apply_lora_to_vit_qkv  # noqa: E402

SPLITS_DIR = OUTPUTS / "splits"
EVAL_DIR = OUTPUTS / "evaluation"
SPLIT_ALIASES = {
    "panda_val": SPLITS_DIR / "panda_val.csv",
    "panda_test": SPLITS_DIR / "panda_test.csv",
    "val": SPLITS_DIR / "panda_val.csv",
    "test": SPLITS_DIR / "panda_test.csv",
}
CLASS_ORDER = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
CANCER_CLASSES = (3, 4, 5)
# PANDA+ annotates Benign/GP3/GP4/GP5 only (see process_panda_plus_geojson.py).
PANDA_PLUS_CLASSES = (2, 3, 4, 5)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl").lower()
        try:
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(hours=2),
                device_id=device,
            )
        except TypeError:
            dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
        return local_rank, rank, world_size, device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return 0, 0, 1, torch.device("cuda", 0)
    return 0, 0, 1, torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class EvalAccumulator:
    """Pixel-level confusion matrix + per-class metrics."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        min_target_class: int = 0,
    ) -> None:
        self.num_classes = num_classes
        # PANDA+: min_target_class=2 → eval_mask = (gt >= 2); unannotated ignored.
        self.min_target_class = min_target_class
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        preds = preds.detach().cpu().numpy().ravel()
        targets = targets.detach().cpu().numpy().ravel()
        valid = (targets >= self.min_target_class) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]
        if len(targets) == 0:
            return
        flat = self.num_classes * targets.astype(np.int64) + preds.astype(np.int64)
        counts = np.bincount(flat, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def all_reduce(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        tensor = torch.from_numpy(self.confusion).to(device=device, dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.confusion = tensor.cpu().numpy()

    @property
    def report_classes(self) -> tuple[int, ...]:
        if self.min_target_class >= 2:
            return PANDA_PLUS_CLASSES
        return tuple(range(self.num_classes))

    def per_class_metrics(self, *, eps: float = 1e-6) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for cls in self.report_classes:
            # Restrict pred mass to labeled GT pixels already (via update mask).
            # Pred stroma/bg on labeled tissue still appear as wrong preds (off-diagonal).
            tp = float(self.confusion[cls, cls])
            # Only count predictions on pixels whose GT is in the report set.
            pred_sum = float(self.confusion[list(self.report_classes), :][:, cls].sum())
            true_sum = float(self.confusion[cls, :].sum())
            dice = (2.0 * tp + eps) / (pred_sum + true_sum + eps)
            iou = (tp + eps) / (pred_sum + true_sum - tp + eps)
            precision = (tp + eps) / (pred_sum + eps)
            recall = (tp + eps) / (true_sum + eps)
            out[CLASS_NAMES[cls]] = {
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            }
        return out

    def summary(self, *, eps: float = 1e-6) -> dict[str, float]:
        per = self.per_class_metrics(eps=eps)
        report = self.report_classes
        dice_vals = [per[CLASS_NAMES[c]]["dice"] for c in report]
        cancer_in_report = [c for c in CANCER_CLASSES if c in report]
        cancer_dice = float(np.mean([per[CLASS_NAMES[c]]["dice"] for c in cancer_in_report]))
        cancer_tp = sum(float(self.confusion[c, c]) for c in cancer_in_report)
        cancer_true = sum(float(self.confusion[c, :].sum()) for c in cancer_in_report)
        cancer_recall = (cancer_tp + eps) / (cancer_true + eps)
        g5_tp = float(self.confusion[5, 5]) if 5 in report else 0.0
        g5_true = float(self.confusion[5, :].sum()) if 5 in report else 0.0
        g5_recall = (g5_tp + eps) / (g5_true + eps)
        return {
            "mean_dice": float(np.mean(dice_vals)),
            "mean_dice_over": "+".join(CLASS_NAMES[c] for c in report),
            "cancer_dice": cancer_dice,
            "cancer_recall": cancer_recall,
            "g5_recall": g5_recall,
        }


def resolve_split(split: str) -> tuple[Path, str]:
    if split in SPLIT_ALIASES:
        return SPLIT_ALIASES[split], split
    path = Path(split)
    if path.exists():
        return path, path.stem
    raise ValueError(f"Unknown split: {split}")


def _peek_state_dict(checkpoint: Path) -> tuple[dict, list[str]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Bad checkpoint state_dict in {checkpoint}")
    # DDP may prefix module.
    if any(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return ckpt, list(state.keys())


def _opt3_flags(keys: list[str]) -> tuple[bool, str, bool]:
    """Return (is_opt3_wrapper, decode_norm, use_lora)."""
    joined = " ".join(keys)
    use_lora = ("lora_A" in joined) or ("lora_B" in joined) or (".lora_" in joined)
    is_opt3 = any(k.startswith("seg.") for k in keys) or ("grade_head." in joined)
    # Opt3 Omar stack uses GroupNorm; teacher A / plain UNI2 use BN.
    decode_norm = "gn" if (is_opt3 or use_lora) else "bn"
    return is_opt3, decode_norm, use_lora


def load_model_weights(checkpoint: Path, model: torch.nn.Module) -> dict:
    ckpt, keys = _peek_state_dict(checkpoint)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    # Option 3 saves SegPlusGrade: keep only seg.* for pixel eval.
    if any(k.startswith("seg.") for k in state):
        state = {k[len("seg.") :]: v for k, v in state.items() if k.startswith("seg.")}
    model.load_state_dict(state, strict=True)
    return ckpt


def detect_arch(checkpoint: Path, arch: str) -> str:
    """Resolve model architecture; prefer explicit --arch, else peek at state dict keys."""
    if arch != "auto":
        return arch
    _ckpt, keys = _peek_state_dict(checkpoint)
    joined = " ".join(keys[:120])
    if (
        "decode_head." in joined
        or "backbone.blocks." in joined
        or "projections." in joined
        or "seg.backbone." in joined
    ):
        return "uni2_upernet"
    return "baseline"


def build_eval_model(
    arch: str,
    *,
    decode_norm: str = "bn",
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
) -> torch.nn.Module:
    if arch == "uni2_upernet":
        # pretrained=False: weights come from the training checkpoint (no HF needed).
        model = build_uni2_upernet(
            num_classes=NUM_CLASSES,
            freeze_backbone=False,
            pretrained=False,
            decode_norm=decode_norm,
        )
        if use_lora:
            n = apply_lora_to_vit_qkv(model.backbone, r=lora_r, alpha=lora_alpha)
            print(f"Eval LoRA QKV wraps={n} | decode_norm={decode_norm}")
        return model
    if arch == "baseline":
        return build_model(num_classes=NUM_CLASSES)
    raise ValueError(f"Unknown arch: {arch}")


ARCH_TITLES = {
    "baseline": "Model A — EfficientNet-B4 + UNet++ — Raw (no normalization)",
    "uni2_upernet": "Model C — UNI2-h + UPerNet — Raw (no normalization)",
}


def run_eval(
    *,
    checkpoint: Path,
    split: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
    device: torch.device,
    rank: int,
    world_size: int,
    allow_missing_h5: bool = False,
    mask_dir: Path | None = None,
    mask_suffix: str = "_mask.tiff",
    prefer_h5_masks: bool = True,
    arch: str = "auto",
    panda_plus_eval: bool = False,
) -> tuple[EvalAccumulator, dict]:
    split_csv, split_name = resolve_split(split)
    ds_kwargs: dict = {
        "mode": mode,
        "allow_missing_h5": allow_missing_h5,
        "mask_suffix": mask_suffix,
        "prefer_h5_masks": prefer_h5_masks,
    }
    if mask_dir is not None:
        ds_kwargs["mask_dir"] = mask_dir
    ds = BaselinePatchDataset(split_csv, **ds_kwargs)
    sampler = DistributedSampler(ds, shuffle=False) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    resolved_arch = detect_arch(checkpoint, arch)
    _ckpt_peek, peek_keys = _peek_state_dict(checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(peek_keys)
    model = build_eval_model(
        resolved_arch, decode_norm=decode_norm, use_lora=use_lora
    )
    ckpt_meta = load_model_weights(checkpoint, model)
    if _is_opt3 or use_lora:
        print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora}")
    model.to(device)
    model.eval()
    if device.type == "cuda":
        # Warmup with B>=2: decode BN uses batch stats (track_running_stats=False),
        # so B=1 raises "Expected more than 1 value per channel".
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=amp):
                _ = model(torch.zeros(2, 3, 512, 512, device=device))

    # PANDA+: score only labeled pixels (gt >= 2). No pred remapping.
    min_tgt = 2 if panda_plus_eval else 0
    acc = EvalAccumulator(min_target_class=min_tgt)
    with torch.no_grad():
        for images, masks, _weights in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            acc.update(logits.argmax(dim=1), masks)

    acc.all_reduce(device)

    meta = {
        "checkpoint": str(checkpoint),
        "arch": resolved_arch,
        "split": split_name,
        "split_csv": str(split_csv),
        "mode": mode,
        "n_patches": len(ds),
        "epoch": int(ckpt_meta.get("epoch", -1)),
        "val_mean_dice_at_save": float(
            ckpt_meta.get("val_mean_dice", ckpt_meta.get("metrics", {}).get("mean_dice", float("nan")))
        ),
        "val_cancer_dice_at_save": float(
            ckpt_meta.get("val_cancer_dice", ckpt_meta.get("metrics", {}).get("cancer_dice", float("nan")))
        ),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "mask_suffix": mask_suffix,
        "allow_missing_h5": allow_missing_h5,
        "panda_plus_eval": panda_plus_eval,
        "min_target_class": min_tgt,
        "report_classes": [CLASS_NAMES[c] for c in acc.report_classes],
    }
    return acc, meta


def confusion_df(acc: EvalAccumulator) -> pd.DataFrame:
    """Confusion with report-class rows; all pred columns (incl. stroma/bg misses)."""
    rows = list(acc.report_classes)
    df = pd.DataFrame(
        acc.confusion[np.ix_(rows, list(range(acc.num_classes)))],
        index=[f"true_{CLASS_NAMES[i]}" for i in rows],
        columns=[f"pred_{n}" for n in CLASS_ORDER],
    )
    return df


def save_results(
    acc: EvalAccumulator,
    meta: dict,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per = acc.per_class_metrics()
    summary = acc.summary()
    conf = confusion_df(acc)

    rows = []
    for cls in acc.report_classes:
        name = CLASS_NAMES[cls]
        m = per[name]
        rows.append({
            "class": name,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
        })
    rows.extend([
        {"class": "mean_dice", "dice": summary["mean_dice"], "iou": np.nan, "precision": np.nan, "recall": np.nan},
        {"class": "cancer_dice", "dice": summary["cancer_dice"], "iou": np.nan, "precision": np.nan, "recall": np.nan},
        {"class": "cancer_recall", "dice": np.nan, "iou": np.nan, "precision": np.nan, "recall": summary["cancer_recall"]},
        {"class": "g5_recall", "dice": np.nan, "iou": np.nan, "precision": np.nan, "recall": summary["g5_recall"]},
    ])
    metrics_df = pd.DataFrame(rows)

    conf_path = out_csv.with_name(out_csv.stem + "_confusion.csv")
    meta_path = out_csv.with_name(out_csv.stem + "_meta.json")

    metrics_df.to_csv(out_csv, index=False, float_format="%.6f")
    conf.to_csv(conf_path, float_format="%.0f")
    meta_path.write_text(json.dumps({**meta, **summary}, indent=2), encoding="utf-8")


def print_report(split_label: str, acc: EvalAccumulator, meta: dict) -> None:
    per = acc.per_class_metrics()
    summary = acc.summary()
    arch = meta.get("arch", "baseline")
    print()
    print(ARCH_TITLES.get(arch, arch))
    print(f"Evaluated on: {split_label}")
    print(f"Checkpoint:   {meta['checkpoint']}")
    print(f"Patches:      {meta['n_patches']}")
    if meta.get("epoch", -1) >= 0:
        print(f"Saved epoch:  {meta['epoch']}")
    if meta.get("panda_plus_eval"):
        print("PANDA+ scoring: eval_mask=(gt>=2); report classes benign/G3/G4/G5 only; no pred remap")
    print()
    print(f"{'':12s}  {'Dice':>6s}  {'IoU':>6s}  {'Precision':>9s}  {'Recall':>6s}")
    for cls in acc.report_classes:
        name = CLASS_NAMES[cls]
        m = per[name]
        print(
            f"{name:12s}  {m['dice']:6.3f}  {m['iou']:6.3f}  "
            f"{m['precision']:9.3f}  {m['recall']:6.3f}"
        )
    print()
    mean_note = summary.get("mean_dice_over", "all_classes")
    print(f"mean_dice:     {summary['mean_dice']:.3f}  (over {mean_note})")
    print(f"cancer_dice:   {summary['cancer_dice']:.3f}  (G3+G4+G5 averaged)")
    print(f"cancer_recall: {summary['cancer_recall']:.3f}  (cancer pixels found)")
    print(f"g5_recall:     {summary['g5_recall']:.3f}")
    print()
    print("Confusion matrix (rows=true labeled, cols=pred — stroma/bg preds = misses):")
    conf = confusion_df(acc)
    print(conf.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", required=True, help="panda_val, panda_test, or path to CSV")
    parser.add_argument("--mode", choices=("raw", "normalized", "normalized_ink_raw"), default="raw")
    parser.add_argument(
        "--arch",
        choices=("auto", "baseline", "uni2_upernet"),
        default="auto",
        help="Model architecture (auto-detect from checkpoint keys)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-h5",
        action="store_true",
        help="Fall back to OpenSlide when H5 is missing or coord not in H5",
    )
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="_mask.tiff",
        help="Mask filename suffix, e.g. _pandaplus_mask.png",
    )
    parser.add_argument(
        "--prefer-h5-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use H5 masks when present (disable for PANDA+ PNG GT)",
    )
    parser.add_argument(
        "--panda-plus-eval",
        action="store_true",
        help=(
            "PANDA+ protocol: score only gt>=2 pixels; report benign/G3/G4/G5 only. "
            "No pred remapping — stroma pred on benign GT is a normal miss."
        ),
    )
    args = parser.parse_args()

    local_rank, rank, world_size, device = setup_distributed()
    try:
        split_csv, split_name = resolve_split(args.split)
        resolved_arch = detect_arch(args.checkpoint, args.arch)
        default_stem = (
            f"uni2_upernet_{args.mode}_{split_name}"
            if resolved_arch == "uni2_upernet"
            else f"baseline_{args.mode}_{split_name}"
        )
        if args.panda_plus_eval and args.out is None:
            default_stem = f"{default_stem}_labeled"
        out_csv = args.out or EVAL_DIR / f"{default_stem}.csv"

        acc, meta = run_eval(
            checkpoint=args.checkpoint,
            split=args.split,
            mode=args.mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            device=device,
            rank=rank,
            world_size=world_size,
            allow_missing_h5=args.allow_missing_h5,
            mask_dir=args.mask_dir,
            mask_suffix=args.mask_suffix,
            prefer_h5_masks=args.prefer_h5_masks,
            arch=args.arch,
            panda_plus_eval=args.panda_plus_eval,
        )
        meta["world_size"] = world_size
        if rank == 0:
            save_results(acc, meta, out_csv)
            split_label = (
                "PANDA val" if "val" in split_name and "plus" not in split_name.lower()
                else "PANDA held-out test" if "test" in split_name
                else "PANDA+" if "plus" in split_name.lower()
                else split_name
            )
            print_report(split_label, acc, meta)
            print(f"Saved metrics:   {out_csv}")
            print(f"Saved confusion: {out_csv.with_name(out_csv.stem + '_confusion.csv')}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
