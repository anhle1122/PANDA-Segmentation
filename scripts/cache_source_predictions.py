"""Cache a frozen teacher's per-pixel argmax + confidence.

Teacher pack (never overwrite an existing slide file):

    outputs/pseudo_label/teacher_<tag>_epXXX/<slide_id>_srcpred.h5
        coords   (N, 2) int32
        preds    (N, 512, 512) uint8 gzip   -- argmax class
        maxprob  (N, 512, 512) float16 gzip -- max softmax

Also writes pack_config.json + clinical_isup.csv (copy of metadata, not model-derived).

Legacy mode: --manifest only caches slides that Rules 1-3 will rewrite.
Teacher-pack mode: --all-slides caches every split slide (needed for the
three-way ISUP referee).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

torch.backends.cudnn.benchmark = False
if os.environ.get("TORCH_CUDNN_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
    torch.backends.cudnn.enabled = False

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    detect_arch,
    load_model_weights,
)
from patch_utils import PATCH_SIZE, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset, _preprocess_image  # noqa: E402
from train.pseudo_label_rules import NO_CORRECTION_RULES  # noqa: E402
from train.uni2_upernet import (  # noqa: E402
    checkpoint_has_bn_running_stats,
    disable_bn_running_stats,
)

DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"
DEFAULT_MANIFEST = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"
DEFAULT_OUT_DIR = PROJECT / "outputs" / "pseudo_label" / "round1_source_pred"
DEFAULT_CLINICAL = PROJECT / "data" / "train.csv"

CACHE_SUFFIX = "_srcpred.h5"


def cache_path(out_dir: Path, slide_id: str) -> Path:
    return out_dir / f"{slide_id}{CACHE_SUFFIX}"


def slides_needing_cache(manifest: pd.DataFrame) -> list[str]:
    corrected = manifest[~manifest["rule_applied"].isin(NO_CORRECTION_RULES)]
    return sorted(corrected["slide_id"].astype(str).unique())


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


class PatchInferenceDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        split_df: pd.DataFrame,
        slide_to_index: dict[str, int],
        *,
        mode: str,
    ) -> None:
        self.base = BaselinePatchDataset(split_csv, mode=mode, allow_missing_h5=True)
        self.base.df = split_df.reset_index(drop=True)
        self.df = self.base.df
        self.slide_to_index = slide_to_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        row = self.df.iloc[idx]
        slide_id = str(row["image_id"])
        x, y = int(row["x"]), int(row["y"])
        rgb = self.base._read_image(slide_id, x, y)
        return _preprocess_image(rgb), self.slide_to_index[slide_id], x, y


class RankSlideSampler(Sampler[int]):
    def __init__(self, dataset: PatchInferenceDataset, rank: int, world_size: int) -> None:
        self.indices = [
            i
            for i, slide_id in enumerate(dataset.df["image_id"].astype(str))
            if dataset.slide_to_index[slide_id] % world_size == rank
        ]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def load_source_model(checkpoint: Path, arch: str, device: torch.device, *, verbose: bool):
    """Match evaluate.py: Opt3 LoRA+GN, strip seg., strict load."""
    resolved_arch = detect_arch(checkpoint, arch)
    _ckpt, peek_keys = _peek_state_dict(checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(peek_keys)
    model = build_eval_model(resolved_arch, decode_norm=decode_norm, use_lora=use_lora)
    ckpt = load_model_weights(checkpoint, model)
    state = ckpt.get("model_state_dict", {})
    if isinstance(state, dict) and not checkpoint_has_bn_running_stats(state):
        n = disable_bn_running_stats(model)
        if verbose and n:
            print(f"No BN running stats -> {n} BN module(s) use batch stats")
    if verbose and (_is_opt3 or use_lora):
        print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora}")
    model.to(device).eval()
    return model, resolved_arch, ckpt


def write_slide_cache(
    out_dir: Path,
    slide_id: str,
    coords: np.ndarray,
    preds: np.ndarray,
    maxprob: np.ndarray | None,
) -> None:
    final = cache_path(out_dir, slide_id)
    if final.exists():
        return
    tmp = final.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords.astype(np.int32))
        f.create_dataset("preds", data=preds.astype(np.uint8), compression="gzip", compression_opts=4)
        if maxprob is not None:
            f.create_dataset(
                "maxprob",
                data=maxprob.astype(np.float16),
                compression="gzip",
                compression_opts=4,
            )
    tmp.replace(final)


def _write_pack_sidecar(
    out_dir: Path,
    *,
    checkpoint: Path,
    ckpt: dict,
    args,
    n_slides: int,
    n_patches: int,
    status: str,
) -> None:
    clinical_src = Path(args.clinical_csv)
    clinical_dst = out_dir / "clinical_isup.csv"
    if clinical_src.is_file() and not clinical_dst.exists():
        meta = pd.read_csv(clinical_src, dtype={"image_id": str})
        keep = [c for c in ("image_id", "isup_grade", "gleason_score", "data_provider") if c in meta.columns]
        meta.loc[:, keep].to_csv(clinical_dst, index=False)
    metrics = ckpt.get("metrics") or {}
    extra = ckpt.get("extra") or {}
    payload = {
        "status": status,
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint.resolve()),
        "epoch": int(ckpt.get("epoch", metrics.get("epoch", -1)) or -1),
        "val_cancer_dice": float(metrics.get("cancer_dice", extra.get("best_cancer_dice", -1)) or -1),
        "run_tag": args.run_tag,
        "lambda_slide": float(args.lambda_slide),
        "lambda_grade": float(args.lambda_grade),
        "lambda_slide_warmup": "0 ep1-5, ramp 6-9, 0.3 from ep10 (Omar-6)",
        "min_area_pct": 0.0,
        "min_slide_patches": 5,
        "adjacent_soft_alpha": 0.1,
        "include_benign_soft": True,
        "seed": 42,
        "split": str(Path(args.split).resolve()),
        "n_slides": int(n_slides),
        "n_patches": int(n_patches),
        "has_maxprob": bool(args.write_maxprob),
        "never_overwrite": True,
    }
    tmp = out_dir / ".pack_config.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "pack_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache teacher preds + max-softmax")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--mode", default="raw", choices=("raw", "normalized", "normalized_ink_raw"))
    parser.add_argument("--arch", default="auto", choices=("auto", "baseline", "uni2_upernet"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--all-slides",
        action="store_true",
        help="Cache every split slide (teacher pack). Default: manifest correction subset.",
    )
    parser.add_argument("--write-maxprob", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--lambda-slide", type=float, default=0.3)
    parser.add_argument("--lambda-grade", type=float, default=0.3)
    args = parser.parse_args()

    _local_rank, rank, world_size, device = setup_distributed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = args.out_dir / "pack_config.json"
    if cfg_path.exists() and not args.overwrite:
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        prev_ckpt = str(Path(prev.get("checkpoint", "")).resolve())
        this_ckpt = str(args.checkpoint.resolve())
        if prev_ckpt and prev_ckpt != this_ckpt:
            raise SystemExit(
                f"Refusing to mix checkpoints in {args.out_dir}:\n  have {prev_ckpt}\n  asked {this_ckpt}"
            )

    if args.all_slides:
        split_all = pd.read_csv(args.split, dtype={"image_id": str})
        wanted = sorted(split_all["image_id"].astype(str).unique())
    else:
        manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
        wanted = slides_needing_cache(manifest)
    if args.max_slides:
        wanted = wanted[: args.max_slides]

    if not args.overwrite:
        todo = [s for s in wanted if not cache_path(args.out_dir, s).exists()]
    else:
        todo = list(wanted)

    if rank == 0:
        print(f"Slides in scope: {len(wanted)}")
        print(f"Already cached: {len(wanted) - len(todo)} | to compute: {len(todo)}")
    if not todo:
        if rank == 0:
            print("Nothing to do.")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    split_df = pd.read_csv(args.split, dtype={"image_id": str})
    split_df = split_df[split_df["image_id"].isin(set(todo))].copy()
    split_df = split_df.sort_values(["image_id", "y", "x"]).reset_index(drop=True)

    slide_ids = sorted(split_df["image_id"].unique())
    slide_to_index = {sid: i for i, sid in enumerate(slide_ids)}
    index_to_slide = {i: sid for sid, i in slide_to_index.items()}
    expected_counts = split_df.groupby("image_id").size().to_dict()

    dataset = PatchInferenceDataset(args.split, split_df, slide_to_index, mode=args.mode)
    sampler = RankSlideSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model, resolved_arch, ckpt = load_source_model(
        args.checkpoint, args.arch, device, verbose=rank == 0
    )
    if rank == 0:
        _write_pack_sidecar(
            args.out_dir,
            checkpoint=args.checkpoint,
            ckpt=ckpt,
            args=args,
            n_slides=len(wanted),
            n_patches=int(split_df.shape[0]) if args.all_slides else int(len(dataset)),
            status="running",
        )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    if rank == 0:
        print(
            f"Caching preds+maxprob={args.write_maxprob}: slides={len(slide_ids)} "
            f"patches={len(dataset)} world={world_size} arch={resolved_arch}"
        )
        print(f"Source checkpoint: {args.checkpoint}")
        print(f"Output: {args.out_dir}")

    buffers: dict[int, dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None]]] = {}
    written = 0

    with torch.inference_mode():
        for step, (images, slide_indices, xs, ys) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            n_real = images.shape[0]
            if n_real < args.batch_size:
                pad = images[-1:].expand(args.batch_size - n_real, *images.shape[1:])
                images = torch.cat([images, pad], dim=0)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
            logits_f = logits[:n_real].float()
            pred = logits_f.argmax(dim=1).to(torch.uint8).cpu().numpy()
            maxp = None
            if args.write_maxprob:
                maxp = torch.softmax(logits_f, dim=1).amax(dim=1).cpu().numpy().astype(np.float16)

            for i, slide_index in enumerate(slide_indices.tolist()):
                buf = buffers.setdefault(slide_index, {})
                buf[(int(xs[i]), int(ys[i]))] = (pred[i], None if maxp is None else maxp[i])
                slide_id = index_to_slide[slide_index]
                if len(buf) == expected_counts[slide_id]:
                    coord_list = sorted(buf.keys())
                    coords = np.asarray(coord_list, dtype=np.int32)
                    stacked = np.stack([buf[c][0] for c in coord_list], axis=0)
                    stacked_p = None
                    if args.write_maxprob:
                        stacked_p = np.stack([buf[c][1] for c in coord_list], axis=0)
                    write_slide_cache(args.out_dir, slide_id, coords, stacked, stacked_p)
                    del buffers[slide_index]
                    written += 1

            if rank == 0 and step % 100 == 0:
                print(
                    f"rank0 batches={step}/{len(loader)} slides_written={written}",
                    flush=True,
                )

    if buffers:
        raise RuntimeError(
            f"rank{rank}: {len(buffers)} slide(s) incomplete — patch counts disagree with split CSV"
        )

    if dist.is_available() and dist.is_initialized():
        counts = torch.tensor([written], device=device if dist.get_backend() == "nccl" else "cpu")
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        written = int(counts.item())

    if rank == 0:
        _write_pack_sidecar(
            args.out_dir,
            checkpoint=args.checkpoint,
            ckpt=ckpt,
            args=args,
            n_slides=len(wanted),
            n_patches=int(pd.read_csv(args.split).shape[0]) if args.all_slides else int(len(wanted)),
            status="complete" if not todo or written >= 0 else "running",
        )
        n_done = sum(1 for s in wanted if cache_path(args.out_dir, s).exists())
        if n_done >= len(wanted):
            cfg = json.loads((args.out_dir / "pack_config.json").read_text(encoding="utf-8"))
            cfg["status"] = "complete"
            (args.out_dir / "pack_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"\nDONE -- wrote {written} slide cache files to {args.out_dir} ({n_done}/{len(wanted)} on disk)")
        sample = cache_path(args.out_dir, slide_ids[0])
        if sample.exists():
            with h5py.File(sample, "r") as f:
                keys = list(f.keys())
                extra = f" maxprob={f['maxprob'].shape}" if "maxprob" in f else ""
                print(
                    f"Sample {sample.name}: coords={f['coords'].shape} "
                    f"preds={f['preds'].shape}{extra} keys={keys} "
                    f"on-disk={sample.stat().st_size / 1e6:.1f} MB"
                )
            print(f"(uncompressed preds would be {expected_counts[slide_ids[0]] * PATCH_SIZE ** 2 / 1e6:.1f} MB)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
