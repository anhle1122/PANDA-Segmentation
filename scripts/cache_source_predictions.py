"""Cache a source model's per-pixel predicted classes for pseudo-label targets.

Rules 1-3 all decide per pixel by asking "what class did the SOURCE model
predict here?" -- which pixels are over-extended (Rule 1 wide margin), which
are the invented class (Rule 2), which sit outside the expected set (Rule 3).
That answer is fixed for a whole round, so we compute it once here instead of
re-running the source model inside the training loop.

Only slides that actually receive a correction are cached (rule_applied not in
{match, none}), which is a small fraction of the split.

Output: one HDF5 per slide, mirroring the existing patch-H5 convention
(``coords`` + a payload dataset) so file counts stay low on the shared
filesystem:

    <out_dir>/<slide_id>_srcpred.h5
        coords  (N, 2) int32   -- patch (x, y), same grid as the split CSV
        preds   (N, 512, 512)  uint8, gzip -- argmax class per pixel

Re-running skips slides already written, so an interrupted job can resume.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
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

from evaluate import build_eval_model, detect_arch  # noqa: E402
from patch_utils import PATCH_SIZE, PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset, _preprocess_image  # noqa: E402
from train.pseudo_label_rules import MATCH, NO_ACTION, NO_CORRECTION_RULES  # noqa: E402
from train.uni2_upernet import (  # noqa: E402
    checkpoint_has_bn_running_stats,
    disable_bn_running_stats,
)

DEFAULT_SPLIT = PROJECT / "outputs" / "splits" / "panda_train.csv"
DEFAULT_MANIFEST = PROJECT / "outputs" / "pseudo_label" / "round1_rule_manifest.csv"
DEFAULT_OUT_DIR = PROJECT / "outputs" / "pseudo_label" / "round1_source_pred"

CACHE_SUFFIX = "_srcpred.h5"


def cache_path(out_dir: Path, slide_id: str) -> Path:
    return out_dir / f"{slide_id}{CACHE_SUFFIX}"


def slides_needing_cache(manifest: pd.DataFrame) -> list[str]:
    """Slide IDs whose pseudo-label correction needs per-pixel source predictions."""
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
    """Image-only view that also reports which slide/coord each patch came from."""

    def __init__(self, split_df: pd.DataFrame, slide_to_index: dict[str, int], *, mode: str) -> None:
        self.base = BaselinePatchDataset(DEFAULT_SPLIT, mode=mode)
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
    """Shard WHOLE slides across ranks so each slide's H5 is written by one rank."""

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
    """Build the source model so it matches how the checkpoint was actually trained.

    Strict loading is kept deliberately: a silent partial load here would
    poison every pseudo-label target downstream.
    """
    resolved_arch = detect_arch(checkpoint, arch)
    model = build_eval_model(resolved_arch)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if not checkpoint_has_bn_running_stats(state_dict):
        n = disable_bn_running_stats(model)
        if verbose:
            print(
                f"Checkpoint has no BatchNorm running stats -> switched {n} BatchNorm "
                "module(s) to batch-statistics mode to match training."
            )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, resolved_arch, ckpt


def write_slide_cache(out_dir: Path, slide_id: str, coords: np.ndarray, preds: np.ndarray) -> None:
    """Write one slide's cache atomically (tmp + rename) so resume never sees a partial file."""
    final = cache_path(out_dir, slide_id)
    tmp = final.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords.astype(np.int32))
        f.create_dataset("preds", data=preds.astype(np.uint8), compression="gzip", compression_opts=4)
    tmp.replace(final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache source-model per-pixel predictions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", default="raw", choices=("raw", "normalized", "normalized_ink_raw"))
    parser.add_argument("--arch", default="auto", choices=("auto", "baseline", "uni2_upernet"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-slides", type=int, default=None, help="Smoke test only")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cache files")
    args = parser.parse_args()

    _local_rank, rank, world_size, device = setup_distributed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    wanted = slides_needing_cache(manifest)
    if args.max_slides:
        wanted = wanted[: args.max_slides]

    if not args.overwrite:
        todo = [s for s in wanted if not cache_path(args.out_dir, s).exists()]
    else:
        todo = list(wanted)

    if rank == 0:
        print(f"Slides needing correction: {len(wanted)}")
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

    dataset = PatchInferenceDataset(split_df, slide_to_index, mode=args.mode)
    sampler = RankSlideSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model, resolved_arch, _ = load_source_model(
        args.checkpoint, args.arch, device, verbose=rank == 0
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    if rank == 0:
        print(
            f"Caching predictions: slides={len(slide_ids)} patches={len(dataset)} "
            f"world={world_size} arch={resolved_arch}"
        )
        print(f"Source checkpoint: {args.checkpoint}")
        print(f"Output: {args.out_dir}")

    # Buffer per slide; flush as soon as a slide has all its patches, so peak
    # memory stays at a few slides rather than the whole shard.
    buffers: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    written = 0

    with torch.inference_mode():
        for step, (images, slide_indices, xs, ys) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)

            # This checkpoint's BatchNorms run on batch statistics (no running
            # stats were saved), and the PPM's global-pooling branch is 1x1
            # spatially -- so a batch of 1 has a single value per channel and
            # BatchNorm raises. Padding every short batch back up to the full
            # batch size both avoids that and keeps the batch statistics
            # computed over a consistent sample count for every patch.
            n_real = images.shape[0]
            if n_real < args.batch_size:
                pad = images[-1:].expand(args.batch_size - n_real, *images.shape[1:])
                images = torch.cat([images, pad], dim=0)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                preds = model(images).argmax(dim=1)
            preds_np = preds[:n_real].to(torch.uint8).cpu().numpy()

            for i, slide_index in enumerate(slide_indices.tolist()):
                buf = buffers.setdefault(slide_index, {})
                buf[(int(xs[i]), int(ys[i]))] = preds_np[i]
                slide_id = index_to_slide[slide_index]
                if len(buf) == expected_counts[slide_id]:
                    coord_list = sorted(buf.keys())
                    coords = np.asarray(coord_list, dtype=np.int32)
                    stacked = np.stack([buf[c] for c in coord_list], axis=0)
                    write_slide_cache(args.out_dir, slide_id, coords, stacked)
                    del buffers[slide_index]
                    written += 1

            if rank == 0 and step % 100 == 0:
                print(
                    f"rank0 batches={step}/{len(loader)} slides_written={written}",
                    flush=True,
                )

    if buffers:
        # Should not happen: every slide's patches are in this rank's shard.
        raise RuntimeError(
            f"rank{rank}: {len(buffers)} slide(s) incomplete at end of pass -- "
            "patch counts disagree with the split CSV"
        )

    if dist.is_available() and dist.is_initialized():
        counts = torch.tensor([written], device=device if dist.get_backend() == "nccl" else "cpu")
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        written = int(counts.item())

    if rank == 0:
        print(f"\nDONE -- wrote {written} slide cache files to {args.out_dir}")
        sample = cache_path(args.out_dir, slide_ids[0])
        if sample.exists():
            with h5py.File(sample, "r") as f:
                print(
                    f"Sample {sample.name}: coords={f['coords'].shape} "
                    f"preds={f['preds'].shape} dtype={f['preds'].dtype} "
                    f"on-disk={sample.stat().st_size / 1e6:.1f} MB"
                )
            print(f"(uncompressed would be {expected_counts[slide_ids[0]] * PATCH_SIZE ** 2 / 1e6:.1f} MB)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
