#!/usr/bin/env python3
"""PANDA+ Dice split: confirmed visual twins vs the other 30 slides.

Does not retrain. Same protocol as evaluate.py --panda-plus-eval
(min_target_class=2, mean G3/G4/G5). Shuffle is off so slide IDs stay aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import (  # noqa: E402
    EvalAccumulator,
    _opt3_flags,
    _peek_state_dict,
    build_eval_model,
    load_model_weights,
)
from patch_utils import PROJECT  # noqa: E402
from train.baseline_dataset import BaselinePatchDataset  # noqa: E402

DUP_CSV = (
    PROJECT
    / "outputs"
    / "docs"
    / "slide_groups"
    / "panda_plus_lookalike_gallery"
    / "panda_plus_confirmed_duplicates.csv"
)
PLUS_PATCHES = PROJECT / "outputs" / "panda_plus" / "panda_plus_patches.csv"
PLUS_MASKS = PROJECT / "outputs" / "panda_plus" / "masks"


class WithSlideId(Dataset):
    def __init__(self, ds: BaselinePatchDataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, mask, _weight = self.ds[idx]
        sid = str(self.ds.df.iloc[idx]["image_id"])
        return image, mask, sid


def collate(batch):
    images, masks, sids = zip(*batch)
    return torch.stack(images, 0), torch.stack(masks, 0), list(sids)


def load_groups() -> dict[str, set[str]]:
    leaked: set[str] = set()
    train_twin: set[str] = set()
    with DUP_CSV.open() as f:
        for row in csv.DictReader(f):
            plus = row["plus_id"]
            leaked.add(plus)
            if row["live_split"] == "train":
                train_twin.add(plus)
    patches = pd.read_csv(PLUS_PATCHES, usecols=["image_id"])
    all_ids = set(patches.image_id.astype(str))
    return {
        "all": all_ids,
        "leaked": leaked,
        "clean": all_ids - leaked,
        "train_twin": train_twin,
        "valtest_twin": leaked - train_twin,
    }


def write_labeled_csv(path: Path, acc: EvalAccumulator) -> None:
    """Match evaluate.py --panda-plus-eval CSV so summarize_epoch_eval can ingest it."""
    per = acc.per_class_metrics()
    summ = acc.summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "dice", "iou", "precision", "recall"])
        w.writeheader()
        for name in ("benign", "G3", "G4", "G5"):
            m = per[name]
            w.writerow(
                {
                    "class": name,
                    "dice": f"{m['dice']:.6f}",
                    "iou": f"{m['iou']:.6f}",
                    "precision": f"{m['precision']:.6f}",
                    "recall": f"{m['recall']:.6f}",
                }
            )
        w.writerow({"class": "mean_dice", "dice": f"{summ['mean_dice']:.6f}", "iou": "", "precision": "", "recall": ""})
        w.writerow({"class": "cancer_dice", "dice": f"{summ['cancer_dice']:.6f}", "iou": "", "precision": "", "recall": ""})
        w.writerow({"class": "cancer_recall", "dice": "", "iou": "", "precision": "", "recall": f"{summ['cancer_recall']:.6f}"})
        w.writerow({"class": "g5_recall", "dice": "", "iou": "", "precision": "", "recall": f"{summ['g5_recall']:.6f}"})


def acc_to_row(name: str, n_slides: int, acc: EvalAccumulator) -> dict:
    per = acc.per_class_metrics()
    summ = acc.summary()
    return {
        "group": name,
        "n_slides": n_slides,
        "cancer_dice": summ["cancer_dice"],
        "mean_dice": summ["mean_dice"],
        "G3_dice": per["G3"]["dice"],
        "G4_dice": per["G4"]["dice"],
        "G5_dice": per["G5"]["dice"],
        "benign_dice": per["benign"]["dice"],
        "cancer_recall": summ["cancer_recall"],
        "g5_recall": summ["g5_recall"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--dice-csv",
        type=Path,
        default=None,
        help="Write evaluate.py-style PANDA+ labeled Dice CSV for the all-48 group.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: no CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    groups = load_groups()
    print(
        f"groups leaked={len(groups['leaked'])} clean={len(groups['clean'])} "
        f"train_twin={len(groups['train_twin'])} valtest_twin={len(groups['valtest_twin'])}",
        flush=True,
    )

    base = BaselinePatchDataset(
        PLUS_PATCHES,
        mode="raw",
        allow_missing_h5=True,
        mask_dir=PLUS_MASKS,
        mask_suffix="_pandaplus_mask.png",
        prefer_h5_masks=False,
    )
    ds = WithSlideId(base)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )

    _, peek_keys = _peek_state_dict(args.checkpoint)
    _is_opt3, decode_norm, use_lora = _opt3_flags(peek_keys)
    model = build_eval_model("uni2_upernet", decode_norm=decode_norm, use_lora=use_lora)
    ckpt_meta = load_model_weights(args.checkpoint, model)
    print(f"Opt3 load: decode_norm={decode_norm} use_lora={use_lora}", flush=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=args.amp):
            _ = model(torch.zeros(2, 3, 512, 512, device=device))

    buckets = {
        "all": EvalAccumulator(min_target_class=2),
        "leaked": EvalAccumulator(min_target_class=2),
        "clean": EvalAccumulator(min_target_class=2),
        "train_twin": EvalAccumulator(min_target_class=2),
        "valtest_twin": EvalAccumulator(min_target_class=2),
    }
    per_slide: dict[str, EvalAccumulator] = {}

    n_seen = 0
    with torch.no_grad():
        for images, masks, sids in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                preds = model(images).argmax(dim=1)
            for i, sid in enumerate(sids):
                p = preds[i]
                m = masks[i]
                buckets["all"].update(p, m)
                if sid in groups["leaked"]:
                    buckets["leaked"].update(p, m)
                else:
                    buckets["clean"].update(p, m)
                if sid in groups["train_twin"]:
                    buckets["train_twin"].update(p, m)
                if sid in groups["valtest_twin"]:
                    buckets["valtest_twin"].update(p, m)
                if sid not in per_slide:
                    per_slide[sid] = EvalAccumulator(min_target_class=2)
                per_slide[sid].update(p, m)
            n_seen += len(sids)
            if n_seen % 400 == 0:
                print(f"patches {n_seen}/{len(ds)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    group_n = {
        "all": len(groups["all"]),
        "leaked": len(groups["leaked"]),
        "clean": len(groups["clean"]),
        "train_twin": len(groups["train_twin"]),
        "valtest_twin": len(groups["valtest_twin"]),
    }
    rows = [acc_to_row(k, group_n[k], buckets[k]) for k in group_n]
    leaked_d = rows[1]["cancer_dice"]
    clean_d = rows[2]["cancer_dice"]
    all_d = rows[0]["cancer_dice"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(ckpt_meta.get("epoch", -1)),
        "n_patches": n_seen,
        "groups": rows,
        "gap_leaked_minus_clean": leaked_d - clean_d,
        "headline_cancer_dice": all_d,
        "clean_only_cancer_dice": clean_d,
        "headline_minus_clean": all_d - clean_d,
        "note": "clean_only is Dice on the 30 non-twin PANDA+ slides. headline_minus_clean is how much the 18 twins pull the reported number.",
    }
    out_json = args.out_dir / "panda_plus_leak_split.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.out_dir / "panda_plus_leak_split_groups.csv", index=False)
    slide_rows = []
    for sid, acc in sorted(per_slide.items()):
        r = acc_to_row(sid, 1, acc)
        r["slide_id"] = sid
        r["bucket"] = (
            "train_twin"
            if sid in groups["train_twin"]
            else ("valtest_twin" if sid in groups["valtest_twin"] else "clean")
        )
        slide_rows.append(r)
    pd.DataFrame(slide_rows).to_csv(args.out_dir / "panda_plus_leak_split_per_slide.csv", index=False)
    if args.dice_csv is not None:
        write_labeled_csv(args.dice_csv, buckets["all"])
        print(f"Wrote {args.dice_csv}", flush=True)
        from evaluate import confusion_df
        conf_path = args.dice_csv.with_name(args.dice_csv.stem + "_confusion.csv")
        confusion_df(buckets["all"]).to_csv(conf_path, float_format="%.0f")
        print(f"Wrote {conf_path}", flush=True)
    print(json.dumps({k: summary[k] for k in ("headline_cancer_dice", "clean_only_cancer_dice", "gap_leaked_minus_clean", "headline_minus_clean")}, indent=2))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
