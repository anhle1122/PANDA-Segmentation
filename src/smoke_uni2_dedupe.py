#!/usr/bin/env python3
"""Smoke test: can UNI2 embeddings find the duplicates shape and colour missed?

Small on purpose -- a few hundred slides, not all 4683 -- so we learn whether
the signal exists before paying for the full run. The set is built to be hard:
the user's serial-cut pairs, a sample of confirmed twins, a sample of confirmed
not-twins, and random distractors so specificity is actually tested.

The decisive metric is not AUC. It is top-1 retrieval: for a slide whose twin is
in the set, is its nearest neighbour that twin? That is what grouping needs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from patch_utils import PROJECT, read_rgb_patch

G = PROJECT / "outputs" / "docs" / "slide_groups"
DUP = PROJECT / "outputs" / "docs" / "slide_duplicates"
SPLITS = PROJECT / "outputs" / "splits"
SLIDES = PROJECT / "data" / "slides"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def slide_path(sid: str):
    hits = list(SLIDES.glob(f"{sid[:12]}*.tiff"))
    return hits[0] if hits else None


def load_patch_index() -> pd.DataFrame:
    """Pre-dedupe CSVs so restored slides still have their patch coordinates."""
    frames = []
    for split in ("train", "val", "test"):
        for name in (f"panda_{split}_pre_dedupe.csv", f"panda_{split}.csv"):
            p = SPLITS / name
            if p.exists():
                frames.append(pd.read_csv(p, usecols=["image_id", "x", "y"], dtype={"image_id": str}))
                break
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def build_set(n_pairs: int, n_random: int, seed: int) -> tuple[list[str], list[tuple], list[tuple]]:
    rng = np.random.default_rng(seed)
    hard = pd.read_csv(G / "benchmark_hard_positives.csv", dtype=str)
    hard_pairs = [(r.id_a, r.id_b) for r in hard.itertuples(index=False)]

    conf = pd.read_csv(DUP / "galleries" / "confirmed_twins_keep_drop" / "all_confirmed_twins_keep_drop.csv", dtype=str)
    conf = conf.sample(min(n_pairs, len(conf)), random_state=seed)
    pos = hard_pairs + [(str(r.keep_id), str(r.drop_id)) for r in conf.itertuples(index=False)]

    negs = pd.read_csv(DUP / "CANONICAL_not_twin_pairs.csv", dtype=str)
    negs = negs.sample(min(n_pairs, len(negs)), random_state=seed)
    neg = [(r.id_a, r.id_b) for r in negs.itertuples(index=False)]

    meta = pd.read_csv(DUP / "fingerprint_meta.csv", dtype={"image_id": str})
    used = {x for p in pos + neg for x in p}
    pool = [i for i in meta.image_id if i not in used]
    distract = list(rng.choice(pool, size=min(n_random, len(pool)), replace=False))
    return sorted(used | set(distract)), pos, neg


@torch.no_grad()
def embed(ids: list[str], patches: pd.DataFrame, n_patch: int, device: str, seed: int) -> tuple[list[str], np.ndarray]:
    from train.uni2_upernet import UNI2UPerNet

    print("loading UNI2-h ...", flush=True)
    net = UNI2UPerNet._load_backbone(img_size=224, pretrained=True, checkpoint_path=None)
    net = net.to(device).eval()

    by_slide = {k: v for k, v in patches.groupby("image_id")}
    rng = np.random.default_rng(seed)
    out_ids, out_vec = [], []

    for sid in tqdm(ids, desc="slides"):
        rows = by_slide.get(sid)
        path = slide_path(sid)
        if rows is None or path is None or len(rows) == 0:
            continue
        take = rows.sample(min(n_patch, len(rows)), random_state=int(rng.integers(1 << 30)))
        imgs = []
        for r in take.itertuples(index=False):
            try:
                p = read_rgb_patch(path, int(r.x), int(r.y))
            except Exception:
                continue
            im = np.asarray(Image.fromarray(p).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            imgs.append((im - MEAN) / STD)
        if not imgs:
            continue
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device, torch.float16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            f = net.forward_features(x)
        if f.ndim == 3:
            f = f[:, 1:].mean(1)  # drop cls/reg prefix, average patch tokens
        v = f.float().mean(0)
        out_ids.append(sid)
        out_vec.append(torch.nn.functional.normalize(v, dim=0).cpu().numpy())

    return out_ids, np.stack(out_vec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--n-patch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids, pos, neg = build_set(args.n_pairs, args.n_random, args.seed)
    print(f"smoke set: {len(ids)} slides | {len(pos)} twin pairs | {len(neg)} not-twin pairs", flush=True)

    ids, vecs = embed(ids, load_patch_index(), args.n_patch, "cuda", args.seed)
    idx = {s: i for i, s in enumerate(ids)}
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    np.savez_compressed(G / "smoke_uni2.npz", ids=np.array(ids), sim=sim)

    have = lambda ps: [(a, b) for a, b in ps if a in idx and b in idx]
    pos, neg = have(pos), have(neg)
    pv = np.array([sim[idx[a], idx[b]] for a, b in pos])
    nv = np.array([sim[idx[a], idx[b]] for a, b in neg])
    auc = float((pv[:, None] > nv[None, :]).mean())

    top1 = sum(1 for a, b in pos if ids[int(sim[idx[a]].argmax())] == b or ids[int(sim[idx[b]].argmax())] == a)
    print(f"\nslides embedded: {len(ids)}")
    print(f"twin cosine   : median {np.median(pv):.4f}  min {pv.min():.4f}")
    print(f"nottwin cosine: median {np.median(nv):.4f}  max {nv.max():.4f}")
    print(f"AUC twins vs not-twins: {auc:.3f}")
    print(f"TOP-1 retrieval: {top1}/{len(pos)} twin pairs are each other's nearest neighbour")

    hard = pd.read_csv(G / "benchmark_hard_positives.csv", dtype=str)
    print("\nyour serial-cut pairs:")
    for r in hard.itertuples(index=False):
        if r.id_a not in idx or r.id_b not in idx:
            print(f"  {r.id_a[:8]}+{r.id_b[:8]}  (missing from set)")
            continue
        s = sim[idx[r.id_a], idx[r.id_b]]
        rank = int((sim[idx[r.id_a]] > s).sum()) + 1
        print(f"  {r.id_a[:8]}+{r.id_b[:8]}  cos={s:.4f}  rank {rank} of {len(ids)-1} "
              f"({'TOP-1' if rank == 1 else 'miss'})")

    (G / "smoke_uni2_report.json").write_text(
        json.dumps({"n_slides": len(ids), "auc": auc, "top1": top1, "n_pos": len(pos)}, indent=2)
    )


if __name__ == "__main__":
    main()
