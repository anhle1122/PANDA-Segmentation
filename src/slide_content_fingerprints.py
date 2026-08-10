#!/usr/bin/env python3
"""Content fingerprints for duplicate detection, scored on the review benchmark.

Shape silhouette IoU ranks true twins at the bottom of the list (P1828-P1833 in
scan3 are genuine serial cuts at IoU 0.30), so it cannot drive the split. These
descriptors look at tissue content instead, and each is scored against the pairs
the user has already adjudicated so we pick a threshold from measured recall
rather than by eye.

Descriptors, cheapest first:
  phash      64-bit DCT hash of the grey thumbnail, min Hamming over 8
             flips/rotations. Catches re-scans; sensitive to how fragments sit.
  colorhist  8x8x8 RGB histogram over tissue pixels only. Ignores layout
             entirely, so a core that broke into three pieces still matches.
  odstats    mean/std optical density per channel -- weak alone, cheap to add.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dedupe_slides_shape_isup import META_PATH
from patch_utils import PROJECT

OUT = PROJECT / "outputs" / "docs" / "slide_groups"
SLIDES = PROJECT / "data" / "slides"
THUMB = 256
HASH_SIDE = 32
HASH_KEEP = 8  # low-frequency 8x8 block of the DCT -> 64 bits


def _dct2(a: np.ndarray) -> np.ndarray:
    from scipy.fft import dct

    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def _phash_bits(gray: np.ndarray) -> np.uint64:
    d = _dct2(gray.astype(np.float32))[:HASH_KEEP, :HASH_KEEP].flatten()
    bits = d > np.median(d[1:])  # skip DC, it only encodes brightness
    return np.packbits(bits).view(np.uint64)[0]


ORIENTS = (
    lambda a: a,
    np.fliplr,
    np.flipud,
    lambda a: np.fliplr(np.flipud(a)),
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: np.fliplr(np.rot90(a, 1)),
)


def describe(sid: str) -> dict | None:
    import openslide

    paths = list(SLIDES.glob(f"{sid[:12]}*.tiff"))
    if not paths:
        return None
    try:
        sl = openslide.OpenSlide(str(paths[0]))
        im = sl.get_thumbnail((THUMB, THUMB)).convert("RGB")
        sl.close()
    except Exception:
        return None

    rgb = np.asarray(im, dtype=np.uint8)
    tissue = rgb.reshape(-1, 3)[(rgb.reshape(-1, 3) < 220).any(1)]
    if len(tissue) < 50:
        return None

    sq = np.asarray(im.convert("L").resize((HASH_SIDE, HASH_SIDE), Image.BILINEAR), dtype=np.float32)
    hashes = np.array([_phash_bits(f(sq)) for f in ORIENTS], dtype=np.uint64)

    q = tissue // 32
    hist = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(np.float32)
    hist /= hist.sum()

    od = -np.log10(np.clip(tissue.astype(np.float32), 1, 255) / 255.0)
    return {
        "image_id": sid,
        "hashes": hashes,
        "hist": hist,
        "od": np.concatenate([od.mean(0), od.std(0)]).astype(np.float32),
    }


def extract(ids: list[str], workers: int) -> dict:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(describe, ids, chunksize=8), total=len(ids), desc="thumbnails"):
            if r is not None:
                rows.append(r)
    print(f"described {len(rows)}/{len(ids)} slides", flush=True)
    return {
        "ids": np.array([r["image_id"] for r in rows]),
        "hashes": np.stack([r["hashes"] for r in rows]),
        "hist": np.stack([r["hist"] for r in rows]),
        "od": np.stack([r["od"] for r in rows]),
    }


def hamming_min(hashes: np.ndarray) -> np.ndarray:
    """Best (lowest) Hamming distance over orientations, for every pair."""
    n = len(hashes)
    canon = hashes[:, 0]
    best = np.full((n, n), 64, dtype=np.uint8)
    for o in range(hashes.shape[1]):
        x = np.bitwise_xor(canon[:, None], hashes[None, :, o])
        d = np.zeros_like(x, dtype=np.uint8)
        for _ in range(64):
            d += (x & np.uint64(1)).astype(np.uint8)
            x >>= np.uint64(1)
        np.minimum(best, d, out=best)
    return np.minimum(best, best.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_PATH)
    ids = meta.image_id.astype(str).tolist()

    cache = OUT / "content_fingerprints.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        data = {k: d[k] for k in d.files}
        print(f"loaded cached fingerprints for {len(data['ids'])} slides", flush=True)
    else:
        data = extract(ids, args.workers)
        np.savez_compressed(cache, **data)
        print(f"wrote {cache}", flush=True)

    sids = list(data["ids"])
    pos = {i: n for n, i in enumerate(sids)}

    print("scoring pairwise...", flush=True)
    ham = hamming_min(data["hashes"])
    phash_sim = 1.0 - ham.astype(np.float32) / 64.0
    hist_sim = np.minimum(data["hist"][:, None, :], data["hist"][None, :, :]).sum(-1)
    np.fill_diagonal(phash_sim, 0.0)
    np.fill_diagonal(hist_sim, 0.0)

    np.savez_compressed(OUT / "content_similarity.npz", ids=data["ids"], phash=phash_sim, hist=hist_sim)
    print(json.dumps({"n": len(sids), "out": str(OUT / 'content_similarity.npz')}, indent=2), flush=True)


if __name__ == "__main__":
    main()
