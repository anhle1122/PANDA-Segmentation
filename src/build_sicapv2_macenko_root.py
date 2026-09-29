"""Build a SICAPv2-like root whose images/ are Macenko-normalized to a PANDA ref.

Masks + partition are symlinked from the official extract. Only unique Test.xlsx
stems across Val folds are written (eval reads those only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from stain_normalize import get_normalizer, load_reference_patch, tissue_content_fraction
from stain_norm_sicap import PANDA_REFERENCE, find_sicap_image_dir, read_sicap_patch

DEFAULT_OFFICIAL = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")


def collect_test_stems(official: Path, folds: list[int]) -> list[str]:
    stems: set[str] = set()
    for fold in folds:
        path = official / "partition" / "Validation" / f"Val{fold}" / "Test.xlsx"
        df = pd.read_excel(path)
        col = "image_name" if "image_name" in df.columns else df.columns[0]
        for name in df[col].tolist():
            stems.add(Path(str(name)).stem)
    return sorted(stems)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--folds", type=str, default="1,2,3,4")
    p.add_argument("--method", type=str, default="macenko", choices=("macenko", "vahadane"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    official = args.official_root
    out = args.out_root
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Symlink masks + partition (always refresh)
    for name in ("masks", "partition"):
        src = official / name
        dst = out / name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                raise SystemExit(f"{dst} exists and is not a symlink; refuse to clobber")
        dst.symlink_to(src)

    stems = collect_test_stems(official, folds)
    src_dir = find_sicap_image_dir(official / "images")
    print(f"stems={len(stems)} method={args.method} src={src_dir} out={img_out}")

    ref = load_reference_patch(PANDA_REFERENCE)
    print(f"ref={PANDA_REFERENCE} tissue={tissue_content_fraction(ref):.3f}")
    norm, backend = get_normalizer(args.method)
    norm.fit(ref)
    print(f"fitted {args.method} backend={backend}")

    meta = {
        "method": args.method,
        "backend": backend,
        "panda_reference": list(PANDA_REFERENCE),
        "n_stems": len(stems),
        "folds": folds,
        "official_root": str(official),
    }
    (out / "stain_norm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Image.fromarray(ref).save(out / "panda_reference.jpg", quality=92)

    n_ok = n_skip = n_err = n_cached = 0
    for stem in tqdm(stems, desc=args.method):
        dest = img_out / f"{stem}.jpg"
        if dest.is_file() and not args.force:
            n_cached += 1
            continue
        src = src_dir / f"{stem}.jpg"
        if not src.is_file():
            n_err += 1
            continue
        before = read_sicap_patch(src)
        tissue = tissue_content_fraction(before)
        try:
            after = norm.transform(before)
            n_ok += 1
        except Exception:
            after = before
            n_err += 1
        if tissue < 0.05:
            n_skip += 1  # still write (transform may no-op / fail)
        bgr = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(
        f"done ok={n_ok} cached={n_cached} err={n_err} low_tissue_note={n_skip} "
        f"images={sum(1 for _ in img_out.glob('*.jpg'))}"
    )


if __name__ == "__main__":
    main()
