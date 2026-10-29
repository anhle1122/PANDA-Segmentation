"""Organize AGGC2022 Subset1/Subset2 on HPC and report what is missing.

Expected slides come from the official annotation zips. Images are taken from
any zip or loose .tiff dropped in the incoming folder (OneDrive_N.zip parts from
the laptop included). Nothing is deleted; files are moved into:

  <dest>/Subset{1,2}/{Train,Test}/images/<slide>.tiff
  <dest>/Subset{1,2}/{Train,Test}/annotations/<slide>/<Class>_Mask.tif
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

ANNOT_DIR = Path("/common/omarmlab/members/anh/panda_data/downloads/aggc2")
DEST = Path("/common/omarmlab/data/AGGC_2022")
SLIDE_RE = re.compile(r"(Subset([12])_(Train|Test)_\d+)\.tiff?$", re.IGNORECASE)


def expected_slides() -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for sub in ("1", "2"):
        for split in ("Train", "Test"):
            z = ANNOT_DIR / f"Subset{sub}_{split}_annotation.zip"
            with zipfile.ZipFile(z) as zf:
                names = {
                    p.split("/")[1]
                    for p in zf.namelist()
                    if p.count("/") >= 2 and p.split("/")[1].startswith(f"Subset{sub}_")
                }
            out[(sub, split)] = sorted(names, key=lambda s: int(s.rsplit("_", 1)[1]))
    return out


def extract_annotations(dest: Path) -> None:
    for sub in ("1", "2"):
        for split in ("Train", "Test"):
            target = dest / f"Subset{sub}" / split / "annotations"
            if target.is_dir() and any(target.iterdir()):
                continue
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ANNOT_DIR / f"Subset{sub}_{split}_annotation.zip") as zf:
                for info in zf.infolist():
                    parts = info.filename.split("/")
                    if info.is_dir() or len(parts) < 3 or parts[-1] == "Thumbs.db":
                        continue
                    out = target / parts[1] / parts[-1]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 << 20)


def gather_images(incoming: Path, dest: Path, apply: bool) -> list[dict]:
    log = []
    for z in sorted(incoming.rglob("*.zip")):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            log.append({"source": str(z), "status": "bad_zip (incomplete download?)"})
            continue
        with zf:
            for info in zf.infolist():
                if info.filename.startswith("__MACOSX") or "/__" in info.filename:
                    continue
                m = SLIDE_RE.search(info.filename)
                if not m:
                    continue
                slide, sub, split = m.group(1), m.group(2), m.group(3).capitalize()
                out = dest / f"Subset{sub}" / split / "images" / f"{slide}.tiff"
                rec = {"slide": slide, "source": f"{z}:{info.filename}", "bytes": info.file_size}
                if out.exists() and out.stat().st_size == info.file_size:
                    rec["status"] = "already_present"
                elif apply:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".tiff.part")
                    with zf.open(info) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=64 << 20)
                    tmp.rename(out)
                    rec["status"] = "extracted"
                else:
                    rec["status"] = "would_extract"
                log.append(rec)
    for t in sorted(list(incoming.rglob("*.tiff")) + list(incoming.rglob("*.tif"))):
        m = SLIDE_RE.search(t.name)
        if not m:
            continue
        slide, sub, split = m.group(1), m.group(2), m.group(3).capitalize()
        out = dest / f"Subset{sub}" / split / "images" / f"{slide}.tiff"
        rec = {"slide": slide, "source": str(t), "bytes": t.stat().st_size}
        if out.exists():
            rec["status"] = "already_present"
        elif apply:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(t), out)
            rec["status"] = "moved"
        else:
            rec["status"] = "would_move"
        log.append(rec)
    return log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--incoming", type=Path, default=DEST / "_incoming")
    p.add_argument("--dest", type=Path, default=DEST)
    p.add_argument("--apply", action="store_true", help="Extract/move files. Default only reports.")
    args = p.parse_args()

    exp = expected_slides()
    if args.apply:
        extract_annotations(args.dest)
    log = gather_images(args.incoming, args.dest, args.apply) if args.incoming.exists() else []

    report = {"expected": {}, "missing_images": {}, "missing_annotations": {}, "sources": log}
    for (sub, split), slides in exp.items():
        key = f"Subset{sub}_{split}"
        img_dir = args.dest / f"Subset{sub}" / split / "images"
        ann_dir = args.dest / f"Subset{sub}" / split / "annotations"
        have_img = {f.stem for f in img_dir.glob("*.tiff")} if img_dir.is_dir() else set()
        if not args.apply:
            have_img |= {
                r["slide"] for r in log if r.get("slide", "").startswith(f"Subset{sub}_{split}_")
            }
        have_ann = {d.name for d in ann_dir.iterdir() if d.is_dir()} if ann_dir.is_dir() else set()
        report["expected"][key] = len(slides)
        report["missing_images"][key] = [s for s in slides if s not in have_img]
        report["missing_annotations"][key] = [s for s in slides if s not in have_ann]
        print(
            f"{key:14s} expected {len(slides):3d} | images have {len(slides) - len(report['missing_images'][key]):3d} "
            f"missing {len(report['missing_images'][key]):3d} | annotations have "
            f"{len(slides) - len(report['missing_annotations'][key]):3d}",
            flush=True,
        )
    bad = [r for r in log if r.get("status", "").startswith("bad_zip")]
    for r in bad:
        print(f"BAD ZIP: {r['source']}", flush=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "subset12_check.json").write_text(json.dumps(report, indent=2))
    print(f"report: {args.dest / 'subset12_check.json'}", flush=True)


if __name__ == "__main__":
    main()
