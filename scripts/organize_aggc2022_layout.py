#!/usr/bin/env python3
"""Put AGGC Subset1 Train TIFFs where the evaluator reads them, then park downloads.

The scorer only looks at ``Subset{1,2}/{Train,Test}/{images,annotations}``.
Batch4 left 19 loose TIFFs and 6 zips under ``_incoming``. This script:

- hardlinks each loose TIFF into ``Subset1/Train/images`` (no second copy)
- extracts TIFF members from the batch4 zips into that same folder
- skips a file under 50 MB (Subset1_Train_76.tiff is a 4.5 MB stub)
- never replaces an image that is already present and at least as large
- moves ``_incoming`` to ``_archive/incoming`` after the install succeeds

Subset1/2 Test and Subset2 Train are already complete and are not rewritten.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/common/omarmlab/data/AGGC_2022")
INCOMING = ROOT / "_incoming"
BATCH4 = INCOMING / "batch4" / "Subset 1_train"
DEST = ROOT / "Subset1" / "Train" / "images"
ARCHIVE = ROOT / "_archive" / "incoming"
MANIFEST = ROOT / "_archive" / "subset1_train_install.json"
MIN_BYTES = 50 * 1024 * 1024


def slide_id(name: str) -> int | None:
    tail = Path(name).stem.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def install_file(src: Path, dest: Path, report: dict) -> None:
    if src.stat().st_size < MIN_BYTES:
        report["skipped_small"].append({"name": src.name, "bytes": src.stat().st_size})
        print(f"SKIP small {src.name} ({src.stat().st_size} bytes)", flush=True)
        return
    if dest.exists() and dest.stat().st_size >= src.stat().st_size:
        report["already_present"].append(dest.name)
        print(f"KEEP existing {dest.name}", flush=True)
        return
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        how = "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        how = "copy"
    report["installed"].append({"name": dest.name, "bytes": dest.stat().st_size, "how": how})
    print(f"INSTALL {how} {dest.name} ({dest.stat().st_size / 1e9:.2f}G)", flush=True)


def install_zip(zpath: Path, report: dict) -> None:
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".tif", ".tiff")):
                continue
            name = Path(info.filename).name
            if slide_id(name) is None:
                report["skipped_name"].append(info.filename)
                continue
            dest = DEST / name
            if info.file_size < MIN_BYTES:
                report["skipped_small"].append({"name": name, "bytes": info.file_size, "zip": zpath.name})
                print(f"SKIP small {name} inside {zpath.name}", flush=True)
                continue
            if dest.exists() and dest.stat().st_size >= info.file_size:
                report["already_present"].append(name)
                print(f"KEEP existing {name}", flush=True)
                continue
            if dest.exists():
                dest.unlink()
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            report["installed"].append({"name": name, "bytes": dest.stat().st_size, "how": f"unzip:{zpath.name}"})
            print(f"INSTALL unzip {name} ({dest.stat().st_size / 1e9:.2f}G)", flush=True)


def main() -> None:
    if not BATCH4.is_dir():
        raise SystemExit(f"missing {BATCH4}")
    DEST.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "installed": [],
        "already_present": [],
        "skipped_small": [],
        "skipped_name": [],
    }
    for tiff in sorted(BATCH4.glob("*.tiff")) + sorted(BATCH4.glob("*.tif")):
        install_file(tiff, DEST / tiff.name, report)
    for zpath in sorted(BATCH4.glob("*.zip")):
        print(f"=== zip {zpath.name} ===", flush=True)
        install_zip(zpath, report)

    ann = ROOT / "Subset1" / "Train" / "annotations"
    ann_ids = {slide_id(p.name) for p in ann.iterdir() if p.is_dir()}
    img_ids = {slide_id(p.name) for p in DEST.glob("*.tif*")}
    ann_ids.discard(None)
    img_ids.discard(None)
    report["n_images_now"] = len(img_ids)
    report["n_annotations"] = len(ann_ids)
    report["still_missing_ids"] = sorted(ann_ids - img_ids)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))
    print(
        f"IMAGES {report['n_images_now']} / annotations {report['n_annotations']} "
        f"missing {report['still_missing_ids']}",
        flush=True,
    )

    if ARCHIVE.exists():
        raise SystemExit(f"refusing to overwrite {ARCHIVE}")
    shutil.move(str(INCOMING), str(ARCHIVE))
    print(f"MOVED {INCOMING} -> {ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
