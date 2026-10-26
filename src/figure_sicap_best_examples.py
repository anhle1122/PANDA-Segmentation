"""Manuscript figure: SICAPv2 official-test tiles, H&E / provided mask / prediction.

Prediction is drawn the way SICAP scores it: background, stroma, and benign are
non-cancer (uncolored). Tiles are picked per sheet grade by per-tile Dice
against the mask, one tile per slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader

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
from evaluate_sicapv2 import (  # noqa: E402
    SicapPatchDataset,
    load_official_test_names,
    predict_batch,
)
from train.sicap_metrics import accumulate_sicap_cm, dice_from_sicap_cm, remap_panda_pred  # noqa: E402
from train.sicap_slide_bag import slide_id_from_patch  # noqa: E402

DEFAULT_ROOT = Path("/common/omarmlab/members/anh/panda_data/sicapv2/SICAPv2")
GRADE_COLORS = {
    3: (241, 196, 15),
    4: (230, 126, 34),
    5: (192, 57, 43),
}
GRADE_NAMES = {0: "Non-cancer", 3: "Gleason 3", 4: "Gleason 4", 5: "Gleason 5"}


def load_model(ckpt: Path, device: torch.device) -> torch.nn.Module:
    arch = detect_arch(ckpt, "auto")
    _c, keys = _peek_state_dict(ckpt)
    _is_opt3, decode_norm, use_lora = _opt3_flags(keys)
    model = build_eval_model(arch, decode_norm=decode_norm, use_lora=use_lora)
    load_model_weights(ckpt, model)
    return model.to(device).eval()


def sheet_grades(root: Path) -> dict[str, int]:
    df = pd.read_excel(root / "partition" / "Test" / "Test.xlsx")
    out: dict[str, int] = {}
    for rec in df.itertuples(index=False):
        stem = str(rec.image_name).replace(".jpg", "")
        if rec.NC == 1:
            out[stem] = 0
        elif rec.G3 == 1:
            out[stem] = 3
        elif rec.G4 == 1:
            out[stem] = 4
        else:
            out[stem] = 5
    return out


def tile_score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Macro Dice over {NC, G3, G4, G5} classes present in mask or prediction."""
    dices = []
    per_class: dict[str, float] = {}
    for c in (0, 3, 4, 5):
        g = gt == c
        p = pred == c
        denom = int(g.sum() + p.sum())
        d = 2.0 * float((g & p).sum()) / denom if denom else float("nan")
        per_class[f"dice_{c}"] = d
        if denom:
            dices.append(d)
    cancer_g = gt >= 3
    cancer_p = pred >= 3
    denom = int(cancer_g.sum() + cancer_p.sum())
    n_ca = int(cancer_g.sum())
    return {
        **per_class,
        **{f"gt_share_{c}": float((gt == c).sum() / n_ca) if n_ca else 0.0 for c in (3, 4, 5)},
        "macro_dice": float(np.mean(dices)) if dices else 0.0,
        "cancer_dice": 2.0 * float((cancer_g & cancer_p).sum()) / denom if denom else 1.0,
        "cancer_frac": float(cancer_g.mean()),
        "pred_cancer_frac": float(cancer_p.mean()),
    }


def tissue_frac(rgb: np.ndarray) -> float:
    return float((rgb.astype(np.float32).mean(axis=2) < 215).mean())


def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32)
    for cls, color in GRADE_COLORS.items():
        m = labels == cls
        out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def flat(labels: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape + (3,), 255, dtype=np.uint8)
    for cls, color in GRADE_COLORS.items():
        out[labels == cls] = color
    return out


def boundary(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return b


def overlay_with_edges(rgb: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    out = overlay(rgb, labels, alpha)
    edge = boundary(labels)
    for _ in range(1):
        grown = edge.copy()
        grown[1:, :] |= edge[:-1, :]
        grown[:, 1:] |= edge[:, :-1]
        edge = grown
    out[edge] = (20, 20, 20)
    return out


def draw_figure(rows: list[dict], path_stem: Path, style: str, alpha: float) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.45 * n), dpi=300)
    if n == 1:
        axes = axes[None, :]
    for r, row in enumerate(rows):
        rgb, gt, pred = row["rgb"], row["gt"], row["pred"]
        if style == "overlay":
            panels = [rgb, overlay_with_edges(rgb, gt, alpha), overlay_with_edges(rgb, pred, alpha)]
        else:
            panels = [rgb, flat(gt), flat(pred)]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color("0.35")
            if r == 0:
                ax.set_title(("H&E", "Provided mask", "Prediction")[c], fontsize=10, pad=4)
        axes[r, 0].set_ylabel(GRADE_NAMES[row["sheet"]], fontsize=10, labelpad=6)
    handles = [
        Patch(facecolor="white", edgecolor="0.4", label="Non-cancer"),
        *[
            Patch(facecolor=np.array(GRADE_COLORS[g]) / 255, edgecolor="0.4", label=GRADE_NAMES[g])
            for g in (3, 4, 5)
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.05, wspace=0.04, hspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sicap-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--per-grade", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.45)
    args = p.parse_args()

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    grades = sheet_grades(args.sicap_root)
    ds = SicapPatchDataset(args.sicap_root, load_official_test_names(args.sicap_root))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    cm = np.zeros((6, 6), dtype=np.int64)
    records: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    for images, masks, stems in loader:
        raw = predict_batch(model, images, scale_factor=1.0, amp=True, device=device)
        gt_b = masks.numpy()
        accumulate_sicap_cm(cm, raw, gt_b)
        scored = remap_panda_pred(raw)
        rgb_b = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for i, stem in enumerate(stems):
            stem = str(stem)
            s = tile_score(scored[i], gt_b[i])
            s.update(
                stem=stem,
                slide=slide_id_from_patch(stem),
                sheet=grades.get(stem, -1),
                tissue=tissue_frac(rgb_b[i]),
                classes_in_mask=sorted(int(c) for c in np.unique(gt_b[i])),
            )
            records.append(s)
            preds[stem] = raw[i].astype(np.uint8)
    overall = dice_from_sicap_cm(cm)
    print(
        f"official cancer={overall['cancer']:.4f} mean4={overall['mean4']:.4f} "
        f"G3={overall['G3']:.4f} G4={overall['G4']:.4f} G5={overall['G5']:.4f}",
        flush=True,
    )

    df = pd.DataFrame(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["classes_in_mask"]).to_csv(args.out_dir / "official_tile_scores.csv", index=False)

    picks: dict[int, list[dict]] = {}
    for g in (3, 4, 5):
        pool = df[
            (df.sheet == g)
            & (df.cancer_frac >= 0.25)
            & (df.cancer_frac <= 0.90)
            & (df.tissue >= 0.5)
            & (df[f"gt_share_{g}"] >= 0.8)
        ].copy()
        pool["grade_score"] = 0.5 * (pool["dice_0"] + pool[f"dice_{g}"])
        pool = pool.sort_values("grade_score", ascending=False)
        picks[g] = pool.drop_duplicates("slide").head(args.per_grade).to_dict("records")
    pool = df[(df.sheet == 0) & (df.tissue >= 0.7)].sort_values(
        ["pred_cancer_frac", "tissue"], ascending=[True, False]
    )
    picks[0] = pool.drop_duplicates("slide").head(args.per_grade).to_dict("records")

    by_name = {n: i for i, n in enumerate(ds.names)}

    def materialize(rec: dict) -> dict:
        x, y, _ = ds[by_name[rec["stem"]]]
        rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return {
            **rec,
            "rgb": rgb,
            "gt": y.numpy().astype(np.uint8),
            "pred": remap_panda_pred(preds[rec["stem"]]),
        }

    cand_dir = args.out_dir / "candidates"
    cand_dir.mkdir(exist_ok=True)
    chosen_meta = []
    for g in (0, 3, 4, 5):
        for k, rec in enumerate(picks[g], start=1):
            row = materialize(rec)
            draw_figure([row], cand_dir / f"{GRADE_NAMES[g].replace(' ', '')}_{k}_{rec['stem']}", "overlay", args.alpha)
            chosen_meta.append(
                {
                    k2: rec[k2]
                    for k2 in (
                    "stem",
                    "slide",
                    "sheet",
                    "macro_dice",
                    "cancer_dice",
                    "dice_0",
                    f"dice_{g}" if g else "dice_0",
                    "cancer_frac",
                    "pred_cancer_frac",
                )
                }
                | {"grade_dice": rec[f"dice_{g}"] if g else rec["dice_0"]}
                | {"rank": k}
            )

    main_rows = [materialize(picks[g][0]) for g in (0, 3, 4, 5) if picks[g]]
    draw_figure(main_rows, args.out_dir / "figure_sicap_examples_overlay", "overlay", args.alpha)
    draw_figure(main_rows, args.out_dir / "figure_sicap_examples_flat", "flat", args.alpha)
    for row in main_rows:
        Image.fromarray(row["rgb"]).save(args.out_dir / f"{row['stem']}__he.png")
        Image.fromarray(row["gt"]).save(args.out_dir / f"{row['stem']}__mask.png")
        Image.fromarray(row["pred"]).save(args.out_dir / f"{row['stem']}__pred.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "partition": "official Test.xlsx (2,122 tiles, 31 slides)",
        "prediction_display": "background/stroma/benign shown as non-cancer, as scored",
        "selection": "per sheet grade, top mean of NC Dice and sheet-grade Dice on the tile; one tile per slide; "
        "cancer 25-90% of tile; sheet grade >= 80% of mask cancer; tissue >= 50%. "
        "NC: tissue >= 70%, least predicted cancer.",
        "official_metrics": {k: float(overall[k]) for k in ("cancer", "mean4", "NC", "G3", "G4", "G5")},
        "per_tile_macro_dice_official_median": float(df.macro_dice.median()),
        "main_figure": [r["stem"] for r in main_rows],
        "candidates": chosen_meta,
    }
    (args.out_dir / "figure_meta.json").write_text(json.dumps(summary, indent=2))
    for m in chosen_meta:
        print(
            f"{GRADE_NAMES[m['sheet']]:10s} #{m['rank']} grade={m['grade_dice']:.3f} "
            f"NC={m['dice_0']:.3f} cancer={m['cancer_dice']:.3f} gt_ca={m['cancer_frac']:.2f} {m['stem']}",
            flush=True,
        )
    print(f"OK {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
