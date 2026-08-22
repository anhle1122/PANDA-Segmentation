"""Between-round correction identity, paths, registry, and gates.

Every artifact is keyed by pack_tag = {run_tag}_{recipe_version} plus epoch.
Nothing here submits training. Round N+1 label source is a manual decision.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from patch_utils import PROJECT

REGISTRY_PATH = PROJECT / "outputs" / "pseudo_label" / "model_registry.json"
COMPARISON_CSV = PROJECT / "outputs" / "pseudo_label" / "correction_comparison.csv"
COMPARISON_MD = PROJECT / "outputs" / "pseudo_label" / "correction_comparison.md"
PACK_ROOT = PROJECT / "outputs" / "pseudo_label"
CKPT_ROOT = PROJECT / "outputs" / "checkpoints"
SCORECARD = PROJECT / "outputs" / "docs" / "opt3_this_run" / "epoch_external_scorecard.csv"
EVAL_ROOT = PROJECT / "outputs" / "pseudo_label" / "epoch_eval"

# Live 5443101 was the Aug 12 in-memory trainer (commit 488c5e7): LoRA wrapped
# but not in AdamW, λ_slide=0.3 from ep10, ISUP used micro=4 not live=64.
KNOWN_JOBS: dict[str, dict[str, str]] = {
    "5443101": {
        "run_tag": "opt3_omar6_grouped_soft01",
        "recipe_version": "pre_lora_fix",
        "code_commit": "488c5e7",
        "code_commit_note": (
            "Original in-memory trainer from Aug 12 (488c5e7). "
            "LoRA QKV was wrapped but not in AdamW; λ_slide=0.3 from ep10; "
            "--live-patches 64 was printed but ISUP perm used micro_batch_size=4; "
            "no --live-chunk / --decoder-checkpoint (OOM path at ep37)."
        ),
    },
    "5445276": {
        "run_tag": "opt3_omar6_locked",
        "recipe_version": "post_lora_fix",
        "code_commit": "5a23882",
        "code_commit_note": (
            "Locked Omar-6 wiring: LoRA in AdamW, WIRING_OK live=64, "
            "chunk=4 + decoder checkpoint, λ_slide warmup to 0.3."
        ),
    },
    "5445445": {
        "run_tag": "opt3_omar6_locked",
        "recipe_version": "locked_r2",
        "code_commit": "5a23882",
        "code_commit_note": (
            "Locked r2 (opt3_omar6_locked): Omar-6 wiring, hang-fix resume job 5445445. "
            "Teacher for between-round correction is epoch 14 (PANDA+ cancer 0.642)."
        ),
    },
    "5445430": {
        "run_tag": "opt3_omar6_lambda015",
        "recipe_version": "post_lora_fix_lambda015",
        "code_commit": "5a23882",
        "code_commit_note": (
            "Same locked wiring as r2, λ_slide target 0.15 with warmup."
        ),
    },
}

DEFAULT_INCUMBENT = {
    "label": "teacher_A_ep5",
    "panda_plus_cancer_dice": 0.563,
    "panda_plus_isup_match": None,
}

STATUS_REGISTERED = "REGISTERED"
STATUS_VALIDATED = "VALIDATED"
STATUS_NOT_VALIDATED = "NOT_VALIDATED"
STATUS_VALIDATION_ONLY = "VALIDATION_ONLY"
FLAG_DICE_UP_ISUP_NOT = "DICE_UP_ISUP_NOT"
FLAG_MISSING_ISUP = "MISSING_PANDA_PLUS_ISUP"
FLAG_MISSING_PLUS_DICE = "MISSING_PANDA_PLUS_DICE"
FLAG_G5_BIAS = "G5_SWAP_BIAS"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pack_tag(run_tag: str, recipe_version: str) -> str:
    run_tag = run_tag.strip()
    recipe_version = recipe_version.strip()
    if not run_tag or not recipe_version:
        raise ValueError("run_tag and recipe_version are required")
    if run_tag == recipe_version or run_tag.endswith(f"_{recipe_version}"):
        return run_tag
    return f"{run_tag}_{recipe_version}"


def model_id(run_tag: str, recipe_version: str, epoch: int) -> str:
    return f"{pack_tag(run_tag, recipe_version)}_ep{int(epoch):03d}"


def teacher_pack_dir(run_tag: str, recipe_version: str, epoch: int, *, root: Path = PACK_ROOT) -> Path:
    return root / f"teacher_{model_id(run_tag, recipe_version, epoch)}"


def corrections_dir(run_tag: str, recipe_version: str, epoch: int, *, root: Path = PACK_ROOT) -> Path:
    return root / f"corrections_{model_id(run_tag, recipe_version, epoch)}"


def parse_epoch_from_ckpt(path: Path) -> int:
    name = path.name
    marker = "epoch_"
    if marker not in name:
        raise ValueError(f"cannot parse epoch from {path.name}")
    rest = name.split(marker, 1)[1]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f"cannot parse epoch from {path.name}")
    return int(digits)


def resolve_checkpoint(path: Path | None, run_tag: str, epoch: int) -> Path:
    if path is not None:
        ckpt = path.expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        return ckpt
    ckpt_dir = CKPT_ROOT / f"uni2_upernet_raw_{run_tag}"
    hits = sorted(ckpt_dir.glob(f"epoch_{int(epoch):03d}_cancer_*.pth"))
    if not hits:
        raise FileNotFoundError(f"no epoch_{int(epoch):03d}_*.pth under {ckpt_dir}")
    return hits[-1]


def pack_exists_for_tag_epoch(pack_dir: Path) -> bool:
    """True if this exact tag+epoch pack dir already exists (do not re-run)."""
    if not pack_dir.exists():
        return False
    if (pack_dir / "pack_config.json").is_file():
        return True
    if any(pack_dir.glob("*_srcpred.h5")):
        return True
    if any(pack_dir.iterdir()):
        return True
    return False


def empty_registry() -> dict[str, Any]:
    return {
        "updated_at": now_iso(),
        "auto_train": False,
        "incumbent_teacher": dict(DEFAULT_INCUMBENT),
        "models": {},
    }


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if not path.is_file():
        return empty_registry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("models", {})
    payload.setdefault("auto_train", False)
    payload.setdefault("incumbent_teacher", dict(DEFAULT_INCUMBENT))
    return payload


def save_registry(payload: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    payload["updated_at"] = now_iso()
    payload["auto_train"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def upsert_model(entry: dict[str, Any], *, path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = load_registry(path)
    mid = entry["model_id"]
    prev = payload["models"].get(mid, {})
    merged = {**prev, **entry}
    if "metrics" in prev or "metrics" in entry:
        merged["metrics"] = {**(prev.get("metrics") or {}), **(entry.get("metrics") or {})}
    if "paths" in prev or "paths" in entry:
        merged["paths"] = {**(prev.get("paths") or {}), **(entry.get("paths") or {})}
    if "jobs" in prev or "jobs" in entry:
        merged["jobs"] = {**(prev.get("jobs") or {}), **(entry.get("jobs") or {})}
    if "g5_bias" in prev or "g5_bias" in entry:
        merged["g5_bias"] = {**(prev.get("g5_bias") or {}), **(entry.get("g5_bias") or {})}
    flags = list(dict.fromkeys((prev.get("validation_flags") or []) + (entry.get("validation_flags") or [])))
    if "validation_flags" in entry:
        flags = list(entry["validation_flags"])
    merged["validation_flags"] = flags
    merged["auto_train"] = False
    payload["models"][mid] = merged
    save_registry(payload, path)
    return merged


def identity_entry(
    *,
    checkpoint: Path,
    run_tag: str,
    recipe_version: str,
    source_job_id: str,
    code_commit: str,
    code_commit_note: str = "",
    epoch: int | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    epoch = int(epoch if epoch is not None else parse_epoch_from_ckpt(checkpoint))
    mid = model_id(run_tag, recipe_version, epoch)
    ptag = pack_tag(run_tag, recipe_version)
    return {
        "model_id": mid,
        "run_tag": run_tag,
        "recipe_version": recipe_version,
        "pack_tag": ptag,
        "epoch": epoch,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "source_job_id": str(source_job_id),
        "code_commit": code_commit,
        "code_commit_note": code_commit_note,
        "paths": {
            "teacher_pack": str(teacher_pack_dir(run_tag, recipe_version, epoch)),
            "corrections": str(corrections_dir(run_tag, recipe_version, epoch)),
            "eval_dir": str(EVAL_ROOT / run_tag / f"ep{epoch:03d}"),
            "train_log": str(CKPT_ROOT / f"uni2_upernet_raw_{run_tag}" / "training_log.csv"),
        },
        "metrics": {},
        "validation_status": STATUS_VALIDATION_ONLY if validation_only else STATUS_REGISTERED,
        "validation_flags": [],
        "g5_bias": {},
        "jobs": {"cache": None, "referee": None, "eval": None},
        "auto_train": False,
        "registered_at": now_iso(),
    }


def _f(row: dict, *keys: str) -> float | None:
    for key in keys:
        raw = row.get(key, "")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def metrics_from_labeled_csv(path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "panda_plus_cancer_dice": None,
        "panda_plus_g5_precision": None,
        "panda_plus_g5_recall": None,
    }
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = str(raw.get("class", "")).strip()
            if name == "cancer_dice":
                out["panda_plus_cancer_dice"] = _f(raw, "dice")
            elif name.upper() == "G5":
                out["panda_plus_g5_precision"] = _f(raw, "precision")
                out["panda_plus_g5_recall"] = _f(raw, "recall")
            elif name == "g5_recall" and out["panda_plus_g5_recall"] is None:
                out["panda_plus_g5_recall"] = _f(raw, "recall")
    return out


def metrics_from_train_log(path: Path, epoch: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {"panda_val_cancer_dice": None}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                ep = int(float(raw["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            if ep == epoch:
                out["panda_val_cancer_dice"] = _f(raw, "cancer_dice")
                return out
    return out


def metrics_from_scorecard(path: Path, run_tag: str, epoch: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if str(raw.get("tag", "")).strip() != run_tag:
                continue
            try:
                ep = int(float(raw.get("epoch", "")))
            except (TypeError, ValueError):
                continue
            if ep != epoch:
                continue
            out = {
                "panda_val_cancer_dice": _f(raw, "panda_val_cancer_dice"),
                "panda_isup_match": _f(raw, "panda_isup_match"),
                "panda_plus_cancer_dice": _f(raw, "panda_plus_cancer_dice"),
                "panda_plus_isup_match": _f(raw, "panda_plus_isup_match"),
                "panda_plus_g5_precision": _f(raw, "panda_plus_g5_precision"),
            }
    return out


def json_rate(path: Path, *keys: str) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return None


def collect_metrics(entry: dict[str, Any], *, scorecard: Path = SCORECARD) -> dict[str, float | None]:
    paths = entry.get("paths") or {}
    epoch = int(entry["epoch"])
    run_tag = str(entry["run_tag"])
    metrics: dict[str, float | None] = {}
    metrics.update(metrics_from_scorecard(scorecard, run_tag, epoch))
    metrics.update({k: v for k, v in metrics_from_train_log(Path(paths.get("train_log", "")), epoch).items() if v is not None})
    eval_dir = Path(paths.get("eval_dir", EVAL_ROOT / run_tag / f"ep{epoch:03d}"))
    labeled = metrics_from_labeled_csv(eval_dir / "panda_plus_dice_labeled.csv")
    metrics.update({k: v for k, v in labeled.items() if v is not None})
    plus_isup = json_rate(eval_dir / "panda_plus_isup_summary.json", "isup_match_rate", "match_rate")
    if plus_isup is not None:
        metrics["panda_plus_isup_match"] = plus_isup
    panda_isup = json_rate(eval_dir / "panda_isup_summary.json", "match_rate")
    if panda_isup is not None:
        metrics["panda_isup_match"] = panda_isup
    return metrics


def assess_validation(
    metrics: dict[str, float | None],
    incumbent: dict[str, Any] | None = None,
    *,
    validation_only: bool = False,
) -> tuple[str, list[str]]:
    """Return (status, flags). Never auto-trains."""
    if validation_only:
        return STATUS_VALIDATION_ONLY, []
    flags: list[str] = []
    plus = metrics.get("panda_plus_cancer_dice")
    isup = metrics.get("panda_plus_isup_match")
    if plus is None:
        flags.append(FLAG_MISSING_PLUS_DICE)
    if isup is None:
        flags.append(FLAG_MISSING_ISUP)
    inc = incumbent or DEFAULT_INCUMBENT
    inc_dice = inc.get("panda_plus_cancer_dice")
    inc_isup = inc.get("panda_plus_isup_match")
    if (
        plus is not None
        and inc_dice is not None
        and plus > float(inc_dice)
        and isup is not None
        and inc_isup is not None
        and float(isup) <= float(inc_isup)
    ):
        flags.append(FLAG_DICE_UP_ISUP_NOT)
    if FLAG_DICE_UP_ISUP_NOT in flags or FLAG_MISSING_ISUP in flags or FLAG_MISSING_PLUS_DICE in flags:
        return STATUS_NOT_VALIDATED, flags
    return STATUS_VALIDATED, flags


def g5_bias_from_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("g5_summary") or {}
    return {
        "pct_high_conf_swaps_to_g5": summary.get("pct_high_conf_swaps_to_g5"),
        "n_swap_to_g5": summary.get("n_swap_to_g5"),
        "n_high_conf_swap": summary.get("n_high_conf_swap"),
        "pct_original_mask_g5": summary.get("pct_original_mask_g5"),
        "n_original_mask_g5": summary.get("n_original_mask_g5"),
        "n_original_mask_pixels": summary.get("n_original_mask_pixels"),
        "g5_bias_flag": bool(report.get("g5_bias_flag")),
        "ratio_swap_g5_vs_mask_g5": (
            (summary["pct_high_conf_swaps_to_g5"] / summary["pct_original_mask_g5"])
            if summary.get("pct_original_mask_g5")
            else None
        ),
    }


def comparison_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mid, entry in sorted(payload.get("models", {}).items()):
        m = entry.get("metrics") or {}
        g5 = entry.get("g5_bias") or {}
        rows.append(
            {
                "model_id": mid,
                "run_tag": entry.get("run_tag", ""),
                "recipe_version": entry.get("recipe_version", ""),
                "epoch": entry.get("epoch", ""),
                "source_job_id": entry.get("source_job_id", ""),
                "code_commit": entry.get("code_commit", ""),
                "panda_val_cancer_dice": m.get("panda_val_cancer_dice"),
                "panda_plus_cancer_dice": m.get("panda_plus_cancer_dice"),
                "panda_plus_isup_match": m.get("panda_plus_isup_match"),
                "panda_plus_g5_precision": m.get("panda_plus_g5_precision"),
                "panda_plus_g5_recall": m.get("panda_plus_g5_recall"),
                "g5_swap_bias_pct": g5.get("pct_high_conf_swaps_to_g5"),
                "mask_g5_pct": g5.get("pct_original_mask_g5"),
                "validation_status": entry.get("validation_status", ""),
                "validation_flags": "|".join(entry.get("validation_flags") or []),
                "teacher_pack": (entry.get("paths") or {}).get("teacher_pack", ""),
                "corrections": (entry.get("paths") or {}).get("corrections", ""),
                "auto_train": False,
            }
        )
    return rows


def write_comparison(
    payload: dict[str, Any],
    *,
    csv_path: Path = COMPARISON_CSV,
    md_path: Path = COMPARISON_MD,
) -> list[dict[str, Any]]:
    rows = comparison_rows(payload)
    fields = [
        "model_id",
        "run_tag",
        "recipe_version",
        "epoch",
        "source_job_id",
        "code_commit",
        "panda_val_cancer_dice",
        "panda_plus_cancer_dice",
        "panda_plus_isup_match",
        "panda_plus_g5_precision",
        "panda_plus_g5_recall",
        "g5_swap_bias_pct",
        "mask_g5_pct",
        "validation_status",
        "validation_flags",
        "teacher_pack",
        "corrections",
        "auto_train",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})

    lines = [
        "# Correction candidate comparison",
        "",
        f"Updated {payload.get('updated_at', now_iso())}.",
        "No row is auto-selected for Round N+1 training. Pick manually after review.",
        "",
        "| model_id | recipe | PANDA+ Dice | PANDA+ ISUP | G5 P / R | G5-swap % | mask G5 % | status |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        g5pr = ""
        if row.get("panda_plus_g5_precision") is not None:
            rec = row.get("panda_plus_g5_recall")
            g5pr = f"{float(row['panda_plus_g5_precision']):.3f}"
            if rec is not None:
                g5pr += f" / {float(rec):.3f}"
        plus = "" if row.get("panda_plus_cancer_dice") is None else f"{float(row['panda_plus_cancer_dice']):.3f}"
        isup = "" if row.get("panda_plus_isup_match") is None else f"{100 * float(row['panda_plus_isup_match']):.1f}%"
        swap = "" if row.get("g5_swap_bias_pct") is None else f"{float(row['g5_swap_bias_pct']):.2f}"
        mask = "" if row.get("mask_g5_pct") is None else f"{float(row['mask_g5_pct']):.2f}"
        flags = row.get("validation_flags") or ""
        status = str(row.get("validation_status") or "")
        if flags:
            status = f"{status} ({flags})"
        lines.append(
            f"| `{row['model_id']}` | {row.get('recipe_version','')} | {plus} | {isup} | {g5pr} | {swap} | {mask} | {status} |"
        )
    lines.extend(["", "auto_train=false for every model.", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return rows
