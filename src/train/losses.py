"""Segmentation losses with per-sample reduction for noise detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _one_hot(labels: Tensor, num_classes: int) -> Tensor:
    """labels: (B, H, W) long -> (B, C, H, W) float."""
    return F.one_hot(labels.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _smoothed_one_hot(
    labels: Tensor,
    num_classes: int,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Match PyTorch CE label smoothing: (1-ε) on true class, ε/(C-1) elsewhere."""
    if label_smoothing <= 0.0:
        return _one_hot(labels, num_classes)
    oh = _one_hot(labels, num_classes)
    eps = float(label_smoothing)
    return oh * (1.0 - eps) + (1.0 - oh) * (eps / max(num_classes - 1, 1))


def gleason_adjacent_soft_targets(
    labels: Tensor,
    num_classes: int = 6,
    *,
    alpha: float = 0.1,
    g45_alpha: float | None = None,
    include_benign: bool = True,
    cancer_classes: tuple[int, ...] = (3, 4, 5),
    benign_class: int = 2,
) -> Tensor:
    """
    Soft labels for noisy Gleason masks: spread mass only to adjacent grades.

    Default (Omar 2026-08-11): chain **benign ↔ G3 ↔ G4 ↔ G5** with α=0.1.
      benign → 0.90 ben + 0.10 G3
      G3     → 0.05 ben + 0.90 G3 + 0.05 G4
      G4     → 0.05 G3 + 0.90 G4 + 0.05 G5
      G5     → 0.10 G4 + 0.90 G5
    Stroma / ignore stay hard.

    Legacy cancer-only (`include_benign=False`): G3↔G4↔G5; optional ``g45_alpha``
    softens only the G4↔G5 edge (e.g. α=0.15, g45=0.22).
    """
    if alpha <= 0.0:
        return _one_hot(labels, num_classes)
    labels = labels.long()
    out = torch.zeros(
        *labels.shape,
        num_classes,
        device=labels.device,
        dtype=torch.float32,
    )

    if include_benign:
        chain = (benign_class, *cancer_classes)
        soft_set = set(chain)
        for c in chain:
            mask = labels == c
            if not mask.any():
                continue
            i = chain.index(c)
            neigh: list[int] = []
            if i > 0:
                neigh.append(chain[i - 1])
            if i < len(chain) - 1:
                neigh.append(chain[i + 1])
            if len(neigh) == 1:
                out[..., c][mask] = 1.0 - alpha
                out[..., neigh[0]][mask] = alpha
            else:
                share = alpha / 2.0
                out[..., c][mask] = 1.0 - alpha
                for n in neigh:
                    out[..., n][mask] = share
        # Hard labels for stroma and any class outside the soft chain.
        for c in range(num_classes):
            if c in soft_set or c == 0:
                continue
            mask = labels == c
            if mask.any():
                out[..., c][mask] = 1.0
        return out.permute(0, 3, 1, 2)

    cancer = set(cancer_classes)
    g45 = float(alpha if g45_alpha is None else g45_alpha)
    for c in cancer_classes:
        mask = labels == c
        if not mask.any():
            continue
        if c == 3:
            out[..., 3][mask] = 1.0 - alpha
            out[..., 4][mask] = alpha
        elif c == 4:
            # Preserve the original G3 share; direct extra uncertainty to G5.
            g3_share = alpha / 2.0
            out[..., 3][mask] = g3_share
            out[..., 5][mask] = g45 - g3_share
            out[..., 4][mask] = 1.0 - g45
        elif c == 5:
            out[..., 4][mask] = g45
            out[..., 5][mask] = 1.0 - g45
        else:
            out[..., c][mask] = 1.0
    # Hard labels for stroma, benign, and any other non-cancer class.
    for c in range(num_classes):
        if c in cancer or c == 0:
            continue
        mask = labels == c
        if mask.any():
            out[..., c][mask] = 1.0
    return out.permute(0, 3, 1, 2)


def _resolve_soft_targets(
    labels: Tensor,
    num_classes: int,
    *,
    label_smoothing: float = 0.0,
    adjacent_soft_alpha: float = 0.0,
    g45_soft_alpha: float | None = None,
    include_benign_soft: bool = True,
) -> Tensor:
    if adjacent_soft_alpha > 0.0:
        return gleason_adjacent_soft_targets(
            labels,
            num_classes,
            alpha=adjacent_soft_alpha,
            g45_alpha=g45_soft_alpha,
            include_benign=include_benign_soft,
        )
    labels_safe = labels.clone()
    return _smoothed_one_hot(
        labels_safe, num_classes, label_smoothing=label_smoothing,
    )


def soft_ce_from_targets(
    logits: Tensor,
    soft_targets: Tensor,
    *,
    class_weight: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Weighted CE against a soft target distribution (B, C, H, W)."""
    log_probs = F.log_softmax(logits, dim=1)
    if class_weight is not None:
        w = class_weight.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        per_pixel = -(soft_targets * log_probs * w).sum(dim=1)
    else:
        per_pixel = -(soft_targets * log_probs).sum(dim=1)
    if valid_mask is not None:
        per_pixel = per_pixel * valid_mask.float()
        return per_pixel.sum() / valid_mask.float().sum().clamp(min=1.0)
    return per_pixel.mean()


def soft_dice_per_sample(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    adjacent_soft_alpha: float = 0.0,
    g45_soft_alpha: float | None = None,
    include_benign_soft: bool = True,
    soft_targets: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Mean soft Dice loss per batch item, excluding ignore_index from denominator."""
    probs = F.softmax(logits, dim=1)
    targets = targets.long()
    valid = targets != ignore_index
    if soft_targets is None:
        targets_safe = targets.clone()
        targets_safe[~valid] = 0
        target_oh = _resolve_soft_targets(
            targets_safe,
            num_classes,
            label_smoothing=label_smoothing,
            adjacent_soft_alpha=adjacent_soft_alpha,
            g45_soft_alpha=g45_soft_alpha,
            include_benign_soft=include_benign_soft,
        )
    else:
        target_oh = soft_targets

    valid_mask = valid.unsqueeze(1).float()
    probs = probs * valid_mask
    target_oh = target_oh * valid_mask

    dims = (2, 3)
    intersection = (probs * target_oh).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    dice_fg = dice[:, 1:]
    has_fg = (target_oh[:, 1:].sum(dim=dims) > 0).float()
    dice_fg = (1.0 - dice_fg) * has_fg
    denom = has_fg.sum(dim=1).clamp(min=1.0)
    return dice_fg.sum(dim=1) / denom


def cross_entropy_per_sample(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = 0,
    class_weight: Tensor | None = None,
) -> Tensor:
    weight = None
    if class_weight is not None:
        weight = class_weight.to(device=logits.device, dtype=logits.dtype)
    per_pixel = F.cross_entropy(
        logits,
        targets.long(),
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = targets != ignore_index
    valid_counts = valid.view(valid.shape[0], -1).sum(dim=1).clamp(min=1)
    loss_sum = (per_pixel * valid.float()).view(per_pixel.shape[0], -1).sum(dim=1)
    return loss_sum / valid_counts


def dice_ce_loss_per_sample(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    class_weight: Tensor | None = None,
) -> Tensor:
    """Per-sample Dice + weighted CE loss, shape (B,)."""
    ce = cross_entropy_per_sample(
        logits, targets, ignore_index=ignore_index, class_weight=class_weight,
    )
    dice = soft_dice_per_sample(
        logits,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return ce_weight * ce + dice_weight * dice


def dice_ce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    class_weight: Tensor | None = None,
) -> Tensor:
    return dice_ce_loss_per_sample(
        logits,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        class_weight=class_weight,
    ).mean()


def segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    weight_map: Tensor,
    class_weights: Tensor,
    *,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    num_classes: int = 6,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    adjacent_soft_alpha: float = 0.0,
    g45_soft_alpha: float | None = None,
    include_benign_soft: bool = True,
    soft_targets: Tensor | None = None,
) -> Tensor:
    """Combined weighted CE (with per-pixel weight map) + custom soft Dice.

    If ``soft_targets`` is provided (B, C, H, W), it is used directly for both
    CE and Dice. Otherwise soft targets are derived from the hard ``targets``
    via label smoothing / adjacent soft.
    """
    targets = targets.long()
    valid = targets != ignore_index
    targets_safe = targets.clone()
    targets_safe[~valid] = 0
    if soft_targets is None:
        soft_targets = _resolve_soft_targets(
            targets_safe,
            num_classes,
            label_smoothing=label_smoothing,
            adjacent_soft_alpha=adjacent_soft_alpha,
            g45_soft_alpha=g45_soft_alpha,
            include_benign_soft=include_benign_soft,
        )

    cw = class_weights.to(device=logits.device, dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=1)
    w = cw.view(1, -1, 1, 1)
    per_pixel_ce = -(soft_targets * log_probs * w).sum(dim=1)
    masked_ce = per_pixel_ce * weight_map
    ce_loss = masked_ce.sum() / (weight_map.sum() + 1e-8)

    dice_loss = soft_dice_per_sample(
        logits,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        soft_targets=soft_targets,
    ).mean()

    return ce_weight * ce_loss + dice_weight * dice_loss


def oeem_weight_map_global(per_pixel_loss: Tensor, *, ignore_mask: Tensor | None = None) -> Tensor:
    """Published OEEM \(W_{l\_norm}\) (Li et al. MICCAI 2022) — global over H×W.

    From ``vendor/OEEM/.../cross_entropy_loss.py``:

        metric = -loss.detach().reshape(B, H*W)
        weight = softmax(metric, dim=1)
        weight = weight / weight.mean(dim=1, keepdim=True)

    **Do not use on this dataset as-is.** Global normalization compares rare
    hard classes (G5) against easy stroma/background and systematically
    down-weights them. Prefer :func:`oeem_weight_map_per_class`.
    """
    loss = per_pixel_loss.detach()
    B, H, W = loss.shape
    metric = -loss.reshape(B, H * W)
    weight = F.softmax(metric, dim=1)
    weight = weight / weight.mean(dim=1, keepdim=True).clamp(min=1e-8)
    weight = weight.reshape(B, H, W)
    if ignore_mask is not None:
        weight = weight.masked_fill(ignore_mask, 1.0)
    return weight


def oeem_weight_map_per_class(
    per_pixel_loss: Tensor,
    seg_target_classes: Tensor,
    *,
    ignore_index: int = 0,
    min_pixels: int = 8,
) -> Tensor:
    """Per-class OEEM easy-example weights (project-specific modification).

    Same normalized-loss idea as OEEM Eq.6, but the softmax / mean
    renormalization runs **inside each class** so a G5 pixel is only judged
    against other G5 pixels' losses — not against easy background/stroma.
    Pixels of a class with fewer than ``min_pixels`` keep weight 1.0
    (default 8 — real 512² batches almost never have 2–7 G5 pixels when G5
    is present, but tiny sets make within-class softmax unstable / near-one-hot).
    Ignore-index pixels keep weight 1.0.
    """
    loss = per_pixel_loss.detach()
    target = seg_target_classes.long()
    weight = torch.ones_like(loss)
    B = loss.shape[0]
    for b in range(B):
        classes = target[b].unique()
        for c in classes.tolist():
            if int(c) == ignore_index:
                continue
            class_mask = target[b] == c
            n = int(class_mask.sum().item())
            if n < min_pixels:
                continue
            class_loss = loss[b][class_mask]
            sm = F.softmax(-class_loss, dim=0)
            weight[b][class_mask] = sm / sm.mean().clamp(min=1e-8)
    return weight


def oeem_weight_map_for_unflagged(
    per_pixel_loss: Tensor,
    seg_target_classes: Tensor,
    flag_mask: Tensor,
    *,
    ignore_index: int = 0,
    per_class: bool = True,
) -> Tensor:
    """OEEM weights with Rules 1-3 flagged pixels forced to weight 1.0.

    Reasoning (not incidental): Rules 1-3 already made an evidence-based,
    full-weight decision about the *target* on flagged pixels. OEEM is a second,
    independent mechanism that reweights loss by how "easy/credible" a pixel
    looks under the current prediction. If OEEM were also allowed to down-weight
    those same pixels, the two corrections would fight: ISUP says "learn this
    corrected label hard," OEEM says "this high-CE pixel is probably noise —
    ignore it." Forcing weight=1.0 on ``flag_mask`` keeps the division of labor
    clean — OEEM only handles UNFLAGGED pixels, where no explicit ISUP edit was
    made.
    """
    if per_class:
        weight = oeem_weight_map_per_class(
            per_pixel_loss, seg_target_classes, ignore_index=ignore_index
        )
    else:
        ignore = seg_target_classes.long() == ignore_index
        weight = oeem_weight_map_global(per_pixel_loss, ignore_mask=ignore)
    weight = torch.where(flag_mask.bool(), torch.ones_like(weight), weight)
    return weight


def mean_oeem_weight_by_class(
    weight_map: Tensor,
    seg_target_classes: Tensor,
    *,
    num_classes: int = 6,
    ignore_index: int = 0,
) -> dict[int, float]:
    """Average OEEM weight per class — smoke-test / monitoring helper."""
    target = seg_target_classes.long()
    out: dict[int, float] = {}
    for c in range(num_classes):
        if c == ignore_index:
            continue
        mask = target == c
        if mask.any():
            out[c] = float(weight_map[mask].mean().item())
        else:
            out[c] = float("nan")
    return out


def isup_informed_segmentation_loss(
    pred: Tensor,
    base_seg_target: Tensor,
    corrected_target: Tensor,
    flag_mask: Tensor,
    class_weights: Tensor,
    loss_weight_map: Tensor,
    *,
    adjacent_soft_alpha: float = 0.1,
    g45_soft_alpha: float | None = None,
    include_benign_soft: bool = True,
    label_smoothing: float = 0.0,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    num_classes: int = 6,
    ignore_index: int = 0,
) -> tuple[Tensor, dict[str, float]]:
    """Single segmentation loss whose targets have been edited by Rules 1-3.

    On pixels ``flag_mask`` marks, the soft target is the rule correction
    (hard one-hot or soft blend). Everywhere else it is the usual soft target
    derived from ``base_seg_target`` (Round 1: original PANDA mask). There is
    no second fighting loss -- ISUP informs segmentation by rewriting the
    area that caused the grade mismatch.
    """
    base = base_seg_target.long()
    valid = base != ignore_index
    base_safe = base.clone()
    base_safe[~valid] = 0
    soft = _resolve_soft_targets(
        base_safe,
        num_classes,
        label_smoothing=label_smoothing,
        adjacent_soft_alpha=adjacent_soft_alpha,
        g45_soft_alpha=g45_soft_alpha,
        include_benign_soft=include_benign_soft,
    )
    flag = flag_mask.to(dtype=soft.dtype).unsqueeze(1)
    soft = soft * (1.0 - flag) + corrected_target.to(dtype=soft.dtype) * flag

    loss = segmentation_loss(
        pred,
        base,
        loss_weight_map,
        class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        num_classes=num_classes,
        ignore_index=ignore_index,
        soft_targets=soft,
    )
    flagged = int(flag_mask.float().sum().item())
    metrics = {
        "seg": float(loss.item()),
        "pseudo": 0.0,  # no separate term; kept for log schema compatibility
        "pseudo_pixels_flagged": flagged,
        "total": float(loss.item()),
    }
    return loss, metrics


def combined_loss(
    pred: Tensor,
    seg_target: Tensor,
    corrected_target: Tensor,
    flag_mask: Tensor,
    class_weights: Tensor,
    loss_weight_map: Tensor,
    *,
    w_seg: float = 0.70,
    w_pseudo: float = 0.30,
    adjacent_soft_alpha: float = 0.1,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    num_classes: int = 6,
    ignore_index: int = 0,
) -> tuple[Tensor, dict[str, float]]:
    """Deprecated dual-loss path (seg + pseudo fight). Prefer
    :func:`isup_informed_segmentation_loss`.

    Kept for smoke-test / archive comparison only.
    """
    seg_loss = segmentation_loss(
        pred,
        seg_target,
        loss_weight_map,
        class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        num_classes=num_classes,
        ignore_index=ignore_index,
        adjacent_soft_alpha=adjacent_soft_alpha,
    )

    log_probs = F.log_softmax(pred, dim=1)
    per_pixel_pseudo_ce = -(corrected_target * log_probs).sum(dim=1)
    flag_mask_f = flag_mask.float()
    flagged_pixels = flag_mask_f.sum()
    pseudo_loss = (per_pixel_pseudo_ce * flag_mask_f).sum() / (flagged_pixels + 1e-8)

    total = w_seg * seg_loss + w_pseudo * pseudo_loss
    metrics = {
        "seg": float(seg_loss.item()),
        "pseudo": float(pseudo_loss.item()),
        "pseudo_pixels_flagged": int(flagged_pixels.item()),
        "total": float(total.item()),
    }
    return total, metrics
