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
    alpha: float = 0.15,
    g45_alpha: float | None = None,
    cancer_classes: tuple[int, ...] = (3, 4, 5),
) -> Tensor:
    """
    Soft labels for noisy Gleason masks: spread mass only to adjacent cancer grades.

    G3 ↔ G4 ↔ G5 chain only — stroma/benign stay hard 0/1.
    Example α=0.15: G4 pixel → 0.85 G4, 0.075 G3, 0.075 G5.

    ``g45_alpha`` can make only the harder G4↔G5 boundary softer. For
    alpha=0.15, g45_alpha=0.22:
      G3 → 0.85 G3 + 0.15 G4
      G4 → 0.78 G4 + 0.075 G3 + 0.145 G5
      G5 → 0.78 G5 + 0.22 G4
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
) -> Tensor:
    if adjacent_soft_alpha > 0.0:
        return gleason_adjacent_soft_targets(
            labels,
            num_classes,
            alpha=adjacent_soft_alpha,
            g45_alpha=g45_soft_alpha,
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
) -> Tensor:
    """Combined weighted CE (with per-pixel weight map) + custom soft Dice."""
    targets = targets.long()
    valid = targets != ignore_index
    targets_safe = targets.clone()
    targets_safe[~valid] = 0
    soft_targets = _resolve_soft_targets(
        targets_safe,
        num_classes,
        label_smoothing=label_smoothing,
        adjacent_soft_alpha=adjacent_soft_alpha,
        g45_soft_alpha=g45_soft_alpha,
    )

    cw = class_weights.to(device=logits.device, dtype=logits.dtype)
    if adjacent_soft_alpha > 0.0 or label_smoothing > 0.0:
        log_probs = F.log_softmax(logits, dim=1)
        w = cw.view(1, -1, 1, 1)
        per_pixel_ce = -(soft_targets * log_probs * w).sum(dim=1)
    else:
        per_pixel_ce = F.cross_entropy(
            logits,
            targets,
            weight=cw,
            ignore_index=ignore_index,
            reduction="none",
        )
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
