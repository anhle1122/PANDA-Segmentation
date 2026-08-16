"""Slide-level ISUP grade head + helpers for Option 3 dual ISUP losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from isup_diagnostic import derive_grade


class ISUPGradeHead(nn.Module):
    """Small MLP: pooled backbone features → ISUP logits (0–5)."""

    def __init__(self, in_dim: int, *, num_isup: int = 6, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, num_isup),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """feats: (B, in_dim) or (in_dim,) → logits (B, num_isup)."""
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return self.net(feats)


def global_average_pool(feat_map: Tensor) -> Tensor:
    """(B, C, H, W) → (B, C)."""
    return feat_map.mean(dim=(2, 3))


def aggregate_softmax_probs(logits_list: list[Tensor]) -> Tensor:
    """Mean softmax over a bag of patch logits.

    Args:
        logits_list: list of (B_micro, C, H, W) tensors (same C).
    Returns:
        (C,) mean probability vector over all pixels in the bag.
    """
    parts = []
    for logits in logits_list:
        probs = F.softmax(logits.float(), dim=1)  # (b, C, H, W)
        parts.append(probs.mean(dim=(0, 2, 3)))
    stacked = torch.stack(parts, dim=0)  # (n_micro, C)
    return stacked.mean(dim=0)


def aggregate_logsumexp_logits(logits_list: list[Tensor]) -> Tensor:
    """WeGleNet-style Log-Sum-Exp pool over spatial+bag → (C,) slide logits."""
    flats = []
    for logits in logits_list:
        # (b, C, H, W) → (C, b*H*W)
        b, c, h, w = logits.shape
        flats.append(logits.float().permute(1, 0, 2, 3).reshape(c, -1))
    all_pix = torch.cat(flats, dim=1)  # (C, N)
    # logsumexp over pixels, then center for numerical ease
    return torch.logsumexp(all_pix, dim=1)


def soft_isup_logits_from_seg_probs(mean_probs: Tensor) -> Tensor:
    """Soft ISUP logits from a 6-class mean probability vector.

    Uses **absolute** G3/G4/G5 masses (already a slice of a distribution that
    sums to 1). Do not divide by cancer-only total — that would make a 1% and
    a 90% tumor slide with the same grade mix share ISUP 1–5 logits.
    """
    p3 = mean_probs[3].clamp_min(0.0)
    p4 = mean_probs[4].clamp_min(0.0)
    p5 = mean_probs[5].clamp_min(0.0)
    total = (p3 + p4 + p5).clamp(0.0, 1.0)
    return torch.stack(
        [
            (1.0 - total) * 4.0,
            p3 * 3.0 - p4 - p5,
            p3 * 2.0 + p4 * 1.5 - p5,
            p4 * 2.0 + p3 * 1.0 - p5 * 0.5,
            p4 * 2.0 + p5 * 1.5 + p3 * 0.5,
            p5 * 3.0 + p4 * 1.0,
        ]
    )


def derived_isup_ce_from_seg_probs(
    mean_probs: Tensor,
    clinician_isup: int | Tensor,
    *,
    min_area_pct: float = 0.0,
) -> tuple[Tensor, int, int]:
    """Slide-level CE using a soft proxy of derive_grade on mean class probs.

    Hard derived ISUP (for logging) uses the same ``derive_grade`` as the
    offline diagnostic. The training loss is CE between clinician ISUP and a
    soft distribution built from cancer-class masses (G3/G4/G5), so gradients
    flow into the segmentation head.

    Omar point 6: keep absolute tumor burden; compare soft argmax to hard
    ``derive_grade``. ``min_area_pct`` only affects the hard log path.
    """
    soft_logits = soft_isup_logits_from_seg_probs(mean_probs)
    target = torch.as_tensor(int(clinician_isup), device=mean_probs.device, dtype=torch.long)
    loss = F.cross_entropy(soft_logits.unsqueeze(0), target.unsqueeze(0))
    soft_isup = int(soft_logits.detach().argmax().item())

    counts = (mean_probs.detach().cpu().numpy() * 1_000_000).astype("int64")
    _, hard_isup = derive_grade(counts, min_area_pct=min_area_pct)
    return loss, int(hard_isup), soft_isup


def grade_head_ce(logits: Tensor, clinician_isup: int | Tensor) -> Tensor:
    target = torch.as_tensor(int(clinician_isup), device=logits.device, dtype=torch.long)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return F.cross_entropy(logits, target.unsqueeze(0))
