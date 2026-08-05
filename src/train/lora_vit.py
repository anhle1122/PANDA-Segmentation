"""Minimal LoRA for timm ViT attention QKV (no peft dependency)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank trainable update: y = W x + scale * B A x."""

    def __init__(self, base: nn.Linear, *, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scale * lora_out


def apply_lora_to_vit_qkv(
    backbone: nn.Module,
    *,
    r: int = 8,
    alpha: float = 16.0,
) -> int:
    """Wrap each block's ``attn.qkv`` Linear with LoRA. Returns #layers wrapped."""
    if not hasattr(backbone, "blocks"):
        raise ValueError("backbone has no .blocks — not a timm ViT?")
    n = 0
    for block in backbone.blocks:
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "qkv"):
            continue
        qkv = attn.qkv
        if isinstance(qkv, LoRALinear):
            continue
        if not isinstance(qkv, nn.Linear):
            continue
        attn.qkv = LoRALinear(qkv, r=r, alpha=alpha)
        n += 1
    return n


def lora_parameter_stats(backbone: nn.Module) -> dict:
    """Counts for verification: total / trainable / frozen-base vs LoRA A/B."""
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    lora_train = 0
    base_requires_grad = 0
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            lora_train += m.lora_A.numel() + m.lora_B.numel()
            if m.base.weight.requires_grad:
                base_requires_grad += 1
            if m.base.bias is not None and m.base.bias.requires_grad:
                base_requires_grad += 1
    return {
        "backbone_total": int(total),
        "backbone_trainable": int(trainable),
        "lora_trainable": int(lora_train),
        "trainable_pct": float(100.0 * trainable / max(1, total)),
        "base_qkv_still_trainable": int(base_requires_grad),
    }
