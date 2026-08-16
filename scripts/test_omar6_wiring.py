"""CPU checks for Omar points 1-gate (documented), 6 (absolute burden + soft vs hard)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from train.grade_head import (  # noqa: E402
    derived_isup_ce_from_seg_probs,
    soft_isup_logits_from_seg_probs,
)


def _probs(*, p3: float, p4: float, p5: float) -> torch.Tensor:
    rest = 1.0 - p3 - p4 - p5
    assert rest >= -1e-6
    return torch.tensor([0.0, 0.0, max(rest, 0.0), p3, p4, p5], dtype=torch.float32)


def main() -> None:
    # Point 6: same G3/G4/G5 *ratio*, different tumor burden → different 1–5 logits.
    tiny = _probs(p3=0.005, p4=0.005, p5=0.0)  # 1% cancer, 50/50 G3/G4
    huge = _probs(p3=0.45, p4=0.45, p5=0.0)  # 90% cancer, same ratio
    lt = soft_isup_logits_from_seg_probs(tiny)
    lh = soft_isup_logits_from_seg_probs(huge)
    assert not torch.allclose(lt[1:], lh[1:]), "1–5 logits must carry tumor burden"
    assert int(lt.argmax()) == 0, "1% cancer should prefer ISUP0"
    assert int(lh.argmax()) != 0, "90% G3/G4 should not collapse to ISUP0"

    loss, hard, soft = derived_isup_ce_from_seg_probs(huge, 2)
    assert 0 <= hard <= 5 and 0 <= soft <= 5
    assert loss.ndim == 0
    print(
        "tiny_soft",
        int(lt.argmax()),
        "huge_soft",
        int(lh.argmax()),
        "huge_hard",
        hard,
        "agree",
        int(soft == hard),
    )

    # Decoder-chunk checkpoint: one loss on all 16, backward must hit weights
    # without keeping 16 graphs (Omar 5b memory trick, still full live set).
    from torch.utils.checkpoint import checkpoint

    conv = torch.nn.Conv2d(3, 6, kernel_size=1)
    x = torch.randn(16, 3, 8, 8)
    parts = []
    for s in range(0, 16, 4):
        parts.append(checkpoint(conv, x[s : s + 4], use_reentrant=False))
    torch.cat(parts, dim=0).mean().backward()
    assert conv.weight.grad is not None and float(conv.weight.grad.abs().sum()) > 0
    print("checkpoint_chunk_grads_ok")
    print("OMAR6_WIRING_OK")


if __name__ == "__main__":
    main()
