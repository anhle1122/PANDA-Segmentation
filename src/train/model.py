"""UNet++ segmentation model for baseline Gleason training."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_model(num_classes: int = 6) -> torch.nn.Module:
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        decoder_attention_type="scse",
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B4 + UNet++ (scse) | Trainable params: {n_params:,}")
    with torch.no_grad():
        out = model(torch.randn(1, 3, 512, 512))
        if out.shape != (1, num_classes, 512, 512):
            raise RuntimeError(f"Unexpected output shape: {out.shape}")
    return model
