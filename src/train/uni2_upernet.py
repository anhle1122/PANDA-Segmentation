"""UNI2-h backbone + UPerNet semantic decoder for Gleason patch segmentation."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Matches SegTME / UNI2-UperNet practice: intermediate ViT blocks → FPN scales.
DEFAULT_OUT_INDICES = (5, 11, 17, 23)
DEFAULT_FPN_CHANNELS = (256, 512, 1024, 2048)
# 512 patches are not divisible by patch=14; pad to 14*37=518 (DINOv2-style).
DEFAULT_MODEL_SIZE = 518


def _uni2_timm_kwargs(*, img_size: int) -> dict:
    return {
        "img_size": img_size,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def _resample_pos_embed(state: dict, model: nn.Module) -> dict:
    """Interpolate a checkpoint's pos_embed onto this model's token grid.

    The released UNI2-h weights are for a 224px input (a 16x16 = 256 token
    grid), but we build the backbone at 518px (37x37 = 1369) so 512px patches
    divide evenly by patch_size=14. A plain load_state_dict would fail on the
    shape mismatch, so resample first -- the same thing timm's own pretrained
    loader does when a ViT is instantiated at a new resolution.

    ``no_embed_class=True`` in our timm kwargs means pos_embed covers only
    spatial tokens, so there are no prefix tokens to hold out.
    """
    key = "pos_embed"
    if key not in state or not hasattr(model, key):
        return state
    src = state[key]
    dst = getattr(model, key)
    if src.shape == dst.shape:
        return state

    from timm.layers import resample_abs_pos_embed

    old_tokens = src.shape[1]
    old_side = int(round(old_tokens**0.5))
    if old_side * old_side != old_tokens:
        raise ValueError(f"pos_embed has {old_tokens} tokens, not a square grid")
    new_grid = list(model.patch_embed.grid_size)
    state = dict(state)
    state[key] = resample_abs_pos_embed(
        src,
        new_size=new_grid,
        old_size=[old_side, old_side],
        num_prefix_tokens=0,
    )
    print(
        f"Resampled pos_embed {old_side}x{old_side} -> {new_grid[0]}x{new_grid[1]} "
        f"({tuple(src.shape)} -> {tuple(state[key].shape)})"
    )
    return state


def _decode_bn(channels: int) -> nn.BatchNorm2d:
    """Decode-head BN matching the project's UNI2+UPerNet training recipe.

    Teacher A and every working checkpoint in this repo were trained with
    ``track_running_stats=False``: batch statistics in both train and eval.
    Default BatchNorm (running stats) causes the train-flat / val-explode
    pattern seen when the backbone unfreezes under small per-GPU batches.
    """
    return nn.BatchNorm2d(channels, track_running_stats=False)


class PPM(nn.Module):
    """Pyramid Pooling Module (UPerNet)."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    _decode_bn(channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _decode_bn(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        priors = [x]
        for stage in self.stages:
            priors.append(
                F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            )
        return self.bottleneck(torch.cat(priors, dim=1))


class UPerNetHead(nn.Module):
    """UPerNet decode head: PPM on deepest map + FPN fusion."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = DEFAULT_FPN_CHANNELS,
        channels: int = 512,
        num_classes: int = 6,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.psp = PPM(in_channels[-1], channels, pool_scales=pool_scales)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, channels, kernel_size=1, bias=False),
                    _decode_bn(channels),
                    nn.ReLU(inplace=True),
                )
                for c in in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    _decode_bn(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels[:-1]
            ]
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, kernel_size=3, padding=1, bias=False),
            _decode_bn(channels),
            nn.ReLU(inplace=True),
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, inputs[:-1])]
        laterals.append(self.psp(inputs[-1]))

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])

        target = fpn_outs[0].shape[2:]
        fpn_outs = [
            F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            if f.shape[2:] != target
            else f
            for f in fpn_outs
        ]
        feats = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.cls_seg(feats)


class UNI2UPerNet(nn.Module):
    """
    UNI2-h (ViT-H/14-reg8) + UPerNet for semantic segmentation.

    Forward accepts ImageNet-normalized NCHW tensors at patch resolution (e.g. 512)
    and returns logits at the same spatial size.
    """

    def __init__(
        self,
        *,
        num_classes: int = 6,
        model_size: int = DEFAULT_MODEL_SIZE,
        out_indices: tuple[int, ...] = DEFAULT_OUT_INDICES,
        fpn_channels: tuple[int, ...] = DEFAULT_FPN_CHANNELS,
        head_channels: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if len(out_indices) != len(fpn_channels):
            raise ValueError("out_indices and fpn_channels must have the same length")
        self.model_size = int(model_size)
        self.out_indices = tuple(out_indices)
        self.patch_size = 14
        if self.model_size % self.patch_size != 0:
            raise ValueError(f"model_size must be divisible by {self.patch_size}")

        self.backbone = self._load_backbone(
            img_size=self.model_size,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
        embed_dim = int(self.backbone.embed_dim)
        self.projections = nn.ModuleList(
            [nn.Conv2d(embed_dim, c, kernel_size=1) for c in fpn_channels]
        )
        # Build a coarse pyramid: deepest stays, shallower get ×2 / ×4 / ×8 upsample.
        self.scale_factors = tuple(2 ** (len(fpn_channels) - 1 - i) for i in range(len(fpn_channels)))
        self.decode_head = UPerNetHead(
            in_channels=fpn_channels,
            channels=head_channels,
            num_classes=num_classes,
        )
        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def _load_backbone(
        *,
        img_size: int,
        pretrained: bool,
        checkpoint_path: str | Path | None,
    ) -> nn.Module:
        kwargs = _uni2_timm_kwargs(img_size=img_size)
        ckpt = Path(checkpoint_path) if checkpoint_path else None

        if ckpt is not None and ckpt.is_file():
            model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _resample_pos_embed(state, model)
            model.load_state_dict(state, strict=True)
            return model

        if not pretrained:
            return timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            try:
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: HF login failed ({type(exc).__name__}); trying cached weights")

        try:
            return timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load MahmoodLab/UNI2-h. Request access on Hugging Face, then "
                "export HF_TOKEN (or pass --uni2-checkpoint pytorch_model.bin)."
            ) from exc

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
        self._backbone_frozen = False

    def backbone_parameters(self):
        return self.backbone.parameters()

    def decoder_parameters(self):
        for p in self.projections.parameters():
            yield p
        for p in self.decode_head.parameters():
            yield p

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen backbone in eval for stable BN/dropout (ViT uses none for BN).
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def _pad_to_model_size(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        if h == self.model_size and w == self.model_size:
            return x, (h, w)
        if h > self.model_size or w > self.model_size:
            x = F.interpolate(
                x, size=(self.model_size, self.model_size), mode="bilinear", align_corners=False
            )
            return x, (h, w)
        pad_h = self.model_size - h
        pad_w = self.model_size - w
        # pad = (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x, (h, w)

    def _extract_pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Tokens shaped (B, N_patch, C) after stripping prefix tokens.
        feats = self.backbone.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=False,
            norm=True,
        )
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        maps: list[torch.Tensor] = []
        for feat, proj, scale in zip(feats, self.projections, self.scale_factors):
            # feat: (B, gh*gw, C)
            fm = feat.reshape(b, gh, gw, -1).permute(0, 3, 1, 2).contiguous()
            fm = proj(fm)
            if scale != 1:
                fm = F.interpolate(
                    fm, scale_factor=float(scale), mode="bilinear", align_corners=False
                )
            maps.append(fm)
        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Seg logits plus GAP of the deepest FPN map (for the ISUP grade head).

        Returns:
            logits: (B, num_classes, H, W) at original resolution
            feats:  (B, fpn_channels[-1]) global-average-pooled deepest map
        """
        x_in, (orig_h, orig_w) = self._pad_to_model_size(x)
        if self._backbone_frozen:
            with torch.no_grad():
                maps = self._extract_pyramid(x_in)
        else:
            maps = self._extract_pyramid(x_in)
        # Deepest projected map for slide-level grade features
        feats = maps[-1].mean(dim=(2, 3))
        logits = self.decode_head(maps)
        logits = F.interpolate(
            logits, size=(self.model_size, self.model_size), mode="bilinear", align_corners=False
        )
        if (orig_h, orig_w) != (self.model_size, self.model_size):
            if orig_h <= self.model_size and orig_w <= self.model_size:
                logits = logits[:, :, :orig_h, :orig_w]
            else:
                logits = F.interpolate(
                    logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
        return logits, feats


def disable_bn_running_stats(model: nn.Module) -> int:
    """Switch every BatchNorm in ``model`` to batch-statistics-only mode.

    The UNI2+UPerNet checkpoints in this project were trained with
    ``track_running_stats=False``, so their decode-head BatchNorms saved only
    weight/bias -- no running_mean/running_var/num_batches_tracked. Rebuilding
    with today's default BatchNorm and strict-loading such a checkpoint fails
    on the missing buffers; loading non-strictly is worse, because eval() would
    then normalize with freshly-initialized mean=0/var=1 and emit garbage.

    Dropping the buffers restores the trained behaviour exactly: with
    ``track_running_stats=False`` BatchNorm uses the current batch's statistics
    in BOTH train and eval mode. Note the consequence -- these models' outputs
    depend slightly on how patches are grouped into batches.

    Returns the number of BatchNorm modules converted.
    """
    converted = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None
            converted += 1
    return converted


def checkpoint_has_bn_running_stats(state_dict: dict) -> bool:
    return any("running_mean" in key for key in state_dict)


def build_uni2_upernet(
    num_classes: int = 6,
    *,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
    model_size: int = DEFAULT_MODEL_SIZE,
) -> UNI2UPerNet:
    model = UNI2UPerNet(
        num_classes=num_classes,
        model_size=model_size,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        checkpoint_path=checkpoint_path,
    )
    # Belt-and-suspenders: any BN that slipped through still matches the recipe.
    n_bn = disable_bn_running_stats(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"Model: UNI2-h + UPerNet | params trainable={n_train:,} / total={n_all:,} "
        f"| freeze_backbone={freeze_backbone} | model_size={model_size} "
        f"| decode_bn_batch_stats={n_bn}"
    )
    return model
