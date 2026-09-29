"""Fine-tune Opt3 UNI2+UPerNet+LoRA on SICAPv2 (live-all slide ISUP).

Init from R3 best (or any Opt3 ckpt). Patient fold Val{N} Train/Test.
Slide loss / grade head use **all** patches (no live=64 cap), chunked for mem.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_utils import OUTPUTS  # noqa: E402
from train.grade_head import (  # noqa: E402
    ISUPGradeHead,
    aggregate_softmax_probs,
    derived_isup_ce_from_seg_probs,
    grade_head_ce,
)
from train.losses import (  # noqa: E402
    segmentation_loss,
    sicap_merged_dice_loss,
    sicap_merged_nc_ce,
)
from train.sicap_metrics import (  # noqa: E402
    accumulate_sicap_cm,
    dice_from_sicap_cm,
    metrics_to_train_log,
)
from train.sicap_slide_bag import (  # noqa: E402
    DEFAULT_SICAP_ROOT,
    SicapPatchDataset,
    SicapSlideBagDataset,
    estimate_class_weights,
    load_patch_batch,
    slide_bag_collate,
    summarize_bags,
)
from train.uni2_upernet import DEFAULT_FPN_CHANNELS, build_uni2_upernet  # noqa: E402
from train_baseline import (  # noqa: E402
    broadcast_flag,
    cleanup_distributed,
    epoch_snapshot_path,
    is_main_process,
    load_checkpoint,
    restore_best_cancer_state,
    save_checkpoint,
    setup_distributed,
    should_early_stop,
    unwrap_model,
)
from train_uni2_opt3_slidebag import SegPlusGrade  # noqa: E402


def _segmentation_parts(args, logits, masks, weights, class_weights):
    """Return (total, ce, dice) for logging. SICAP merge does not ignore class 0."""
    if getattr(args, "sicap_nc_merge", False):
        ce = sicap_merged_nc_ce(
            logits,
            masks,
            weights,
            class_weights,
            adjacent_soft_alpha=args.adjacent_soft_alpha,
        )
        dice = sicap_merged_dice_loss(logits, masks)
        total = 0.5 * ce + 0.5 * dice
        return total, ce, dice
    total = segmentation_loss(
        logits,
        masks,
        weights,
        class_weights,
        adjacent_soft_alpha=args.adjacent_soft_alpha,
        include_benign_soft=args.include_benign_soft,
        sicap_nc_merge=False,
    )
    return total, total, total * 0.0


def train(args: argparse.Namespace) -> None:
    local_rank, rank, world_size, device = setup_distributed()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = OUTPUTS / "checkpoints" / f"uni2_upernet_{args.mode}_{args.run_tag}"
    if is_main_process(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if any(ckpt_dir.glob("epoch_*.pth")) and not args.resume:
            raise SystemExit(
                f"Refusing non-empty ckpt dir {ckpt_dir} without --resume "
                "(cold start needs empty dir)."
            )
    log_path = ckpt_dir / "training_log.csv"
    root = Path(args.sicap_root)

    train_ds = SicapSlideBagDataset(
        root, fold=args.fold, split="Train", augment=bool(args.augment), seed=args.seed
    )
    val_store = SicapSlideBagDataset(
        root, fold=args.fold, split="Test", augment=False, seed=args.seed
    ).base
    if args.max_val_patches and len(val_store) > args.max_val_patches:
        rng = np.random.default_rng(args.seed)
        keep = sorted(
            rng.choice(len(val_store), size=int(args.max_val_patches), replace=False).tolist()
        )
        val_store.stems = [val_store.stems[i] for i in keep]
        val_store.slide_ids = [val_store.slide_ids[i] for i in keep]
    val_ds = SicapPatchDataset(val_store)

    if is_main_process(rank):
        print(
            f"SICAP FT | fold={args.fold} root={root} | train {summarize_bags(train_ds)}",
            flush=True,
        )
        print(
            f"val_patches={len(val_ds)} λ_slide={args.lambda_slide} λ_grade={args.lambda_grade} "
            f"micro={args.micro_batch_size} live=ALL slides/ep={args.slides_per_epoch} "
            f"lora={args.lora} decode_norm={args.decode_norm}",
            flush=True,
        )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    nw = max(0, int(args.num_workers))
    dl_kw: dict = dict(
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        collate_fn=slide_bag_collate,
        pin_memory=True,
    )
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **dl_kw)
    val_nw = max(1, nw // 2) if nw > 0 else 0
    vkw: dict = dict(
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=val_nw,
        pin_memory=True,
    )
    if val_nw > 0:
        vkw["persistent_workers"] = True
        vkw["prefetch_factor"] = 2
    val_loader = DataLoader(val_ds, **vkw)

    class_weights = estimate_class_weights(train_ds.base, max_patches=2000, seed=args.seed).to(
        device
    )
    if is_main_process(rank):
        print(f"class_weights={class_weights.detach().cpu().tolist()}", flush=True)

    seg = build_uni2_upernet(
        num_classes=6,
        pretrained=True,
        freeze_backbone=args.freeze_backbone_epochs > 0,
        checkpoint_path=args.uni2_checkpoint or None,
        decode_norm=getattr(args, "decode_norm", "gn"),
    )
    if args.lora:
        from train.lora_vit import apply_lora_to_vit_qkv

        n_lora = apply_lora_to_vit_qkv(seg.backbone, r=8, alpha=16.0)
        if is_main_process(rank):
            print(f"LoRA QKV wraps={n_lora}", flush=True)
    if args.grad_checkpoint and hasattr(seg.backbone, "set_grad_checkpointing"):
        seg.backbone.set_grad_checkpointing(True)
    grade_head = ISUPGradeHead(DEFAULT_FPN_CHANNELS[-1], num_isup=6)
    model: nn.Module = SegPlusGrade(seg, grade_head).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    def _build_optim(lr: float, backbone_on: bool) -> torch.optim.Optimizer:
        core = unwrap_model(model)
        groups = [
            {
                "params": [p for p in core.seg.decoder_parameters() if p.requires_grad],
                "lr": lr,
            },
            {"params": list(core.grade_head.parameters()), "lr": lr},
        ]
        bb = [p for p in core.seg.backbone_parameters() if p.requires_grad]
        if bb:
            groups.append(
                {
                    "params": bb,
                    "lr": lr if not backbone_on else lr * args.backbone_lr_mult,
                }
            )
        return torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)

    backbone_frozen = args.freeze_backbone_epochs > 0
    optimizer = _build_optim(args.lr, backbone_on=not backbone_frozen)
    if is_main_process(rank) and args.lora:
        from train.lora_vit import LoRALinear, lora_parameter_stats

        stats = lora_parameter_stats(unwrap_model(model).seg.backbone)
        lora_ids = {
            id(p)
            for m in unwrap_model(model).seg.backbone.modules()
            if isinstance(m, LoRALinear)
            for p in (m.lora_A, m.lora_B)
        }
        n_lora_opt = sum(
            p.numel() for g in optimizer.param_groups for p in g["params"] if id(p) in lora_ids
        )
        print(
            f"LoRA stats={stats} | lora_params_in_optimizer={n_lora_opt} | frozen={backbone_frozen}",
            flush=True,
        )
        if n_lora_opt <= 0:
            raise RuntimeError("--lora on but 0 LoRA params in optimizer")
        core = unwrap_model(model)
        n_dec = sum(p.numel() for p in core.seg.decoder_parameters() if p.requires_grad)
        n_grade = sum(p.numel() for p in core.grade_head.parameters() if p.requires_grad)
        n_bb = sum(p.numel() for p in core.seg.backbone_parameters() if p.requires_grad)
        print(
            f"TRAINABLE decoder={n_dec} grade_head={n_grade} backbone_incl_lora={n_bb} "
            f"lora_in_optim={n_lora_opt}",
            flush=True,
        )
        if n_dec <= 0 or n_grade <= 0:
            raise RuntimeError("decoder or grade head missing from optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_cancer, last_improve_epoch = restore_best_cancer_state(ckpt_dir)
    patience = int(getattr(args, "early_stop_patience", 10))

    if args.init_checkpoint and Path(args.init_checkpoint).is_file() and not args.resume:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = unwrap_model(model).load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if is_main_process(rank):
            print(
                f"INIT_OK from {args.init_checkpoint} missing={len(missing)} "
                f"unexpected={len(unexpected)}",
                flush=True,
            )

    if args.resume and Path(args.resume).is_file():
        start_epoch = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        if is_main_process(rank):
            print(f"Resumed → epoch {start_epoch} best_cancer={best_cancer:.4f}", flush=True)

    if is_main_process(rank) and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "cancer_dice",
                    "mean_dice",
                    "L_pixel",
                    "L_slide",
                    "L_grade",
                    "L_ce",
                    "L_dice",
                    "nc_to_cancer",
                    "binary_cancer",
                    "lr",
                    "soft_hard_isup_agree",
                ]
            )

    micro = max(1, int(args.micro_batch_size))
    live_chunk = max(1, int(args.live_chunk))
    decoder_ckpt = bool(args.decoder_checkpoint)
    if is_main_process(rank):
        print(
            f"WIRING_OK live=ALL (not micro={micro}) chunk={live_chunk} "
            f"decoder_ckpt={int(decoder_ckpt)} fold={args.fold} "
            f"sicap_nc_merge={int(bool(args.sicap_nc_merge))} "
            f"include_benign_soft={int(bool(args.include_benign_soft))}",
            flush=True,
        )
        print(
            f"EARLY_STOP_OK patience={patience} metric=val_cancer_dice "
            f"last_improve={last_improve_epoch} best={best_cancer:.4f}",
            flush=True,
        )

    def _lambda_slide_now(epoch: int) -> float:
        if not args.lambda_slide_warmup:
            return float(args.lambda_slide)
        if epoch <= 2:
            return 0.0
        if epoch >= 6:
            return float(args.lambda_slide)
        return float(args.lambda_slide) * (epoch - 2) / 4.0

    for epoch in range(start_epoch, args.epochs + 1):
        lam_s = _lambda_slide_now(epoch)
        if is_main_process(rank):
            print(
                f"=== Epoch {epoch}: λ_slide={lam_s:.3f} λ_grade={args.lambda_grade} live=ALL ===",
                flush=True,
            )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if backbone_frozen and epoch > args.freeze_backbone_epochs:
            unwrap_model(model).unfreeze_backbone()
            backbone_frozen = False
            optimizer = _build_optim(args.lr, backbone_on=True)
            if is_main_process(rank):
                print(f"=== Epoch {epoch}: unfroze backbone ===", flush=True)

        model.train()
        t0 = time.time()
        running = {
            "loss": 0.0,
            "pixel": 0.0,
            "slide": 0.0,
            "grade": 0.0,
            "ce": 0.0,
            "dice": 0.0,
            "n": 0,
        }
        n_slides = 0
        isup_agree = 0
        isup_cmp_n = 0

        def _bn_safe(t: torch.Tensor) -> tuple[torch.Tensor, int]:
            n = int(t.shape[0])
            if n >= 2:
                return t, n
            return torch.cat([t, t], dim=0), n

        def _ddp_no_sync():
            if world_size > 1 and isinstance(model, DDP):
                return model.no_sync()
            return nullcontext()

        def _dummy_synced_backward() -> None:
            dummy = torch.zeros(2, 3, 512, 512, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_d, feats_d, grade_d = model(dummy)
                z = (
                    logits_d.float().sum() + feats_d.float().sum() + grade_d.float().sum()
                ) * 0.0
            scaler.scale(z).backward()

        for bag in train_loader:
            if args.slides_per_epoch and n_slides >= args.slides_per_epoch:
                break
            patch_idxs = [int(i) for i in bag["patch_indices"].tolist()]
            isup = int(bag["isup"].item())
            n_patches = len(patch_idxs)
            apply_slide_isup = n_patches >= int(args.min_slide_patches)

            if args.augment:
                from train.augmentations import sample_slide_aug_params

                train_ds.base.set_slide_aug_params(sample_slide_aug_params())
            else:
                train_ds.base.set_slide_aug_params(None)

            optimizer.zero_grad(set_to_none=True)
            logits_chunks: list[torch.Tensor] = []
            feat_chunks: list[torch.Tensor] = []
            pixel_loss_acc = 0.0
            ce_acc = 0.0
            dice_acc = 0.0
            l_slide_val = 0.0
            l_grade_val = 0.0
            slide_term_val = 0.0

            if n_patches == 0:
                _dummy_synced_backward()
            else:
                n_micro = int(math.ceil(n_patches / micro))
                for m_i in range(n_micro):
                    sl = slice(m_i * micro, min((m_i + 1) * micro, n_patches))
                    n_real = sl.stop - sl.start
                    images, masks, weights = load_patch_batch(
                        train_ds.base, patch_idxs[sl]
                    )
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    weights = weights.to(device, non_blocking=True)
                    imgs_b, _ = _bn_safe(images)
                    masks_b, _ = _bn_safe(masks)
                    weights_b, _ = _bn_safe(weights)
                    with _ddp_no_sync():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            logits, feats, _grade = model(imgs_b)
                            p_loss, p_ce, p_dice = _segmentation_parts(
                                args,
                                logits[:n_real],
                                masks_b[:n_real],
                                weights_b[:n_real],
                                class_weights,
                            )
                            scaled = p_loss * (n_real / n_patches)
                        scaler.scale(scaled).backward()
                    pixel_loss_acc += float(scaled.detach().item())
                    scale = n_real / n_patches
                    ce_acc += float(p_ce.detach().item()) * scale
                    dice_acc += float(p_dice.detach().item()) * scale
                    if apply_slide_isup:
                        logits_chunks.append(logits[:n_real].detach())
                        feat_chunks.append(feats[:n_real].detach())
                    del images, masks, weights, imgs_b, masks_b, weights_b

                if apply_slide_isup:
                    live_idxs = list(patch_idxs)
                    images_g, _, _ = load_patch_batch(train_ds.base, live_idxs)
                    images_g = images_g.to(device, non_blocking=True)
                    imgs_g, n_g = _bn_safe(images_g)
                    core = unwrap_model(model)

                    def _live_chunk_forward(x: torch.Tensor):
                        return core(x)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logit_parts: list[torch.Tensor] = []
                        feat_parts: list[torch.Tensor] = []
                        grade_parts: list[torch.Tensor] = []
                        for s in range(0, int(imgs_g.size(0)), live_chunk):
                            sl = imgs_g[s : s + live_chunk]
                            if decoder_ckpt:
                                lg, fg, gg = checkpoint(
                                    _live_chunk_forward,
                                    sl,
                                    use_reentrant=False,
                                    preserve_rng_state=True,
                                )
                            else:
                                lg, fg, gg = _live_chunk_forward(sl)
                            logit_parts.append(lg)
                            feat_parts.append(fg)
                            grade_parts.append(gg)
                        logits_g = torch.cat(logit_parts, dim=0)[:n_g]
                        feats_g = torch.cat(feat_parts, dim=0)[:n_g]
                        grade_live = torch.cat(grade_parts, dim=0)[:n_g]
                        del logit_parts, feat_parts, grade_parts
                        mean_live = torch.softmax(logits_g.float(), dim=1).mean(dim=(0, 2, 3))
                        bag_mean = aggregate_softmax_probs(logits_chunks).to(mean_live.device)
                        mean_probs = 0.5 * mean_live + 0.5 * bag_mean
                        l_slide, hard_isup, soft_isup = derived_isup_ce_from_seg_probs(
                            mean_probs, isup, min_area_pct=args.min_area_pct
                        )
                        isup_cmp_n += 1
                        if int(soft_isup) == int(hard_isup):
                            isup_agree += 1
                        feat_bag = torch.cat(feat_chunks, dim=0).mean(dim=0)
                        feat_mix = 0.5 * feats_g.mean(dim=0) + 0.5 * feat_bag
                        g_logits = 0.5 * grade_live.mean(dim=0) + 0.5 * unwrap_model(
                            model
                        ).grade_head(feat_mix)
                        l_grade = grade_head_ce(g_logits, isup)
                        slide_term = lam_s * l_slide + args.lambda_grade * l_grade
                    scaler.scale(slide_term).backward()
                    l_slide_val = float(l_slide.detach().item())
                    l_grade_val = float(l_grade.detach().item())
                    slide_term_val = float(slide_term.detach().item())
                    del images_g, imgs_g, logits_chunks, feat_chunks
                _dummy_synced_backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total = pixel_loss_acc + slide_term_val
            running["loss"] += total
            running["pixel"] += pixel_loss_acc
            running["slide"] += l_slide_val
            running["grade"] += l_grade_val
            running["ce"] += ce_acc
            running["dice"] += dice_acc
            running["n"] += 1
            n_slides += 1
            if is_main_process(rank):
                print(
                    f"BAG_LOSS n={n_slides} total={total:.6f} pix={pixel_loss_acc:.6f} "
                    f"ce={ce_acc:.6f} dice={dice_acc:.6f} slide={l_slide_val:.6f} "
                    f"grade={l_grade_val:.6f}",
                    flush=True,
                )
            if is_main_process(rank) and n_slides == 1 and device.type == "cuda":
                print(
                    f"peak_cuda_gb_after_bag1="
                    f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f}",
                    flush=True,
                )

        if world_size > 1:
            dist.barrier()

        eval_model = unwrap_model(model)
        eval_model.eval()
        cm = np.zeros((6, 6), dtype=np.int64)
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for images_v, masks_v, weights_v in val_loader:
                images_v = images_v.to(device, non_blocking=True)
                masks_v = masks_v.to(device, non_blocking=True)
                weights_v = weights_v.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits_v, _, _ = eval_model(images_v)
                    vloss, _, _ = _segmentation_parts(
                        args, logits_v, masks_v, weights_v, class_weights
                    )
                val_loss_sum += float(vloss.item())
                val_n += 1
                accumulate_sicap_cm(
                    cm,
                    logits_v.argmax(1).detach().cpu().numpy(),
                    masks_v.detach().cpu().numpy(),
                )
        val_ds.clear_open_handles()

        sicap_m = dice_from_sicap_cm(cm)
        metrics = metrics_to_train_log(sicap_m)
        cancer = float(metrics.get("cancer_dice", 0.0))
        mean_dice = float(metrics.get("mean_dice", 0.0))
        train_loss = running["loss"] / max(1, running["n"])
        val_loss = val_loss_sum / max(1, val_n)
        soft_hard_agree = float(isup_agree) / max(1, isup_cmp_n)
        scheduler.step()

        if is_main_process(rank):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.4f} "
                f"(pix={running['pixel']/max(1,running['n']):.4f} "
                f"ce={running['ce']/max(1,running['n']):.4f} "
                f"dice={running['dice']/max(1,running['n']):.4f} "
                f"slide={running['slide']/max(1,running['n']):.4f} "
                f"grade={running['grade']/max(1,running['n']):.4f}) "
                f"| val={val_loss:.4f} cancer={cancer:.4f} mean4={mean_dice:.4f} "
                f"NC={metrics['dice_0']:.4f} bin={metrics['binary_cancer']:.4f} "
                f"nc_to_ca={metrics['nc_to_cancer']:.4f} "
                f"| {time.time()-t0:.0f}s",
                flush=True,
            )
            print(
                f"SICAP_VAL_OK cancer={cancer:.6f} mean4={mean_dice:.6f} "
                f"ignore_index=off remap_1_2_to_0=1",
                flush=True,
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{cancer:.6f}",
                        f"{mean_dice:.6f}",
                        f"{running['pixel']/max(1,running['n']):.6f}",
                        f"{running['slide']/max(1,running['n']):.6f}",
                        f"{running['grade']/max(1,running['n']):.6f}",
                        f"{running['ce']/max(1,running['n']):.6f}",
                        f"{running['dice']/max(1,running['n']):.6f}",
                        f"{metrics['nc_to_cancer']:.6f}",
                        f"{metrics['binary_cancer']:.6f}",
                        lr,
                        f"{soft_hard_agree:.6f}",
                    ]
                )
            save_checkpoint(
                ckpt_dir / "latest.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics={"mean_dice": mean_dice, "cancer_dice": cancer},
                class_weights=class_weights,
                mode=args.mode,
            )
            snap = epoch_snapshot_path(ckpt_dir, epoch, cancer)
            if not snap.exists():
                save_checkpoint(
                    snap,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics={"mean_dice": mean_dice, "cancer_dice": cancer},
                    class_weights=class_weights,
                    mode=args.mode,
                )
            if cancer > best_cancer:
                best_cancer = cancer
                last_improve_epoch = epoch
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics={"mean_dice": mean_dice, "cancer_dice": cancer},
                    class_weights=class_weights,
                    mode=args.mode,
                )
            print(f"  saved {snap.name} | best_cancer={best_cancer:.4f}", flush=True)
            stop_now = should_early_stop(
                epoch=epoch,
                last_improve_epoch=last_improve_epoch,
                patience=patience,
            )
            if stop_now:
                print(
                    f"EARLY_STOP epoch={epoch} best_cancer={best_cancer:.4f} "
                    f"last_improve={last_improve_epoch} patience={patience}",
                    flush=True,
                )
        else:
            stop_now = False

        if world_size > 1:
            dist.barrier()
        stop_now = broadcast_flag(stop_now, device=device, world_size=world_size)
        if stop_now:
            break

    if is_main_process(rank):
        (ckpt_dir / "TRAINING_COMPLETE.txt").write_text(
            f"best_cancer={best_cancer}\n"
            f"last_improve_epoch={last_improve_epoch}\n"
            f"early_stop_patience={patience}\n",
            encoding="utf-8",
        )
        print(f"DONE — best SICAP-fold val cancer_dice={best_cancer:.4f}", flush=True)
    cleanup_distributed()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SICAPv2 fine-tune from Opt3 (live=all)")
    p.add_argument("--mode", default="raw", choices=["raw", "normalized", "normalized_ink_raw"])
    p.add_argument("--run-tag", default="opt3_sicap_ft_r3ep6_val1_liveall")
    p.add_argument("--sicap-root", type=str, default=str(DEFAULT_SICAP_ROOT))
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after this many epochs with no val cancer_dice improvement (0=off).",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--backbone-lr-mult", type=float, default=0.05)
    p.add_argument("--freeze-backbone-epochs", type=int, default=100)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--slides-per-epoch", type=int, default=0, help="0 = all train slides")
    p.add_argument("--max-val-patches", type=int, default=0, help="0 = all Test patches")
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lambda-slide", type=float, default=0.3)
    p.add_argument("--lambda-grade", type=float, default=0.3)
    p.add_argument("--lambda-slide-warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-slide-patches", type=int, default=5)
    p.add_argument("--min-area-pct", type=float, default=0.0)
    p.add_argument("--adjacent-soft-alpha", type=float, default=0.1)
    p.add_argument(
        "--include-benign-soft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PANDA benign↔G3 soft. Off for SICAP (0 is NC, not stroma).",
    )
    p.add_argument(
        "--sicap-nc-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SICAP-only: GT=0 uses -log(p0+p1+p2); Dice counts cancer-on-NC as FP.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uni2-checkpoint", type=str, default="")
    p.add_argument("--init-checkpoint", type=str, default="", help="R3 ep6 weights (cold start)")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decode-norm", default="gn", choices=["bn", "gn"])
    p.add_argument("--live-chunk", type=int, default=4)
    p.add_argument(
        "--decoder-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
