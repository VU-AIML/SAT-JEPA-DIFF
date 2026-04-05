"""
IJEPA + Caption + SD Loss Ablation Study
==========================================
Runs ablation configurations covering IJEPA loss components, caption forecasting,
and SD conditioning to isolate the contribution of each component.

Usage:
    python ablation.py --fname s2_future_vith16.yaml [--epochs 30] [--output_dir ./ablation_results]

Each configuration trains from scratch with identical seeds, data splits,
and hyperparameters — only the loss weights and enabled components differ.

Configurations:
    A) L1 only                          (IJEPA baseline)
    B) L1 + Cosine                      (+ direction alignment)
    C) L1 + Cosine + Spatial            (+ variance preservation)
    D) L1 + Cosine + Contrastive        (+ instance discrimination)
    E) Full IJEPA                       (all IJEPA components)
    F) Full IJEPA + SD (no captions)    (+ diffusion loss, IJEPA tokens only)
    G) Full IJEPA + SD + Captions (no forecast)  (+ informative/geometric/semantic(t))
    H) Full IJEPA + SD + Captions + Forecast     (+ caption forecaster for semantic(t+1))
"""

import os
import sys
import copy
import math
import time
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── project imports ──────────────────────────────────────────────────────────
from data.data import S2FutureEmbeddingDataset
from masks.multiblock import MaskCollator as MBMaskCollator
from masks.utils import apply_masks
from utils.distributed import init_distributed
from utils.logging import AverageMeter
from utils.tensors import repeat_interleave_batch
from helper import init_model, init_opt

from sd_models import load_sd_model, encode_caption_batch
from sd_joint_loss import compute_sd_loss_and_metrics
from caption_forecaster import CaptionForecaster, CaptionForecastLoss

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ablation")

_GLOBAL_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────────────────────
# Format: (name, ijepa_weights, enable_sd, enable_captions, enable_caption_forecast)
# ijepa_weights = (l1, cosine, spatial, contrastive, feat_reg)

ABLATION_CONFIGS = [
    # IJEPA-only ablations (no SD, no captions)
    ("A: L1 only",
     (20.0, 0.0, 0.0, 0.0, 0.0), False, False, False),
    ("B: L1+Cos",
     (20.0, 2.0, 0.0, 0.0, 0.0), False, False, False),
    ("C: L1+Cos+Spatial",
     (20.0, 2.0, 2.0, 0.0, 0.0), False, False, False),
    ("D: L1+Cos+Contrastive",
     (20.0, 2.0, 0.0, 0.5, 0.0), False, False, False),
    ("E: Full IJEPA",
     (20.0, 2.0, 2.0, 0.5, 5.0), False, False, False),

    # SD + Caption ablations (all use full IJEPA loss)
    ("F: +SD (no captions)",
     (20.0, 2.0, 2.0, 0.5, 5.0), True, False, False),
    ("G: +SD+Captions (no forecast)",
     (20.0, 2.0, 2.0, 0.5, 5.0), True, True, False),
    ("H: +SD+Captions+Forecast (Ours)",
     (20.0, 2.0, 2.0, 0.5, 5.0), True, True, True),
]


# ─────────────────────────────────────────────────────────────────────────────
# HybridIJEPALoss (identical to train.py — kept here for self-containment)
# ─────────────────────────────────────────────────────────────────────────────
class HybridIJEPALoss(torch.nn.Module):
    def __init__(
        self,
        l1_weight=20.0,
        cosine_weight=1.0,
        contrastive_weight=1.0,
        spatial_var_weight=10.0,
        feature_reg_weight=5.0,
        temperature=0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        self.contrastive_weight = contrastive_weight
        self.spatial_var_weight = spatial_var_weight
        self.feature_reg_weight = feature_reg_weight
        self.temperature = temperature

    def forward(self, pred, target, pred_raw=None, target_raw=None):
        B, N, C = pred.shape
        losses = {}

        # L1
        l1_loss = F.l1_loss(pred, target)
        losses["l1"] = l1_loss

        # Cosine
        pred_n = F.normalize(pred, dim=-1, eps=1e-6)
        tgt_n = F.normalize(target, dim=-1, eps=1e-6)
        cos_sim = (pred_n * tgt_n).sum(dim=-1).mean()
        cosine_loss = 1.0 - cos_sim
        losses["cosine"] = cosine_loss
        losses["cos_sim"] = cos_sim

        # Spatial variance
        pred_std = torch.sqrt(pred.var(dim=1) + 1e-6).mean()
        tgt_std = torch.sqrt(target.var(dim=1) + 1e-6).mean()
        spatial_loss = F.mse_loss(pred_std, tgt_std)
        losses["spatial"] = spatial_loss

        # Log raw variance values for collapse detection
        losses["pred_var"] = pred.var(dim=1).mean()
        losses["tgt_var"] = target.var(dim=1).mean()

        # Contrastive
        if B > 1:
            pg = F.normalize(pred.mean(dim=1), dim=-1)
            tg = F.normalize(target.mean(dim=1), dim=-1)
            logits = pg @ tg.T / self.temperature
            labels = torch.arange(B, device=pred.device)
            contr_loss = F.cross_entropy(logits, labels)
            losses["contr"] = contr_loss
        else:
            contr_loss = torch.tensor(0.0, device=pred.device)
            losses["contr"] = contr_loss

        # Feature regression
        if pred_raw is not None and target_raw is not None:
            p_rn = F.layer_norm(pred_raw, (pred_raw.size(-1),))
            t_rn = F.layer_norm(target_raw, (target_raw.size(-1),))
            feat_loss = F.l1_loss(p_rn, t_rn)
            losses["feat_reg"] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=pred.device)
            losses["feat_reg"] = feat_loss

        total = (
            self.l1_weight * l1_loss
            + self.cosine_weight * cosine_loss
            + self.spatial_var_weight * spatial_loss
            + self.contrastive_weight * contr_loss
            + self.feature_reg_weight * feat_loss
        )
        return total, losses


# ─────────────────────────────────────────────────────────────────────────────
# Helper: patches from spatial embeddings
# ─────────────────────────────────────────────────────────────────────────────
def embedding_to_patches(embs, patch_size):
    B, C, H, W = embs.shape
    Hp, Wp = H // patch_size, W // patch_size
    embs = embs.view(B, C, Hp, patch_size, Wp, patch_size)
    embs = embs.mean(dim=(3, 5))
    return embs.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Collate for Caption-Aware Ablation
# ─────────────────────────────────────────────────────────────────────────────
def make_caption_collate(mask_collator):
    """
    Custom collate for caption-aware ablation.
    Dataset returns 6-tuple: (rgb_t, emb_tp1, rgb_tp1, meta, cap_emb_t, cap_emb_tp1)
    """
    def collate_fn(batch):
        rgb_t = torch.stack([item[0] for item in batch])
        emb_tp1 = torch.stack([item[1] for item in batch])
        rgb_tp1 = torch.stack([item[2] for item in batch])
        batch_data = (rgb_t, emb_tp1, rgb_tp1)

        B = len(batch)
        _, masks_enc, masks_pred = mask_collator(
            [(item[0], item[1], item[2]) for item in batch]
        )

        # Extract captions from meta dicts
        metas = [item[3] for item in batch]
        captions_t = {"informative": [], "geometric": [], "semantic": []}
        captions_tp1 = {"informative": [], "geometric": [], "semantic": []}
        for m in metas:
            ct = m.get("captions_t", {"informative": "", "geometric": "", "semantic": ""})
            ctp1 = m.get("captions_tp1", {"informative": "", "geometric": "", "semantic": ""})
            for k in ["informative", "geometric", "semantic"]:
                captions_t[k].append(ct.get(k, ""))
                captions_tp1[k].append(ctp1.get(k, ""))

        # Stack precomputed embeddings
        cap_emb_t = None
        cap_emb_tp1 = None
        emb_list_t = [item[4] for item in batch]
        emb_list_tp1 = [item[5] for item in batch]
        if emb_list_t[0] is not None:
            try:
                cap_emb_t = {k: torch.stack([e[k] for e in emb_list_t]) for k in emb_list_t[0].keys()}
            except Exception:
                cap_emb_t = None
        if emb_list_tp1[0] is not None:
            try:
                cap_emb_tp1 = {k: torch.stack([e[k] for e in emb_list_tp1]) for k in emb_list_tp1[0].keys()}
            except Exception:
                cap_emb_tp1 = None

        return batch_data, masks_enc, masks_pred, captions_t, captions_tp1, cap_emb_t, cap_emb_tp1

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Single ablation run
# ─────────────────────────────────────────────────────────────────────────────
def run_single_ablation(
    config_name, ijepa_weights, enable_sd, enable_captions, enable_caption_forecast,
    args, train_loader, val_loader,
    device, num_epochs, use_bfloat16,
):
    """
    Train IJEPA (+ optionally SD + captions) from scratch with given config.

    Args:
        config_name: str, human-readable name
        ijepa_weights: tuple (l1, cosine, spatial, contrastive, feat_reg)
        enable_sd: bool, whether to include SD loss
        enable_captions: bool, whether to use caption conditioning for SD
        enable_caption_forecast: bool, whether to forecast semantic caption at t+1
        args: full yaml config dict
        train_loader / val_loader: DataLoaders
        device: torch device
        num_epochs: int
        use_bfloat16: bool
    Returns:
        history: dict with per-epoch metrics
    """
    l1_w, cos_w, spat_w, contr_w, feat_w = ijepa_weights

    logger.info(f"\n{'='*70}")
    logger.info(f"  ABLATION: {config_name}")
    logger.info(f"  IJEPA Weights: L1={l1_w}, Cos={cos_w}, Spatial={spat_w}, "
                f"Contr={contr_w}, FeatReg={feat_w}")
    logger.info(f"  SD={enable_sd}, Captions={enable_captions}, Forecast={enable_caption_forecast}")
    logger.info(f"{'='*70}")

    # ── Reproducibility ──
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.cuda.manual_seed_all(_GLOBAL_SEED)

    # ── Parse config ──
    patch_size     = args["mask"]["patch_size"]
    crop_size      = args["data"]["crop_size"]
    pred_depth     = args["meta"]["pred_depth"]
    pred_emb_dim   = args["meta"]["pred_emb_dim"]
    target_emb_dim = args["meta"].get("target_emb_dim", 64)
    model_name     = args["meta"]["model_name"]

    sd_loss_weight          = float(args["meta"].get("sd_loss_weight", 1.0))
    ssim_weight             = float(args["meta"].get("ssim_weight", 0.1))
    caption_forecast_weight = float(args["meta"].get("caption_forecast_weight", 0.5))
    caption_dropout_prob    = float(args["meta"].get("caption_dropout_prob", 0.1))
    caption_forecaster_layers = int(args["meta"].get("caption_forecaster_layers", 3))
    caption_forecaster_hidden = int(args["meta"].get("caption_forecaster_hidden", 1024))

    use_lora   = args["meta"].get("use_lora", True)
    lora_rank  = int(args["meta"].get("lora_rank", 16))
    lora_alpha = int(args["meta"].get("lora_alpha", 32))

    wd        = float(args["optimization"]["weight_decay"])
    final_wd  = float(args["optimization"]["final_weight_decay"])
    start_lr  = args["optimization"]["start_lr"]
    lr        = args["optimization"]["lr"]
    final_lr  = args["optimization"]["final_lr"]
    warmup    = args["optimization"]["warmup"]
    ema_range = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]

    # ── Build IJEPA models ──
    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    enc_dim = getattr(encoder, "embed_dim", 768)
    proj_head = torch.nn.Linear(enc_dim, target_emb_dim).to(device)

    # ── IJEPA Loss ──
    ijepa_loss_fn = HybridIJEPALoss(
        l1_weight=l1_w,
        cosine_weight=cos_w,
        spatial_var_weight=spat_w,
        contrastive_weight=contr_w,
        feature_reg_weight=feat_w,
    ).to(device)

    # ── SD model (optional) ──
    sd_state = None
    if enable_sd:
        logger.info(f"  Loading Stable Diffusion for ablation config...")
        sd_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

        # Check if precomputed embeddings exist
        caption_dir = args["meta"].get("caption_dir", None)
        _has_precomputed = caption_dir and os.path.isdir(
            os.path.join(caption_dir, "caption_embeddings")
        )
        _load_text_enc = enable_captions and not _has_precomputed

        sd_state = load_sd_model(
            device=device,
            checkpoint_dir=args["meta"].get("sd_checkpoint_dir", "./sd_finetuned"),
            dtype=sd_dtype,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            hf_token=args["meta"].get("hf_token", None),
            target_emb_dim=enc_dim,
            load_text_encoders=_load_text_enc,
        )
        if "vae" in sd_state:
            sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    # ── Caption Forecaster (optional) ──
    caption_forecaster = None
    caption_forecast_loss_fn = None
    if enable_caption_forecast and enable_captions:
        logger.info(f"  Initializing Caption Forecaster...")
        caption_forecaster = CaptionForecaster(
            ijepa_dim=enc_dim,
            text_dim=4096,
            hidden_dim=caption_forecaster_hidden,
            num_layers=caption_forecaster_layers,
        ).to(device)
        caption_forecast_loss_fn = CaptionForecastLoss()
        cap_params = sum(p.numel() for p in caption_forecaster.parameters())
        logger.info(f"  CaptionForecaster: {cap_params:,} params")

    # ── Optimizer ──
    ipe = len(train_loader)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor,
        wd=wd, final_wd=final_wd,
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr,
        iterations_per_epoch=ipe, warmup=warmup,
        num_epochs=num_epochs, ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    optimizer.add_param_group({"params": proj_head.parameters()})

    if sd_state is not None and "trainable_params" in sd_state:
        optimizer.add_param_group({"params": sd_state["trainable_params"]})

    if caption_forecaster is not None:
        optimizer.add_param_group({"params": caption_forecaster.parameters()})

    momentum_scheduler = (
        ema_range[0] + i * (ema_range[1] - ema_range[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    # Cache unet dtype
    _unet_dtype = next(sd_state["unet"].parameters()).dtype if sd_state else None

    # ── History ──
    history = defaultdict(list)
    num_patches = (crop_size // patch_size) ** 2

    # ── Metric keys ──
    ijepa_meter_keys = [
        "loss", "l1", "cosine", "cos_sim", "spatial",
        "contr", "feat_reg", "pred_var", "tgt_var",
    ]
    sd_meter_keys = ["sd_loss", "sd_mse", "sd_ssim"]
    caption_meter_keys = [
        "caption_loss", "caption_mse", "caption_cos_sim", "caption_contr",
    ]
    all_meter_keys = ijepa_meter_keys + sd_meter_keys + caption_meter_keys

    # ── Helper: extract caption embeddings ──
    def get_caption_embeddings(cap_emb_t, cap_emb_tp1, captions_t, captions_tp1):
        """
        Extract caption hidden/pooled from precomputed or runtime encoding.
        Returns info_h, info_p, geo_h, geo_p, sem_t_h, sem_t_p, sem_tp1_gt_h
        """
        info_h, info_p = None, None
        geo_h, geo_p = None, None
        sem_t_h, sem_t_p = None, None
        sem_tp1_gt_h = None

        if cap_emb_t is not None:
            # Precomputed embeddings (fast path)
            info_h = cap_emb_t["informative_hidden"].to(device=device, dtype=_unet_dtype)
            info_p = cap_emb_t["informative_pooled"].to(device=device, dtype=_unet_dtype)
            geo_h = cap_emb_t["geometric_hidden"].to(device=device, dtype=_unet_dtype)
            geo_p = cap_emb_t["geometric_pooled"].to(device=device, dtype=_unet_dtype)
            sem_t_h = cap_emb_t["semantic_hidden"].to(device=device, dtype=_unet_dtype)
            sem_t_p = cap_emb_t["semantic_pooled"].to(device=device, dtype=_unet_dtype)
        elif captions_t is not None and sd_state is not None and sd_state.get("caption_encoder") is not None:
            # Runtime text encoding (slow path)
            cap_enc = sd_state["caption_encoder"]
            with torch.no_grad():
                info_h, info_p = encode_caption_batch(cap_enc, captions_t["informative"], device, _unet_dtype)
                geo_h, geo_p = encode_caption_batch(cap_enc, captions_t["geometric"], device, _unet_dtype)
                sem_t_h, sem_t_p = encode_caption_batch(cap_enc, captions_t["semantic"], device, _unet_dtype)

        # Ground truth semantic at t+1 (for caption forecast loss)
        if cap_emb_tp1 is not None:
            sem_tp1_gt_h = cap_emb_tp1["semantic_hidden"].to(device=device, dtype=_unet_dtype)
        elif captions_tp1 is not None and sd_state is not None and sd_state.get("caption_encoder") is not None:
            with torch.no_grad():
                sem_tp1_gt_h, _ = encode_caption_batch(
                    sd_state["caption_encoder"], captions_tp1["semantic"], device, _unet_dtype
                )

        return info_h, info_p, geo_h, geo_p, sem_t_h, sem_t_p, sem_tp1_gt_h

    # ── Training loop ──
    for epoch in range(num_epochs):
        encoder.train(); predictor.train(); proj_head.train()
        if caption_forecaster is not None:
            caption_forecaster.train()

        epoch_t0 = time.time()
        epoch_meters = {k: AverageMeter() for k in all_meter_keys}

        num_iters = len(train_loader)
        for itr, batch in enumerate(train_loader):
            # ── Unpack batch ──
            if len(batch) == 7:
                batch_data, masks_enc, masks_pred, captions_t, captions_tp1, cap_emb_t, cap_emb_tp1 = batch
            elif len(batch) == 3:
                batch_data, masks_enc, masks_pred = batch
                captions_t, captions_tp1, cap_emb_t, cap_emb_tp1 = None, None, None, None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            if not enable_captions:
                captions_t, captions_tp1, cap_emb_t, cap_emb_tp1 = None, None, None, None

            imgs_t   = batch_data[0].to(device, non_blocking=True)
            embs_tp1 = batch_data[1].to(device, non_blocking=True)
            rgb_tp1  = batch_data[2].to(device, non_blocking=True)
            masks_enc  = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            caption_loss = torch.tensor(0.0, device=device)
            caption_metrics = {}

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                # ── IJEPA Forward ──
                with torch.no_grad():
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),))
                    B_local = h.size(0)

                    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(B_local, -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)

                    h_masked = apply_masks(h, masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(masks_enc))

                    h_raw_masked = apply_masks(h_raw_all, masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(masks_enc))

                z_enc = encoder(imgs_t, masks_enc)
                z_raw = predictor(z_enc, masks_enc, masks_pred)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                z_proj = proj_head(z_raw_norm)

                loss_ijepa, ijepa_stats = ijepa_loss_fn(
                    pred=z_proj, target=h_masked,
                    pred_raw=z_raw_norm, target_raw=h_raw_masked,
                )

                # ── Caption Conditioning ──
                info_h, info_p = None, None
                geo_h, geo_p = None, None
                sem_h, sem_p = None, None

                if enable_captions and sd_state is not None:
                    (info_h, info_p, geo_h, geo_p,
                     sem_t_h, sem_t_p, sem_tp1_gt_h) = get_caption_embeddings(
                        cap_emb_t, cap_emb_tp1, captions_t, captions_tp1
                    )

                    if enable_caption_forecast and caption_forecaster is not None and sem_t_h is not None:
                        # Forecast semantic caption at t+1
                        sem_forecast_h = caption_forecaster(z_raw_norm.detach(), sem_t_h)
                        sem_h = sem_forecast_h
                        sem_p = sem_t_p

                        # Caption forecast loss
                        if sem_tp1_gt_h is not None and caption_forecast_loss_fn is not None:
                            caption_loss, caption_metrics = caption_forecast_loss_fn(
                                sem_forecast_h, sem_tp1_gt_h
                            )
                    else:
                        # No forecasting: use semantic(t) directly
                        sem_h, sem_p = sem_t_h, sem_t_p

                # ── SD Loss ──
                sd_loss = torch.tensor(0.0, device=device)
                sd_metrics = {}

                if enable_sd and sd_state is not None:
                    rgb_for_sd = rgb_tp1.to(device=device, dtype=torch.float32)

                    sd_loss, sd_metrics, _ = compute_sd_loss_and_metrics(
                        sd_state=sd_state,
                        ijepa_tokens=z_raw_norm,
                        rgb_target=rgb_for_sd,
                        step=epoch * num_iters + itr,
                        do_diffusion_sample=False,
                        ssim_weight=ssim_weight,
                        informative_hidden=info_h,
                        informative_pooled=info_p,
                        geometric_hidden=geo_h,
                        geometric_pooled=geo_p,
                        semantic_hidden=sem_h,
                        semantic_pooled=sem_p,
                        caption_dropout_prob=caption_dropout_prob,
                    )

                # ── Total Loss ──
                total_loss = loss_ijepa
                if enable_sd:
                    total_loss = total_loss + sd_loss_weight * sd_loss
                if enable_caption_forecast:
                    total_loss = total_loss + caption_forecast_weight * caption_loss

            # ── Backward ──
            scheduler.step()
            wd_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            all_params = (
                list(encoder.parameters())
                + list(predictor.parameters())
                + list(proj_head.parameters())
            )
            if sd_state is not None and "trainable_params" in sd_state:
                all_params.extend(sd_state["trainable_params"])
            if caption_forecaster is not None:
                all_params.extend(list(caption_forecaster.parameters()))

            if use_bfloat16:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
            else:
                if scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()

            # ── EMA ──
            with torch.no_grad():
                m = next(momentum_scheduler)
                for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)

            # ── Record meters ──
            epoch_meters["loss"].update(total_loss.item())
            for k in ["l1", "cosine", "cos_sim", "spatial", "contr", "feat_reg", "pred_var", "tgt_var"]:
                v = ijepa_stats.get(k, torch.tensor(0.0))
                epoch_meters[k].update(v.item() if torch.is_tensor(v) else v)

            epoch_meters["sd_loss"].update(sd_loss.item() if torch.is_tensor(sd_loss) else sd_loss)
            epoch_meters["sd_mse"].update(sd_metrics.get("mse_loss", 0.0))
            epoch_meters["sd_ssim"].update(sd_metrics.get("ssim_loss", 0.0))

            epoch_meters["caption_loss"].update(caption_loss.item() if torch.is_tensor(caption_loss) else 0.0)
            for ck in ["caption_mse", "caption_cos_sim", "caption_contr"]:
                v = caption_metrics.get(ck, 0.0)
                epoch_meters[ck].update(v.item() if torch.is_tensor(v) else v)

            # Progress log
            if itr % 100 == 0:
                elapsed_ep = time.time() - epoch_t0
                eta_ep = elapsed_ep / (itr + 1) * (num_iters - itr - 1)
                extra = ""
                if enable_sd:
                    extra += f"  sd={epoch_meters['sd_loss'].avg:.4f}"
                if enable_caption_forecast:
                    extra += f"  cap={epoch_meters['caption_loss'].avg:.4f}"
                logger.info(
                    f"    [{config_name}] Ep {epoch+1} iter {itr}/{num_iters}  "
                    f"loss={epoch_meters['loss'].avg:.4f}  "
                    f"cos_sim={epoch_meters['cos_sim'].avg:.4f}"
                    f"{extra}  ETA={eta_ep/60:.1f}min"
                )

        epoch_elapsed = (time.time() - epoch_t0) / 60
        logger.info(f"    [{config_name}] Ep {epoch+1} train done in {epoch_elapsed:.1f} min")

        # ── Epoch logging ──
        for k, meter in epoch_meters.items():
            history[f"train_{k}"].append(meter.avg)

        # ── Validation ──
        encoder.eval(); predictor.eval(); proj_head.eval()
        if caption_forecaster is not None:
            caption_forecaster.eval()

        val_meters = {k: AverageMeter() for k in all_meter_keys}

        with torch.no_grad():
            for v_batch in val_loader:
                if len(v_batch) == 7:
                    v_batch_data, v_masks_enc, v_masks_pred, v_ct, v_ctp1, v_ce_t, v_ce_tp1 = v_batch
                elif len(v_batch) == 3:
                    v_batch_data, v_masks_enc, v_masks_pred = v_batch
                    v_ct, v_ctp1, v_ce_t, v_ce_tp1 = None, None, None, None
                else:
                    raise ValueError(f"Unexpected v_batch length: {len(v_batch)}")

                if not enable_captions:
                    v_ct, v_ctp1, v_ce_t, v_ce_tp1 = None, None, None, None

                imgs_t   = v_batch_data[0].to(device, non_blocking=True)
                embs_tp1 = v_batch_data[1].to(device, non_blocking=True)
                rgb_tp1  = v_batch_data[2].to(device, non_blocking=True)
                v_masks_enc  = [m.to(device) for m in v_masks_enc]
                v_masks_pred = [m.to(device) for m in v_masks_pred]

                dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                    # IJEPA
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),))
                    B_local = h.size(0)

                    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(B_local, -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)

                    h_masked = apply_masks(h, v_masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(v_masks_enc))
                    h_raw_masked = apply_masks(h_raw_all, v_masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(v_masks_enc))

                    z_enc = encoder(imgs_t, v_masks_enc)
                    z_raw = predictor(z_enc, v_masks_enc, v_masks_pred)
                    z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                    z_proj = proj_head(z_raw_norm)

                    loss_ijepa, ijepa_stats = ijepa_loss_fn(
                        pred=z_proj, target=h_masked,
                        pred_raw=z_raw_norm, target_raw=h_raw_masked,
                    )

                    # Caption conditioning for validation
                    v_info_h, v_info_p = None, None
                    v_geo_h, v_geo_p = None, None
                    v_sem_h, v_sem_p = None, None
                    v_caption_loss = torch.tensor(0.0, device=device)
                    v_caption_metrics = {}

                    if enable_captions and sd_state is not None:
                        (v_info_h, v_info_p, v_geo_h, v_geo_p,
                         v_sem_t_h, v_sem_t_p, v_sem_tp1_gt_h) = get_caption_embeddings(
                            v_ce_t, v_ce_tp1, v_ct, v_ctp1
                        )

                        if enable_caption_forecast and caption_forecaster is not None and v_sem_t_h is not None:
                            v_sem_forecast_h = caption_forecaster(z_raw_norm, v_sem_t_h)
                            v_sem_h = v_sem_forecast_h
                            v_sem_p = v_sem_t_p

                            if v_sem_tp1_gt_h is not None and caption_forecast_loss_fn is not None:
                                v_caption_loss, v_caption_metrics = caption_forecast_loss_fn(
                                    v_sem_forecast_h, v_sem_tp1_gt_h
                                )
                        else:
                            v_sem_h, v_sem_p = v_sem_t_h, v_sem_t_p

                    # SD loss for validation
                    v_sd_loss = torch.tensor(0.0, device=device)
                    v_sd_metrics = {}

                    if enable_sd and sd_state is not None:
                        rgb_for_sd = rgb_tp1.to(device=device, dtype=torch.float32)
                        v_sd_loss, v_sd_metrics, _ = compute_sd_loss_and_metrics(
                            sd_state=sd_state,
                            ijepa_tokens=z_raw_norm,
                            rgb_target=rgb_for_sd,
                            step=0,
                            do_diffusion_sample=False,
                            ssim_weight=ssim_weight,
                            informative_hidden=v_info_h,
                            informative_pooled=v_info_p,
                            geometric_hidden=v_geo_h,
                            geometric_pooled=v_geo_p,
                            semantic_hidden=v_sem_h,
                            semantic_pooled=v_sem_p,
                            caption_dropout_prob=0.0,  # No dropout during validation
                        )

                    v_total = loss_ijepa
                    if enable_sd:
                        v_total = v_total + sd_loss_weight * v_sd_loss
                    if enable_caption_forecast:
                        v_total = v_total + caption_forecast_weight * v_caption_loss

                val_meters["loss"].update(v_total.item())
                for k in ["l1", "cosine", "cos_sim", "spatial", "contr", "feat_reg", "pred_var", "tgt_var"]:
                    v = ijepa_stats.get(k, torch.tensor(0.0))
                    val_meters[k].update(v.item() if torch.is_tensor(v) else v)

                val_meters["sd_loss"].update(v_sd_loss.item() if torch.is_tensor(v_sd_loss) else 0.0)
                val_meters["sd_mse"].update(v_sd_metrics.get("mse_loss", 0.0))
                val_meters["sd_ssim"].update(v_sd_metrics.get("ssim_loss", 0.0))

                val_meters["caption_loss"].update(v_caption_loss.item() if torch.is_tensor(v_caption_loss) else 0.0)
                for ck in ["caption_mse", "caption_cos_sim", "caption_contr"]:
                    v = v_caption_metrics.get(ck, 0.0)
                    val_meters[ck].update(v.item() if torch.is_tensor(v) else v)

        for k, meter in val_meters.items():
            history[f"val_{k}"].append(meter.avg)

        # Variance ratio (collapse indicator)
        pv = val_meters["pred_var"].avg
        tv = val_meters["tgt_var"].avg
        var_ratio = pv / (tv + 1e-8)
        history["val_var_ratio"].append(var_ratio)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            extra = ""
            if enable_sd:
                extra += f"  sd={val_meters['sd_loss'].avg:.4f}"
            if enable_caption_forecast:
                extra += f"  cap={val_meters['caption_loss'].avg:.4f}  cap_cos={val_meters['caption_cos_sim'].avg:.4f}"
            logger.info(
                f"  [{config_name}] Ep {epoch+1:3d}/{num_epochs}  "
                f"train_loss={epoch_meters['loss'].avg:.4f}  "
                f"val_loss={val_meters['loss'].avg:.4f}  "
                f"cos_sim={val_meters['cos_sim'].avg:.4f}  "
                f"var_ratio={var_ratio:.4f}"
                f"{extra}"
            )

    # Cleanup
    del encoder, predictor, target_encoder, proj_head, optimizer
    if caption_forecaster is not None:
        del caption_forecaster
    if sd_state is not None:
        del sd_state
    torch.cuda.empty_cache()

    return dict(history)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

# Extended colour palette (colourblind-safe)
COLORS = [
    "#E24A33", "#348ABD", "#988ED5", "#FBC15E",
    "#8EBA42", "#FFB5B8", "#777777", "#2CA02C",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_ablation_curves(all_histories, output_dir):
    """Generate publication-quality training curves."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("IJEPA + Caption + SD Loss Ablation — Training Dynamics",
                 fontsize=15, fontweight="bold", y=0.98)

    metrics_to_plot = [
        ("val_loss",          "Total Validation Loss",          "lower is better"),
        ("val_cos_sim",       "Cosine Similarity (val)",        "higher is better"),
        ("val_var_ratio",     "Variance Ratio (pred/target)",   "≈1.0 is ideal"),
        ("val_l1",            "L1 Loss (val)",                  "lower is better"),
        ("val_spatial",       "Spatial Variance Loss (val)",    "lower is better"),
        ("val_pred_var",      "Predicted Embedding Var (val)",  "should match target"),
        ("val_sd_loss",       "SD Flow Matching Loss (val)",    "lower is better"),
        ("val_caption_loss",  "Caption Forecast Loss (val)",    "lower is better"),
        ("val_caption_cos_sim", "Caption Forecast Cos Sim (val)", "higher is better"),
    ]

    for ax, (metric, title, note) in zip(axes.flat, metrics_to_plot):
        has_data = False
        for idx, (name, hist) in enumerate(all_histories.items()):
            vals = hist.get(metric, [])
            if len(vals) == 0 or all(v == 0.0 for v in vals):
                continue
            has_data = True
            epochs = list(range(1, len(vals) + 1))
            ax.plot(
                epochs, vals,
                color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                markevery=max(1, len(epochs) // 10),
                markersize=4, linewidth=1.5, alpha=0.85,
                label=name,
            )

        # Reference line for variance ratio
        if metric == "val_var_ratio":
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(note, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        if not has_data:
            ax.text(0.5, 0.5, "N/A for selected configs",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray", style="italic")

    # Shared legend
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(handles, labels, loc="lower center", ncol=min(len(all_histories), 4),
               fontsize=8, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(output_dir, "ablation_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves → {path}")
    return path


def plot_collapse_figure(all_histories, output_dir):
    """
    Dedicated spatial-collapse figure:
    predicted embedding variance across epochs for each config.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spatial Collapse Analysis", fontsize=13, fontweight="bold")

    for idx, (name, hist) in enumerate(all_histories.items()):
        epochs = list(range(1, len(hist.get("val_pred_var", [])) + 1))
        if len(epochs) == 0:
            continue

        ax1.plot(epochs, hist["val_pred_var"],
                 color=COLORS[idx % len(COLORS)],
                 marker=MARKERS[idx % len(MARKERS)],
                 markevery=max(1, len(epochs) // 10),
                 markersize=4, linewidth=1.5, label=name)

        ax2.plot(epochs, hist["val_var_ratio"],
                 color=COLORS[idx % len(COLORS)],
                 marker=MARKERS[idx % len(MARKERS)],
                 markevery=max(1, len(epochs) // 10),
                 markersize=4, linewidth=1.5, label=name)

    # Target variance reference
    last_hist = list(all_histories.values())[-1]
    if "val_tgt_var" in last_hist and len(last_hist["val_tgt_var"]) > 0:
        ax1.axhline(y=last_hist["val_tgt_var"][-1], color="black",
                     linestyle="--", linewidth=1, alpha=0.7, label="Target var")

    ax1.set_title("Predicted Embedding Variance", fontsize=11)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Variance")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("Variance Ratio (pred / target)", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Ratio")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_collapse.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved collapse figure → {path}")
    return path


def plot_caption_figure(all_histories, output_dir):
    """
    Caption forecasting analysis: compares configs that use captions.
    """
    # Filter to configs that have caption data
    caption_histories = {
        name: hist for name, hist in all_histories.items()
        if any(v != 0.0 for v in hist.get("val_caption_loss", [0.0]))
    }

    if not caption_histories:
        logger.info("No caption data to plot — skipping caption figure.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Caption Forecasting Analysis", fontsize=13, fontweight="bold")

    for idx, (name, hist) in enumerate(caption_histories.items()):
        epochs = list(range(1, len(hist.get("val_caption_loss", [])) + 1))
        if len(epochs) == 0:
            continue

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax1.plot(epochs, hist["val_caption_loss"],
                 color=color, marker=marker,
                 markevery=max(1, len(epochs) // 10),
                 markersize=4, linewidth=1.5, label=name)

        if "val_caption_cos_sim" in hist:
            ax2.plot(epochs, hist["val_caption_cos_sim"],
                     color=color, marker=marker,
                     markevery=max(1, len(epochs) // 10),
                     markersize=4, linewidth=1.5, label=name)

    ax1.set_title("Caption Forecast Loss (val)", fontsize=11)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_title("Caption Forecast Cosine Similarity (val)", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cosine Similarity")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_caption.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved caption figure → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
def make_summary_table(all_histories, output_dir):
    """
    Produce CSV and LaTeX table of final-epoch validation metrics.
    """
    rows = []
    header = [
        "Config", "Val Loss↓", "L1↓", "Cos Sim↑",
        "Spatial↓", "Var Ratio", "Contr↓",
        "SD Loss↓", "Cap Loss↓", "Cap Cos↑",
    ]

    for name, hist in all_histories.items():
        def last(key, fmt=".4f"):
            vals = hist.get(key, [])
            if not vals or all(v == 0.0 for v in vals):
                return "—"
            return f"{vals[-1]:{fmt}}"

        row = [
            name,
            last("val_loss"),
            last("val_l1"),
            last("val_cos_sim"),
            last("val_spatial", ".6f"),
            last("val_var_ratio"),
            last("val_contr"),
            last("val_sd_loss"),
            last("val_caption_loss"),
            last("val_caption_cos_sim"),
        ]
        rows.append(row)

    # CSV
    csv_path = os.path.join(output_dir, "ablation_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    # LaTeX
    latex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        ncols = len(header)
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Ablation study of IJEPA + Caption + SD loss components on validation set. "
                "Var Ratio $\\approx 1.0$ indicates no spatial collapse. "
                "Cap Cos measures caption forecast quality.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{" + "l" + "c" * (ncols - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for i, row in enumerate(rows):
            # Add a midrule between IJEPA-only and SD configs
            if i == 5:
                f.write("\\midrule\n")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n\\end{table}\n")

    logger.info(f"Saved table → {csv_path}")
    logger.info(f"Saved LaTeX → {latex_path}")

    # Print table to console
    logger.info("\n" + "=" * 120)
    logger.info("  ABLATION RESULTS (Final Epoch)")
    logger.info("=" * 120)
    col_widths = [max(len(header[i]), max(len(r[i]) for r in rows)) + 2 for i in range(ncols)]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    logger.info(fmt.format(*header))
    logger.info("-" * sum(col_widths))
    for row in rows:
        logger.info(fmt.format(*row))
    logger.info("=" * 120)

    return csv_path, latex_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IJEPA + Caption + SD Loss Ablation Study")
    parser.add_argument("--fname", type=str, required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: use config value)")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="Directory for outputs")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config indices (0-7) or 'all' or 'ijepa' or 'full'")
    parser.add_argument("--subset_ratio", type=float, default=0.05,
                        help="Use fraction of training data (e.g. 0.3 = 30%%) for faster runs")
    cli = parser.parse_args()

    # Load config
    with open(cli.fname, "r") as f:
        args = yaml.full_load(f)

    num_epochs = cli.epochs if cli.epochs else args["optimization"]["epochs"]
    output_dir = cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Which configs to run
    if cli.configs == "all":
        configs_to_run = list(range(len(ABLATION_CONFIGS)))
    elif cli.configs == "ijepa":
        configs_to_run = [0, 1, 2, 3, 4]  # A-E only
    elif cli.configs == "full":
        configs_to_run = [4, 5, 6, 7]  # E-H only (incremental SD/caption)
    else:
        configs_to_run = [int(x.strip()) for x in cli.configs.split(",")]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    use_bfloat16 = args["meta"]["use_bfloat16"]
    use_captions = args["meta"].get("use_captions", True)

    # ── Data ──
    logger.info("Building data loaders (shared across all configs)...")

    patch_size  = args["mask"]["patch_size"]
    crop_size   = args["data"]["crop_size"]
    batch_size  = args["data"]["batch_size"]
    pin_mem     = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path   = args["data"]["root_path"]
    caption_dir = args["meta"].get("caption_dir", None)

    mask_collator = MBMaskCollator(
        input_size=crop_size, patch_size=patch_size,
        pred_mask_scale=args["mask"]["pred_mask_scale"],
        enc_mask_scale=args["mask"]["enc_mask_scale"],
        aspect_ratio=args["mask"]["aspect_ratio"],
        nenc=args["mask"]["num_enc_masks"],
        npred=args["mask"]["num_pred_masks"],
        allow_overlap=args["mask"]["allow_overlap"],
        min_keep=args["mask"]["min_keep"],
    )

    # Check if any config needs captions
    any_needs_captions = any(
        ABLATION_CONFIGS[i][3] for i in configs_to_run  # enable_captions flag
    )

    full_dataset = S2FutureEmbeddingDataset(
        root_dir=root_path, patch_size=crop_size, transform=None,
        caption_dir=caption_dir if (use_captions and any_needs_captions) else None,
    )

    np.random.seed(_GLOBAL_SEED)
    num_samples = len(full_dataset)
    indices = np.random.permutation(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    # Optionally subsample for faster ablation runs
    if cli.subset_ratio < 1.0:
        n_train = max(batch_size * 2, int(len(train_indices) * cli.subset_ratio))
        n_val = max(batch_size, int(len(val_indices) * cli.subset_ratio))
        train_indices = train_indices[:n_train]
        val_indices = val_indices[:n_val]
        logger.info(f"Subsampled to {cli.subset_ratio*100:.0f}%: {n_train} train, {n_val} val")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    # Choose collate: caption-aware if any config needs captions
    if any_needs_captions and use_captions:
        collate_fn = make_caption_collate(mask_collator)
    else:
        collate_fn = mask_collator

    train_loader = DataLoader(
        train_dataset, collate_fn=collate_fn, shuffle=True,
        batch_size=batch_size, drop_last=True,
        pin_memory=pin_mem, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=collate_fn, shuffle=False,
        batch_size=batch_size, drop_last=False,
        pin_memory=pin_mem, num_workers=num_workers,
    )

    logger.info(f"Dataset: {num_samples} total, {len(train_indices)} train, "
                f"{len(val_indices)} val, {len(train_loader)} iters/epoch")
    logger.info(f"Running {len(configs_to_run)} ablation configs × {num_epochs} epochs")
    logger.info(f"Captions available: {any_needs_captions and use_captions}")

    # ── Distributed init ──
    try:
        mp.set_start_method("spawn")
    except:
        pass
    init_distributed()

    # ── Run ablations ──
    all_histories = {}
    t_start = time.time()

    for cfg_idx in configs_to_run:
        name, ijepa_weights, en_sd, en_cap, en_forecast = ABLATION_CONFIGS[cfg_idx]
        t0 = time.time()

        history = run_single_ablation(
            config_name=name,
            ijepa_weights=ijepa_weights,
            enable_sd=en_sd,
            enable_captions=en_cap,
            enable_caption_forecast=en_forecast,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )

        elapsed = (time.time() - t0) / 60
        logger.info(f"  ✓ {name} completed in {elapsed:.1f} min")
        all_histories[name] = history

    total_time = (time.time() - t_start) / 3600
    logger.info(f"\nAll ablations done in {total_time:.2f} hours")

    # ── Save raw histories ──
    json_path = os.path.join(output_dir, "ablation_histories.json")
    with open(json_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Saved raw histories → {json_path}")

    # ── Generate outputs ──
    curves_path = plot_ablation_curves(all_histories, output_dir)
    collapse_path = plot_collapse_figure(all_histories, output_dir)
    caption_path = plot_caption_figure(all_histories, output_dir)
    csv_path, latex_path = make_summary_table(all_histories, output_dir)

    logger.info(f"\n{'='*70}")
    logger.info("  ABLATION STUDY COMPLETE")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Files:")
    logger.info(f"    - {curves_path}")
    logger.info(f"    - {collapse_path}")
    if caption_path:
        logger.info(f"    - {caption_path}")
    logger.info(f"    - {csv_path}")
    logger.info(f"    - {latex_path}")
    logger.info(f"    - {json_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()