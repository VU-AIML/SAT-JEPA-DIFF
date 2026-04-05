# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# IJEPA + Stable Diffusion + Caption-Guided Training Script
#
# CAPTION-GUIDED ARCHITECTURE:
# 1. IJEPA encoder predicts visual tokens at t+1
# 2. CaptionForecaster predicts semantic caption at t+1
# 3. SD3.5 conditioned on: informative(t) + geometric(t) + semantic(t+1) + IJEPA tokens
# 4. No coarse RGB, no Panopticon — pure embedding + caption conditioning

import os
import copy
import logging
import math
import sys
import yaml
import faulthandler
from pathlib import Path

# Enable faulthandler to catch segfaults and print traceback
faulthandler.enable()
# Also dump to a file so we can see what happened after crash
_fault_file = open("/tmp/train_faulthandler.log", "w")
faulthandler.enable(file=_fault_file, all_threads=True)

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from data.data import S2FutureEmbeddingDataset
from masks.multiblock import MaskCollator as MBMaskCollator
from masks.utils import apply_masks
from utils.distributed import init_distributed, AllReduce
from utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter,
)
from utils.tensors import repeat_interleave_batch

from helper import load_checkpoint, init_model, init_opt
from sd_models import (
    load_sd_model, save_full_checkpoint, load_full_checkpoint,
    encode_caption_batch,
)
from sd_joint_loss import compute_sd_loss_and_metrics, diffusion_sample
from caption_forecaster import CaptionForecaster, CaptionForecastLoss
from embedding_validation import EmbeddingConsistencyValidator
from metrics import compute_caption_metrics, compute_all_metrics


# ============================================================================
# Global Settings
# ============================================================================
log_freq = 10
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", message="resource_tracker")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ============================================================================
# Hybrid IJEPA Loss
# ============================================================================

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

        l1_loss = F.l1_loss(pred, target)
        losses['l1'] = l1_loss

        pred_norm = F.normalize(pred, dim=-1, eps=1e-6)
        target_norm = F.normalize(target, dim=-1, eps=1e-6)
        cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        cosine_loss = 1.0 - cos_sim
        losses['cosine'] = cosine_loss
        losses['cos_sim'] = cos_sim

        pred_std = torch.sqrt(pred.var(dim=1) + 1e-6).mean()
        target_std = torch.sqrt(target.var(dim=1) + 1e-6).mean()
        spatial_loss = F.mse_loss(pred_std, target_std)
        losses['spatial'] = spatial_loss

        if B > 1:
            pred_global = F.normalize(pred.mean(dim=1), dim=-1)
            target_global = F.normalize(target.mean(dim=1), dim=-1)
            logits = torch.matmul(pred_global, target_global.T) / self.temperature
            labels = torch.arange(B, device=pred.device)
            contrastive_loss = F.cross_entropy(logits, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=pred.device)
        losses['contr'] = contrastive_loss

        if pred_raw is not None and target_raw is not None:
            p_raw_n = F.layer_norm(pred_raw, (pred_raw.size(-1),))
            t_raw_n = F.layer_norm(target_raw, (target_raw.size(-1),))
            feat_loss = F.l1_loss(p_raw_n, t_raw_n)
        else:
            feat_loss = torch.tensor(0.0, device=pred.device)
        losses['feat_reg'] = feat_loss

        total = (
            self.l1_weight * l1_loss
            + self.cosine_weight * cosine_loss
            + self.spatial_var_weight * spatial_loss
            + self.contrastive_weight * contrastive_loss
            + self.feature_reg_weight * feat_loss
        )
        return total, losses


# ============================================================================
# Utility Functions
# ============================================================================

def embedding_to_patches(embs, patch_size):
    B, C, H, W = embs.shape
    Hp, Wp = H // patch_size, W // patch_size
    embs = embs.view(B, C, Hp, patch_size, Wp, patch_size)
    embs = embs.mean(dim=(3, 5))
    embs = embs.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
    return embs


def visualize_tokens(tokens, size):
    B, N, C = tokens.shape
    h = w = int(math.sqrt(N))
    x = tokens.reshape(-1, C).float()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-5
    x_norm = (x - mean) / std
    try:
        U, S, V = torch.pca_lowrank(x_norm, q=3, center=False, niter=2)
        pca_proj = torch.matmul(x_norm, V[:, :3])
    except Exception:
        pca_proj = x_norm[:, :3]
    pca_proj = pca_proj.reshape(B, N, 3)
    pca_min = pca_proj.min(dim=1, keepdim=True)[0]
    pca_max = pca_proj.max(dim=1, keepdim=True)[0]
    pca_proj = (pca_proj - pca_min) / (pca_max - pca_min + 1e-6)
    vis = pca_proj.reshape(B, h, w, 3).permute(0, 3, 1, 2)
    if vis.shape[-2:] != (size, size):
        vis = F.interpolate(vis, size=(size, size), mode='nearest')
    return vis.clamp(0, 1)


def save_visualization_grid(rgb_t, emb_gt, emb_pred, rgb_tp1_gt, rgb_tp1_pred, save_path, crop_size=128, max_samples=4):
    B = min(rgb_t.shape[0], max_samples)
    rgb_t = rgb_t[:B].float().clamp(0, 1)
    rgb_tp1_gt = rgb_tp1_gt[:B].float().clamp(0, 1)
    rgb_tp1_pred = rgb_tp1_pred[:B].float().clamp(0, 1) if rgb_tp1_pred is not None else torch.zeros_like(rgb_tp1_gt)
    emb_gt_vis = visualize_tokens(emb_gt[:B], crop_size)
    emb_pred_vis = visualize_tokens(emb_pred[:B], crop_size)
    rows = []
    for b in range(B):
        row = torch.cat([rgb_t[b], emb_gt_vis[b], emb_pred_vis[b], rgb_tp1_gt[b], rgb_tp1_pred[b]], dim=2)
        rows.append(row)
    grid = torch.cat(rows, dim=1)
    save_image(grid, save_path)


# ============================================================================
# Custom Collate (handles captions from meta dict)
# ============================================================================

def make_caption_collate(mask_collator, dataset_ref=None):
    """
    Custom collate for caption-aware training.
    Dataset returns 6-tuple: (rgb_t, emb_tp1, rgb_tp1, meta, cap_emb_t, cap_emb_tp1)
    We stack tensors manually and use mask_collator only for mask generation.
    """
    def collate_fn(batch):
        # batch = list of 6-tuples from dataset

        # 1. Stack image tensors manually (guaranteed correct)
        rgb_t = torch.stack([item[0] for item in batch])
        emb_tp1 = torch.stack([item[1] for item in batch])
        rgb_tp1 = torch.stack([item[2] for item in batch])
        batch_data = (rgb_t, emb_tp1, rgb_tp1)

        # 2. Generate masks — mask_collator needs B and image size
        #    Call it with just the image tensors wrapped as expected
        B = len(batch)
        _, masks_enc, masks_pred = mask_collator(
            [(item[0], item[1], item[2]) for item in batch]
        )

        # 3. Extract captions from meta dicts
        metas = [item[3] for item in batch]
        captions_t = {"informative": [], "geometric": [], "semantic": []}
        captions_tp1 = {"informative": [], "geometric": [], "semantic": []}
        for m in metas:
            ct = m.get("captions_t", {"informative": "", "geometric": "", "semantic": ""})
            ctp1 = m.get("captions_tp1", {"informative": "", "geometric": "", "semantic": ""})
            for k in ["informative", "geometric", "semantic"]:
                captions_t[k].append(ct.get(k, ""))
                captions_tp1[k].append(ctp1.get(k, ""))

        # 4. Stack precomputed embeddings
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


# ============================================================================
# Main Training Function
# ============================================================================

def main(args, resume_preempt=False):

    # ========================================================================
    # Parse Configuration
    # ========================================================================

    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    target_emb_dim = args["meta"].get("target_emb_dim", 64)
    sd_loss_weight = float(args["meta"].get("sd_loss_weight", 1.0))
    ssim_weight = float(args["meta"].get("ssim_weight", 0.1))
    sd_checkpoint_dir = args["meta"].get("sd_checkpoint_dir", "./sd_finetuned")
    enable_ijepa = args["meta"].get("enable_ijepa", True)
    run_val_first = args["meta"].get("run_val_first", False)

    use_lora = args["meta"].get("use_lora", True)
    lora_rank = int(args["meta"].get("lora_rank", 16))
    lora_alpha = int(args["meta"].get("lora_alpha", 32))

    # Caption config
    use_captions = args["meta"].get("use_captions", True)
    caption_dir = args["meta"].get("caption_dir", None)
    caption_forecast_weight = float(args["meta"].get("caption_forecast_weight", 0.5))
    caption_dropout_prob = float(args["meta"].get("caption_dropout_prob", 0.1))
    caption_forecaster_layers = int(args["meta"].get("caption_forecaster_layers", 3))
    caption_forecaster_hidden = int(args["meta"].get("caption_forecaster_hidden", 1024))

    # IJEPA Loss Weights
    ijepa_l1_weight = float(args["meta"].get("ijepa_l1_weight", 20.0))
    ijepa_cosine_weight = float(args["meta"].get("ijepa_cosine_weight", 1.0))
    ijepa_contrastive_weight = float(args["meta"].get("ijepa_contrastive_weight", 1.0))
    ijepa_spatial_weight = float(args["meta"].get("ijepa_spatial_weight", 10.0))
    ijepa_feature_reg_weight = float(args["meta"].get("ijepa_feature_reg_weight", 5.0))

    # Data
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    crop_size = args["data"]["crop_size"]

    # Mask
    patch_size = args["mask"]["patch_size"]
    allow_overlap = args["mask"]["allow_overlap"]
    aspect_ratio = args["mask"]["aspect_ratio"]
    enc_mask_scale = args["mask"]["enc_mask_scale"]
    pred_mask_scale = args["mask"]["pred_mask_scale"]
    min_keep = args["mask"]["min_keep"]
    num_enc_masks = args["mask"]["num_enc_masks"]
    num_pred_masks = args["mask"]["num_pred_masks"]

    # Optimization
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]

    # Logging
    folder = args["logging"].get("folder", "./experiments/default")
    tag = args["logging"].get("write_tag", "ijepa_sd")
    vis_interval_iters = int(args["logging"].get("vis_interval_iters", 500))
    diffusion_vis_steps = int(args["logging"].get("diffusion_vis_steps", 20))

    # ========================================================================
    # Device
    # ========================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    os.makedirs(folder, exist_ok=True)
    vis_folder = os.path.join(folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    with open(os.path.join(folder, "config.yaml"), "w") as f:
        yaml.dump(args, f)

    # ========================================================================
    # Distributed
    # ========================================================================

    try:
        mp.set_start_method("spawn")
    except:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized rank {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and world_size > 1
    )

    # ========================================================================
    # Paths & Loggers
    # ========================================================================

    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    val_log_file = os.path.join(folder, f"{tag}_val_r{rank}.csv")
    best_path = os.path.join(folder, f"{tag}-best.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")

    load_path = None
    if load_model:
        if r_file:
            # Support both full path and just filename
            if os.path.isabs(r_file) and os.path.exists(r_file):
                load_path = r_file
            elif os.path.exists(os.path.join(folder, r_file)):
                load_path = os.path.join(folder, r_file)
            else:
                logger.warning(f"Checkpoint not found: {r_file} (tried full path and {folder})")
        if load_path is None:
            # Fallback: try best, then latest
            if os.path.exists(best_path):
                load_path = best_path
            elif os.path.exists(latest_path):
                load_path = latest_path

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"), ("%d", "itr"),
        ("%.5f", "loss"), ("%.5f", "loss_ijepa"), ("%.5f", "loss_sd"), ("%.5f", "loss_cap"),
        ("%.5f", "mask_enc"), ("%.5f", "mask_pred"), ("%d", "time_ms"),
    )
    val_csv_logger = CSVLogger(
        val_log_file,
        ("%d", "epoch"), ("%.5f", "val_loss"),
        ("%.5f", "val_ijepa"), ("%.5f", "val_sd"),
    )

    # ========================================================================
    # Build Models
    # ========================================================================

    logger.info("Building models...")

    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim, model_name=model_name,
    )

    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Encoder embed dim
    encoder_embed_dim = getattr(encoder, "embed_dim", 768)

    # Projection head
    proj_head = torch.nn.Linear(encoder_embed_dim, target_emb_dim).to(device)
    logger.info(f"proj_head: {encoder_embed_dim} -> {target_emb_dim}")

    # Stable Diffusion
    logger.info("Loading Stable Diffusion...")
    sd_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    # Check if precomputed embeddings exist — if so, skip loading text encoders entirely
    _has_precomputed = os.path.isdir(
        os.path.join(args["meta"].get("caption_dir", ""), "caption_embeddings")
    )
    _load_text_enc = use_captions and not _has_precomputed
    if _has_precomputed:
        logger.info("[VRAM] Precomputed embeddings found — skipping text encoder loading (~5GB saved)")

    sd_state = load_sd_model(
        device=device, checkpoint_dir=sd_checkpoint_dir, dtype=sd_dtype,
        use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
        hf_token=args["meta"].get("hf_token", None),
        target_emb_dim=encoder_embed_dim,
        load_text_encoders=_load_text_enc,
    )
    if "vae" in sd_state:
        sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    # ========================================================================
    # Caption Forecaster
    # ========================================================================

    caption_forecaster = None
    caption_forecast_loss_fn = None

    if use_captions:
        logger.info("Initializing Caption Forecaster...")
        caption_forecaster = CaptionForecaster(
            ijepa_dim=encoder_embed_dim,
            text_dim=4096,  # T5 hidden dim
            hidden_dim=caption_forecaster_hidden,
            num_layers=caption_forecaster_layers,
        ).to(device)
        caption_forecast_loss_fn = CaptionForecastLoss()
        cap_params = sum(p.numel() for p in caption_forecaster.parameters())
        logger.info(f"CaptionForecaster: {cap_params:,} params")

    # ========================================================================
    # Data Loaders
    # ========================================================================

    logger.info("Building data loaders...")

    mask_collator = MBMaskCollator(
        input_size=crop_size, patch_size=patch_size,
        pred_mask_scale=pred_mask_scale, enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio, nenc=num_enc_masks, npred=num_pred_masks,
        allow_overlap=allow_overlap, min_keep=min_keep,
    )

    full_dataset = S2FutureEmbeddingDataset(
        root_dir=root_path, patch_size=crop_size, transform=None,
        caption_dir=caption_dir if use_captions else None,
    )

    num_samples = len(full_dataset)
    indices = np.random.permutation(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_dataset = Subset(full_dataset, indices[:split])
    val_dataset = Subset(full_dataset, indices[split:])

    collate_fn = make_caption_collate(mask_collator) if use_captions else mask_collator

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, collate_fn=collate_fn, sampler=train_sampler,
        shuffle=(train_sampler is None), batch_size=batch_size,
        drop_last=True, pin_memory=pin_mem, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False  
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=collate_fn, sampler=val_sampler,
        shuffle=False, batch_size=batch_size, drop_last=False,
        pin_memory=pin_mem, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False  
    )

    ipe = len(train_loader)
    logger.info(f"Dataset: {num_samples} samples, {ipe} iters/epoch")

    # ========================================================================
    # Optimizer
    # ========================================================================

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor,
        wd=wd, final_wd=final_wd, start_lr=start_lr, ref_lr=lr,
        final_lr=final_lr, iterations_per_epoch=ipe, warmup=warmup,
        num_epochs=num_epochs, ipe_scale=ipe_scale, use_bfloat16=use_bfloat16,
    )

    optimizer.add_param_group({"params": proj_head.parameters()})

    if "trainable_params" in sd_state and len(sd_state["trainable_params"]) > 0:
        optimizer.add_param_group({"params": sd_state["trainable_params"]})
        logger.info(f"Added {len(sd_state['trainable_params'])} SD trainable params")

    if caption_forecaster is not None:
        optimizer.add_param_group({"params": caption_forecaster.parameters()})
        logger.info("Added CaptionForecaster params to optimizer")

    # ========================================================================
    # DDP
    # ========================================================================

    if distributed:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        proj_head = DistributedDataParallel(proj_head, static_graph=True)
        if caption_forecaster is not None:
            caption_forecaster = DistributedDataParallel(caption_forecaster, static_graph=True)

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Loss
    ijepa_loss_fn = HybridIJEPALoss(
        l1_weight=ijepa_l1_weight, cosine_weight=ijepa_cosine_weight,
        contrastive_weight=ijepa_contrastive_weight,
        spatial_var_weight=ijepa_spatial_weight,
        feature_reg_weight=ijepa_feature_reg_weight,
    )

    # Cache dtype once (avoid repeated next(params).dtype in every iteration)
    _unet_dtype = next(sd_state["unet"].parameters()).dtype

    # ========================================================================
    # Forward Pass
    # ========================================================================

    def forward_pass(
        imgs_t, embs_tp1, rgb_tp1, masks_enc, masks_pred,
        captions_t=None, captions_tp1=None,
        cap_emb_t=None, cap_emb_tp1=None,
        step=0, compute_sd=True,
    ):
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        caption_loss = torch.tensor(0.0, device=device)
        caption_metrics = {}

        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):

            # === IJEPA Forward ===
            if enable_ijepa:
                with torch.no_grad():
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),))

                    num_patches = (crop_size // patch_size) ** 2
                    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(imgs_t.size(0), -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)

                    B_local = h.size(0)
                    h_masked = apply_masks(h, masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(masks_enc))
                    h_raw_masked = apply_masks(h_raw_all, masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(masks_enc))

                z_enc = encoder(imgs_t, masks_enc)
                z_raw = predictor(z_enc, masks_enc, masks_pred)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                z_proj = proj_head(z_raw_norm)

                loss_ijepa, loss_stats = ijepa_loss_fn(
                    pred=z_proj, target=h_masked,
                    pred_raw=z_raw_norm, target_raw=h_raw_masked,
                )
                if distributed:
                    loss_ijepa = AllReduce.apply(loss_ijepa)
            else:
                B = imgs_t.size(0)
                num_patches = (crop_size // patch_size) ** 2
                edim = encoder.module.embed_dim if distributed else encoder.embed_dim
                z_raw_norm = torch.randn(B, num_patches, edim, device=device, dtype=dtype)
                loss_ijepa = torch.tensor(0.0, device=device)
                loss_stats = {k: torch.tensor(0.0) for k in ['l1', 'cosine', 'contr', 'spatial', 'feat_reg']}

            # === Caption Embeddings ===
            # Priority: precomputed .pt tensors > runtime text encoding > none
            info_h, info_p = None, None
            geo_h, geo_p = None, None
            sem_h, sem_p = None, None
            sem_t_h, sem_t_p = None, None

            if use_captions:
                unet_dtype = _unet_dtype

                if cap_emb_t is not None:
                    # === FAST PATH: precomputed embeddings (no text encoder) ===
                    info_h = cap_emb_t["informative_hidden"].to(device=device, dtype=unet_dtype)
                    info_p = cap_emb_t["informative_pooled"].to(device=device, dtype=unet_dtype)
                    geo_h = cap_emb_t["geometric_hidden"].to(device=device, dtype=unet_dtype)
                    geo_p = cap_emb_t["geometric_pooled"].to(device=device, dtype=unet_dtype)
                    sem_t_h = cap_emb_t["semantic_hidden"].to(device=device, dtype=unet_dtype)
                    sem_t_p = cap_emb_t["semantic_pooled"].to(device=device, dtype=unet_dtype)

                elif captions_t is not None and sd_state.get("caption_encoder") is not None:
                    # === SLOW PATH: runtime text encoding (fallback if no precomputed) ===
                    cap_enc = sd_state["caption_encoder"]
                    with torch.no_grad():
                        info_h, info_p = encode_caption_batch(cap_enc, captions_t["informative"], device, unet_dtype)
                        geo_h, geo_p = encode_caption_batch(cap_enc, captions_t["geometric"], device, unet_dtype)
                        sem_t_h, sem_t_p = encode_caption_batch(cap_enc, captions_t["semantic"], device, unet_dtype)

                # Forecast semantic caption at t+1
                # REUSE z_raw_norm from IJEPA forward above — no 2nd encoder call!
                if caption_forecaster is not None and sem_t_h is not None:
                    sem_forecast_h = caption_forecaster(z_raw_norm.detach(), sem_t_h)
                    sem_h = sem_forecast_h
                    sem_p = sem_t_p

                    # Caption forecast loss
                    sem_tp1_gt_h = None
                    if cap_emb_tp1 is not None:
                        sem_tp1_gt_h = cap_emb_tp1["semantic_hidden"].to(device=device, dtype=unet_dtype)
                    elif captions_tp1 is not None and sd_state.get("caption_encoder") is not None:
                        with torch.no_grad():
                            sem_tp1_gt_h, _ = encode_caption_batch(
                                sd_state["caption_encoder"], captions_tp1["semantic"], device, unet_dtype
                            )

                    if sem_tp1_gt_h is not None:
                        caption_loss, caption_metrics = caption_forecast_loss_fn(
                            sem_forecast_h, sem_tp1_gt_h
                        )
                else:
                    sem_h, sem_p = sem_t_h, sem_t_p

            # === SD Loss ===
            if compute_sd:
                rgb_for_sd = rgb_tp1.to(device=device, dtype=torch.float32)

                sd_loss, sd_metrics, _ = compute_sd_loss_and_metrics(
                    sd_state=sd_state,
                    ijepa_tokens=z_raw_norm,
                    rgb_target=rgb_for_sd,
                    step=step,
                    do_diffusion_sample=False,
                    ssim_weight=ssim_weight,
                    informative_hidden=info_h, informative_pooled=info_p,
                    geometric_hidden=geo_h, geometric_pooled=geo_p,
                    semantic_hidden=sem_h, semantic_pooled=sem_p,
                    caption_dropout_prob=caption_dropout_prob,
                )
            else:
                sd_loss = torch.tensor(0.0, device=device)
                sd_metrics = {}

            total_loss = loss_ijepa + sd_loss_weight * sd_loss + caption_forecast_weight * caption_loss

        # Logging
        sd_metrics['l1'] = loss_stats['l1'].item() if torch.is_tensor(loss_stats['l1']) else loss_stats['l1']
        sd_metrics['feat_reg'] = loss_stats.get('feat_reg', torch.tensor(0)).item() if torch.is_tensor(loss_stats.get('feat_reg', 0)) else 0
        sd_metrics['cos'] = loss_stats['cosine'].item() if torch.is_tensor(loss_stats['cosine']) else loss_stats['cosine']
        sd_metrics['contr'] = loss_stats['contr'].item() if torch.is_tensor(loss_stats['contr']) else loss_stats['contr']
        sd_metrics['spatial'] = loss_stats['spatial'].item() if torch.is_tensor(loss_stats['spatial']) else loss_stats['spatial']
        sd_metrics['cap_loss'] = caption_loss.item() if torch.is_tensor(caption_loss) else 0
        for k, v in caption_metrics.items():
            sd_metrics[k] = v.item() if torch.is_tensor(v) else v

        return total_loss, loss_ijepa, sd_loss, caption_loss, sd_metrics

    # ========================================================================
    # Visualization
    # ========================================================================

    def generate_visualization(imgs_t, embs_tp1, rgb_tp1, captions_t=None):
        B = imgs_t.size(0)
        num_patches = (crop_size // patch_size) ** 2
        full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)]
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                z_enc = encoder(imgs_t, full_mask)
                z_raw = predictor(z_enc, full_mask, full_mask)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                z_proj = proj_head(z_raw_norm)
                h = embedding_to_patches(embs_tp1, patch_size)
                h = F.layer_norm(h, (h.size(-1),))

            z_for_sd = z_raw_norm.to(dtype=adapter_dtype)

            # Encode captions if available (on CPU, then move to GPU)
            info_h, info_p, geo_h, geo_p, sem_h, sem_p = None, None, None, None, None, None
            if use_captions and captions_t is not None and sd_state.get("caption_encoder"):
                cap_enc = sd_state["caption_encoder"]
                unet_dtype = _unet_dtype
                info_h, info_p = encode_caption_batch(cap_enc, captions_t["informative"], device, unet_dtype)
                geo_h, geo_p = encode_caption_batch(cap_enc, captions_t["geometric"], device, unet_dtype)
                sem_t_h, sem_t_p = encode_caption_batch(cap_enc, captions_t["semantic"], device, unet_dtype)
                if caption_forecaster is not None and sem_t_h is not None:
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                        sem_h = caption_forecaster(z_raw_norm, sem_t_h)
                    sem_p = sem_t_p
                else:
                    sem_h, sem_p = sem_t_h, sem_t_p

            rgb_pred = diffusion_sample(
                unet=sd_state["unet"], vae=sd_state["vae"],
                scheduler=sd_state["noise_scheduler"],
                cond_adapter=sd_state["cond_adapter"],
                ijepa_tokens=z_for_sd,
                num_steps=diffusion_vis_steps,
                image_size=(crop_size, crop_size),
                device=device,
                informative_hidden=info_h, informative_pooled=info_p,
                geometric_hidden=geo_h, geometric_pooled=geo_p,
                semantic_hidden=sem_h, semantic_pooled=sem_p,
                ref_rgb=imgs_t,
            )

        return z_proj, h, rgb_pred

    # ========================================================================
    # Checkpoint
    # ========================================================================

    def save_checkpoint(epoch, val_loss, is_best=False):
        if rank == 0:
            save_full_checkpoint(
                save_path=latest_path, encoder=encoder, predictor=predictor,
                proj_head=proj_head, target_encoder=target_encoder,
                sd_state=sd_state, optimizer=optimizer, scaler=scaler,
                epoch=epoch, best_val_loss=val_loss, config=args,
                caption_forecaster=caption_forecaster,
            )
            if is_best:
                save_full_checkpoint(
                    save_path=best_path, encoder=encoder, predictor=predictor,
                    proj_head=proj_head, target_encoder=target_encoder,
                    sd_state=sd_state, optimizer=optimizer, scaler=scaler,
                    epoch=epoch, best_val_loss=val_loss, config=args,
                    caption_forecaster=caption_forecaster,
                )
                logger.info(f"New best model! val_loss={val_loss:.4f}")

    # ========================================================================
    # Load Checkpoint
    # ========================================================================

    if load_model and load_path and os.path.exists(load_path):
        logger.info(f"Loading checkpoint: {load_path}")
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device, r_path=load_path, encoder=encoder,
            predictor=predictor, target_encoder=target_encoder,
            opt=optimizer, scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            global_step += 1

    # ========================================================================
    # Embedding Validator
    # ========================================================================

    embedding_validator = None
    if rank == 0:
        emb_val_dir = os.path.join(folder, "embedding_validation")
        enc_unwrap = encoder.module if distributed else encoder
        pred_unwrap = predictor.module if distributed else predictor
        tgt_unwrap = target_encoder.module if distributed else target_encoder
        embedding_validator = EmbeddingConsistencyValidator(
            encoder=enc_unwrap, predictor=pred_unwrap, target_encoder=tgt_unwrap,
            sd_state=sd_state, device=device, output_dir=emb_val_dir,
            diffusion_steps=diffusion_vis_steps,
        )

    # ========================================================================
    # Health Check — test everything before committing to long training
    # ========================================================================

    logger.info("\n" + "=" * 60)
    logger.info("HEALTH CHECK — testing all components before training...")
    logger.info("=" * 60)

    try:
        # 1. Data loading
        hc_batch = next(iter(train_loader))
        if use_captions:
            hc_data, hc_me, hc_mp, hc_ct, hc_ctp1, hc_ce_t, hc_ce_tp1 = hc_batch
        else:
            hc_data, hc_me, hc_mp = hc_batch
            hc_ct, hc_ctp1, hc_ce_t, hc_ce_tp1 = None, None, None, None
        hc_imgs = hc_data[0].to(device)
        hc_embs = hc_data[1].to(device)
        hc_rgb = hc_data[2].to(device)
        hc_me = [m.to(device) for m in hc_me]
        hc_mp = [m.to(device) for m in hc_mp]
        logger.info("[HC] ✓ Data loading + collate")

        # 2. Forward pass (train mode)
        encoder.train(); predictor.train(); proj_head.train()
        if caption_forecaster: caption_forecaster.train()
        tl, lij, lsd, lcap, hc_m = forward_pass(
            hc_imgs, hc_embs, hc_rgb, hc_me, hc_mp,
            captions_t=hc_ct, captions_tp1=hc_ctp1,
            cap_emb_t=hc_ce_t, cap_emb_tp1=hc_ce_tp1,
            step=0, compute_sd=True,
        )
        logger.info(f"[HC] ✓ Forward pass (loss={tl.item():.4f} ij={lij.item():.4f} sd={lsd.item():.4f} cap={lcap.item():.4f})")

        # 3. Backward
        optimizer.zero_grad(set_to_none=True)
        tl.backward()
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
        optimizer.step()
        logger.info("[HC] ✓ Backward + optimizer step")
        del tl, lij, lsd, lcap, hc_m

        # 4. Validation forward (compute_sd=False)
        encoder.eval(); predictor.eval(); proj_head.eval()
        if caption_forecaster: caption_forecaster.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            vtl, vlij, vlsd, vlcap, _ = forward_pass(
                hc_imgs, hc_embs, hc_rgb, hc_me, hc_mp,
                captions_t=hc_ct, captions_tp1=hc_ctp1,
                cap_emb_t=hc_ce_t, cap_emb_tp1=hc_ce_tp1,
                step=0, compute_sd=False,
            )
        logger.info(f"[HC] ✓ Validation forward (ij={vlij.item():.4f} cap={vlcap.item():.4f})")
        del vtl, vlij, vlsd, vlcap

        # 5. Visualization (4 samples, diffusion)
        torch.cuda.empty_cache()
        z_pred, h_gt, rgb_pred = generate_visualization(hc_imgs, hc_embs, hc_rgb, hc_ct)
        hc_vis = os.path.join(vis_folder, "health_check.png")
        save_visualization_grid(
            rgb_t=hc_imgs, emb_gt=h_gt, emb_pred=z_pred,
            rgb_tp1_gt=hc_rgb, rgb_tp1_pred=rgb_pred,
            save_path=hc_vis, crop_size=crop_size,
        )
        logger.info(f"[HC] ✓ Visualization (4 samples, saved: {hc_vis})")
        del z_pred, h_gt, rgb_pred

        # 6. Checkpoint save/load
        torch.cuda.empty_cache()
        hc_ckpt = os.path.join(folder, "health_check.pth.tar")
        save_checkpoint(0, 999.0, is_best=False)
        logger.info("[HC] ✓ Checkpoint save")

        # 7. Caption metrics
        if use_captions and hc_ct and hc_ctp1:
            cm = compute_caption_metrics(hc_ct["semantic"], hc_ctp1["semantic"])
            logger.info(
                f"[HC] ✓ Caption metrics: "
                f"BLEU-1={cm['bleu_1']:.4f} BLEU-2={cm['bleu_2']:.4f} "
                f"BLEU-3={cm['bleu_3']:.4f} BLEU-4={cm['bleu_4']:.4f} "
                f"ROUGE-L={cm['rouge_l']:.4f} CIDEr={cm['cider']:.4f}"
            )

        # Cleanup
        del hc_imgs, hc_embs, hc_rgb, hc_me, hc_mp
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"[HC] Peak VRAM: {peak:.1f}GB")

        logger.info("=" * 60)
        logger.info("HEALTH CHECK PASSED — starting training!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"HEALTH CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError("Health check failed, fix errors before training") from e

    # ========================================================================
    # Run Validation First (if configured)
    # ========================================================================

    if run_val_first and load_model:
        logger.info("\n" + "=" * 60)
        logger.info("RUN_VAL_FIRST: Running validation before training starts...")
        logger.info("=" * 60)

        encoder.eval(); predictor.eval(); proj_head.eval()
        if caption_forecaster: caption_forecaster.eval()
        torch.cuda.empty_cache()

        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_free = vram_total - vram_used
        rvf_compute_sd = vram_free > 6.0
        rvf_vis_n = 4 if vram_free > 6.0 else 2
        logger.info(f"[Val] VRAM free={vram_free:.1f}GB → compute_sd={rvf_compute_sd}, vis={rvf_vis_n}")

        rvf_loss = AverageMeter()
        rvf_ij = AverageMeter()
        rvf_sd = AverageMeter()
        rvf_cap = AverageMeter()

        # Added Meters for Image Metrics
        rvf_l1 = AverageMeter()
        rvf_mse = AverageMeter()
        rvf_psnr = AverageMeter()
        rvf_ssim = AverageMeter()
        rvf_gssim = AverageMeter()
        rvf_lpips = AverageMeter()

        rvf_sem_t, rvf_sem_tp1 = [], []
        rvf_cos_sum, rvf_cos_n = 0.0, 0
        rvf_vis_done = False

        try:
            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    if use_captions:
                        vd, vme, vmp, vct, vctp1, vce_t, vce_tp1 = vb
                    else:
                        vd, vme, vmp = vb
                        vct, vctp1, vce_t, vce_tp1 = None, None, None, None

                    vi_t = vd[0].to(device)
                    vi_e = vd[1].to(device)
                    vi_r = vd[2].to(device)
                    vme = [m.to(device) for m in vme]
                    vmp = [m.to(device) for m in vmp]

                    tl, lij, lsd, lcap, vm = forward_pass(
                        vi_t, vi_e, vi_r, vme, vmp,
                        captions_t=vct, captions_tp1=vctp1,
                        cap_emb_t=vce_t, cap_emb_tp1=vce_tp1,
                        step=vi, compute_sd=rvf_compute_sd,
                    )
                    rvf_loss.update(tl.item())
                    rvf_ij.update(lij.item())
                    rvf_cap.update(lcap.item())

                    if rvf_compute_sd: 
                        rvf_sd.update(lsd.item())
                        
                        # Generating Images to Compute Metrics
                        _, _, rvf_rgb_pred = generate_visualization(
                            vi_t, vi_e, vi_r,
                            {k: vs for k, vs in vct.items()} if vct else None
                        )
                        batch_img_metrics = compute_all_metrics(rvf_rgb_pred, vi_r, device=device, use_lpips=True)
                        
                        rvf_l1.update(batch_img_metrics['l1'])
                        rvf_mse.update(batch_img_metrics['mse'])
                        if batch_img_metrics['psnr'] != float('inf'): rvf_psnr.update(batch_img_metrics['psnr'])
                        rvf_ssim.update(batch_img_metrics['ssim'])
                        rvf_gssim.update(batch_img_metrics['gssim'])
                        if not math.isnan(batch_img_metrics['lpips']): rvf_lpips.update(batch_img_metrics['lpips'])
                        
                        del rvf_rgb_pred

                    if 'caption_cos_sim' in vm:
                        rvf_cos_sum += vm['caption_cos_sim']; rvf_cos_n += 1
                    if use_captions and vct and vctp1:
                        rvf_sem_t.extend(vct["semantic"])
                        rvf_sem_tp1.extend(vctp1["semantic"])

                    if vi == 0 and rank == 0 and not rvf_vis_done:
                        torch.cuda.empty_cache()
                        n = min(rvf_vis_n, vi_t.size(0))
                        zp, hg, rp = generate_visualization(
                            vi_t[:n], vi_e[:n], vi_r[:n],
                            {k: v[:n] for k, v in vct.items()} if vct else None
                        )
                        vp = os.path.join(vis_folder, "val_pre_training.png")
                        save_visualization_grid(rgb_t=vi_t[:n], emb_gt=hg, emb_pred=zp,
                                                rgb_tp1_gt=vi_r[:n], rgb_tp1_pred=rp,
                                                save_path=vp, crop_size=crop_size)
                        logger.info(f"Saved pre-training val vis: {vp}")
                        del zp, hg, rp; rvf_vis_done = True

                    del vi_t, vi_e, vi_r, vme, vmp, tl, lij, lsd, lcap, vm
                    torch.cuda.empty_cache()
                    if (vi + 1) % 100 == 0:
                        logger.info(f"[Pre-Val] {vi+1} batches...")

            cc = rvf_cos_sum / max(rvf_cos_n, 1)
            logger.info(f"[Pre-Val] loss={rvf_loss.avg:.4f} ij={rvf_ij.avg:.4f} sd={rvf_sd.avg:.4f} cap={rvf_cap.avg:.4f} cos={cc:.4f}")
            
            # Logging Pre-Val Image Metrics
            if rvf_compute_sd:
                logger.info(
                    f"[Pre-Val Image Metrics] L1={rvf_l1.avg:.4f} MSE={rvf_mse.avg:.4f} "
                    f"PSNR={rvf_psnr.avg:.4f} SSIM={rvf_ssim.avg:.4f} "
                    f"GSSIM={rvf_gssim.avg:.4f} LPIPS={rvf_lpips.avg:.4f}"
                )

            if use_captions and len(rvf_sem_tp1) > 0 and rank == 0:
                try:
                    cm = compute_caption_metrics(rvf_sem_t, rvf_sem_tp1)
                    logger.info(
                        f"[Pre-Val Caption] BLEU-1={cm['bleu_1']:.4f} BLEU-2={cm['bleu_2']:.4f} "
                        f"BLEU-3={cm['bleu_3']:.4f} BLEU-4={cm['bleu_4']:.4f} "
                        f"ROUGE-L={cm['rouge_l']:.4f} CIDEr={cm['cider']:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"[Pre-Val] Caption metrics failed: {e}")

        except Exception as e:
            logger.error(f"[Pre-Val] Error: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

        encoder.train(); predictor.train(); proj_head.train()
        if caption_forecaster: caption_forecaster.train()
        logger.info("Pre-training validation complete.\n")

    # ========================================================================
    # Training Loop
    # ========================================================================

    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Caption conditioning: {'ENABLED' if use_captions else 'DISABLED'}")
    logger.info(f"Caption forecast weight: {caption_forecast_weight}")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        loss_ijepa_meter = AverageMeter()
        loss_sd_meter = AverageMeter()
        loss_cap_meter = AverageMeter()
        time_meter = AverageMeter()

        encoder.train()
        predictor.train()
        proj_head.train()
        if caption_forecaster is not None:
            caption_forecaster.train()

        for itr, batch in enumerate(train_loader):

            # Unpack batch — with or without captions
            if use_captions:
                batch_data, masks_enc, masks_pred, captions_t, captions_tp1, cap_emb_t, cap_emb_tp1 = batch
            else:
                batch_data, masks_enc, masks_pred = batch
                captions_t, captions_tp1, cap_emb_t, cap_emb_tp1 = None, None, None, None

            imgs_t = batch_data[0].to(device, non_blocking=True)
            embs_tp1 = batch_data[1].to(device, non_blocking=True)
            rgb_tp1 = batch_data[2].to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            def train_step():
                nonlocal global_step

                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                total_loss, loss_ijepa, loss_sd, loss_cap, sd_metrics = forward_pass(
                    imgs_t, embs_tp1, rgb_tp1,
                    masks_enc, masks_pred,
                    captions_t=captions_t, captions_tp1=captions_tp1,
                    cap_emb_t=cap_emb_t, cap_emb_tp1=cap_emb_tp1,
                    step=global_step, compute_sd=True,
                )

                # Backward
                if use_bfloat16:
                    total_loss.backward()
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                    optimizer.step()
                else:
                    if scaler:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        for group in optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        for group in optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                        optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())

                # EMA update
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for p_q, p_k in zip(encoder.parameters(), target_encoder.parameters()):
                        p_k.data.mul_(m).add_((1 - m) * p_q.detach().data)

                global_step += 1
                return (total_loss.item(), loss_ijepa.item(), loss_sd.item(),
                        loss_cap.item(), sd_metrics, new_lr, grad_stats)

            (loss, l_ij, l_sd, l_cap, sd_m, new_lr, grad_stats), elapsed = gpu_timer(train_step)

            loss_meter.update(loss)
            loss_ijepa_meter.update(l_ij)
            loss_sd_meter.update(l_sd)
            loss_cap_meter.update(l_cap)
            time_meter.update(elapsed)

            csv_logger.log(
                epoch+1, itr, loss, l_ij, l_sd, l_cap,
                len(masks_enc[0][0]), len(masks_pred[0][0]), elapsed
            )

            if itr % log_freq == 0 or np.isnan(loss):
                logger.info(
                    f"[{epoch+1},{itr:5d}] loss={loss_meter.avg:.4f} "
                    f"ij={loss_ijepa_meter.avg:.4f} sd={loss_sd_meter.avg:.4f} "
                    f"cap={loss_cap_meter.avg:.4f} "
                    f"lr={new_lr:.2e} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                    f"({time_meter.avg:.0f}ms)"
                )
                if sd_m:
                    m_str = " | ".join(f"{k}:{v:.4f}" for k, v in sd_m.items() if isinstance(v, (int, float)) and not np.isnan(v))
                    if m_str:
                        logger.info(f"[{epoch+1},{itr:5d}] SD: {m_str}")

            # Visualization
            if rank == 0 and vis_interval_iters > 0 and (itr + 1) % vis_interval_iters == 0:
                logger.info(f"Generating visualization...")
                encoder.eval(); predictor.eval(); proj_head.eval()
                if caption_forecaster: caption_forecaster.eval()

                z_pred, h_gt, rgb_pred = generate_visualization(imgs_t, embs_tp1, rgb_tp1, captions_t)
                vis_path = os.path.join(vis_folder, f"train_ep{epoch+1}_iter{itr+1}.png")
                save_visualization_grid(rgb_t=imgs_t, emb_gt=h_gt, emb_pred=z_pred,
                                        rgb_tp1_gt=rgb_tp1, rgb_tp1_pred=rgb_pred,
                                        save_path=vis_path, crop_size=crop_size)
                logger.info(f"Saved: {vis_path}")

                encoder.train(); predictor.train(); proj_head.train()
                if caption_forecaster: caption_forecaster.train()

            assert not np.isnan(loss), "Loss is NaN!"

        # End of epoch
        logger.info(f"Epoch {epoch+1} done. loss={loss_meter.avg:.4f} "
                     f"(ij={loss_ijepa_meter.avg:.4f} sd={loss_sd_meter.avg:.4f} cap={loss_cap_meter.avg:.4f})")

        # ====================================================================
        # Validation (every 5 epochs)
        # ====================================================================

        if (epoch + 1) % 5 != 0:
            # Save checkpoint every epoch (not just val epochs)
            save_checkpoint(epoch + 1, best_val_loss, is_best=False)
            continue

        # Save checkpoint BEFORE validation — if val crashes, state is safe
        save_checkpoint(epoch + 1, best_val_loss, is_best=False)
        logger.info("Pre-val checkpoint saved")

        logger.info("Running validation...")
        encoder.eval(); predictor.eval(); proj_head.eval()
        if caption_forecaster: caption_forecaster.eval()
        torch.cuda.empty_cache()

        # Auto-detect VRAM headroom
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_free = vram_total - vram_used
        val_compute_sd = vram_free > 6.0
        val_vis_samples = 4 if vram_free > 6.0 else 2
        logger.info(f"[Val] VRAM: {vram_used:.1f}/{vram_total:.1f}GB (free={vram_free:.1f}GB) → compute_sd={val_compute_sd}, vis_samples={val_vis_samples}")

        val_loss_meter = AverageMeter()
        val_ijepa_meter = AverageMeter()
        val_sd_meter = AverageMeter()
        val_cap_meter = AverageMeter()
        
        # Added Meters for Image Metrics in Validation loop
        val_l1_meter = AverageMeter()
        val_mse_meter = AverageMeter()
        val_psnr_meter = AverageMeter()
        val_ssim_meter = AverageMeter()
        val_gssim_meter = AverageMeter()
        val_lpips_meter = AverageMeter()

        all_baseline_semantic = []
        all_gt_semantic = []
        val_cap_cos_sum = 0.0
        val_cap_cos_count = 0
        val_vis_saved = False

        try:
            with torch.no_grad():
                for v_itr, v_batch in enumerate(val_loader):
                    if use_captions:
                        v_data, v_me, v_mp, v_ct, v_ctp1, v_ce_t, v_ce_tp1 = v_batch
                    else:
                        v_data, v_me, v_mp = v_batch
                        v_ct, v_ctp1, v_ce_t, v_ce_tp1 = None, None, None, None

                    v_imgs_t = v_data[0].to(device)
                    v_embs_tp1 = v_data[1].to(device)
                    v_rgb_tp1 = v_data[2].to(device)
                    v_me = [m.to(device) for m in v_me]
                    v_mp = [m.to(device) for m in v_mp]

                    tl, lij, lsd, lcap, v_sd_m = forward_pass(
                        v_imgs_t, v_embs_tp1, v_rgb_tp1, v_me, v_mp,
                        captions_t=v_ct, captions_tp1=v_ctp1,
                        cap_emb_t=v_ce_t, cap_emb_tp1=v_ce_tp1,
                        step=global_step + v_itr, compute_sd=val_compute_sd,
                    )
                    val_loss_meter.update(tl.item())
                    val_ijepa_meter.update(lij.item())
                    val_cap_meter.update(lcap.item())
                    
                    if val_compute_sd:
                        val_sd_meter.update(lsd.item())
                        
                        # Generate Predictions for Image Evaluation Metrics
                        _, _, v_rgb_pred = generate_visualization(
                            v_imgs_t, v_embs_tp1, v_rgb_tp1,
                            {k: vs for k, vs in v_ct.items()} if v_ct else None
                        )
                        v_img_metrics = compute_all_metrics(v_rgb_pred, v_rgb_tp1, device=device, use_lpips=True)
                        
                        val_l1_meter.update(v_img_metrics['l1'])
                        val_mse_meter.update(v_img_metrics['mse'])
                        if v_img_metrics['psnr'] != float('inf'): val_psnr_meter.update(v_img_metrics['psnr'])
                        val_ssim_meter.update(v_img_metrics['ssim'])
                        val_gssim_meter.update(v_img_metrics['gssim'])
                        if not math.isnan(v_img_metrics['lpips']): val_lpips_meter.update(v_img_metrics['lpips'])
                        
                        del v_rgb_pred

                    if 'caption_cos_sim' in v_sd_m:
                        val_cap_cos_sum += v_sd_m['caption_cos_sim']
                        val_cap_cos_count += 1

                    if use_captions and v_ct is not None and v_ctp1 is not None:
                        all_baseline_semantic.extend(v_ct["semantic"])
                        all_gt_semantic.extend(v_ctp1["semantic"])

                    # Visualization on first batch
                    if v_itr == 0 and rank == 0 and not val_vis_saved:
                        torch.cuda.empty_cache()
                        vis_n = min(val_vis_samples, v_imgs_t.size(0))
                        z_pred, h_gt, rgb_pred = generate_visualization(
                            v_imgs_t[:vis_n], v_embs_tp1[:vis_n], v_rgb_tp1[:vis_n],
                            {k: vs[:vis_n] for k, vs in v_ct.items()} if v_ct else None
                        )
                        vis_path = os.path.join(vis_folder, f"val_ep{epoch+1}.png")
                        save_visualization_grid(
                            rgb_t=v_imgs_t[:vis_n], emb_gt=h_gt, emb_pred=z_pred,
                            rgb_tp1_gt=v_rgb_tp1[:vis_n], rgb_tp1_pred=rgb_pred,
                            save_path=vis_path, crop_size=crop_size,
                        )
                        logger.info(f"Saved val vis: {vis_path}")
                        del z_pred, h_gt, rgb_pred
                        val_vis_saved = True

                    del v_imgs_t, v_embs_tp1, v_rgb_tp1, v_me, v_mp, tl, lij, lsd, lcap, v_sd_m
                    torch.cuda.empty_cache()

                    if (v_itr + 1) % 100 == 0:
                        logger.info(f"[Val] {v_itr + 1} batches done...")

            logger.info(f"[Val] Loop complete ({v_itr + 1} batches)")

        except Exception as e:
            logger.error(f"[Val] Error during validation loop: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

        # ====================================================================
        # Evaluation Metrics (text-level and image-level)
        # ====================================================================

        val_loss = val_loss_meter.avg if val_loss_meter.count > 0 else 0.0
        cap_cos_avg = val_cap_cos_sum / max(val_cap_cos_count, 1)

        logger.info(
            f"Val: loss={val_loss:.4f} "
            f"(ij={val_ijepa_meter.avg:.4f} sd={val_sd_meter.avg:.4f} "
            f"cap={val_cap_meter.avg:.4f} cap_cos={cap_cos_avg:.4f})"
        )
        
        if val_compute_sd:
            logger.info(
                f"[Val-Image] L1={val_l1_meter.avg:.4f} MSE={val_mse_meter.avg:.4f} "
                f"PSNR={val_psnr_meter.avg:.4f} SSIM={val_ssim_meter.avg:.4f} "
                f"GSSIM={val_gssim_meter.avg:.4f} LPIPS={val_lpips_meter.avg:.4f}"
            )

        if use_captions and len(all_gt_semantic) > 0 and rank == 0:
            try:
                baseline_cap_metrics = compute_caption_metrics(
                    all_baseline_semantic, all_gt_semantic
                )
                logger.info(
                    f"[Val-Caption] Baseline semantic(t) vs GT semantic(t+1): "
                    f"BLEU-1={baseline_cap_metrics['bleu_1']:.4f} "
                    f"BLEU-2={baseline_cap_metrics['bleu_2']:.4f} "
                    f"BLEU-3={baseline_cap_metrics['bleu_3']:.4f} "
                    f"BLEU-4={baseline_cap_metrics['bleu_4']:.4f} "
                    f"ROUGE-L={baseline_cap_metrics['rouge_l']:.4f} "
                    f"CIDEr={baseline_cap_metrics['cider']:.4f}"
                )
            except Exception as e:
                logger.warning(f"[Val-Caption] Caption metrics failed: {e}")
            logger.info(
                f"[Val-Caption] Forecast embedding cosine sim: {cap_cos_avg:.4f} "
                f"(1.0 = perfect forecast, baseline = copy semantic(t))"
            )

        val_csv_logger.log(epoch+1, val_loss, val_ijepa_meter.avg, val_sd_meter.avg)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch+1, val_loss, is_best=is_best)
        logger.info(f"Checkpoint saved. Best val loss: {best_val_loss:.4f}")

        encoder.train(); predictor.train(); proj_head.train()
        if caption_forecaster: caption_forecaster.train()

    logger.info(f"\nTRAINING COMPLETE. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    raise RuntimeError("Use main.py as entry point: python main.py --fname config.yaml")