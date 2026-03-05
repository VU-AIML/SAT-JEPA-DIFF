# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# IJEPA + Stable Diffusion Training Script
#
# Key features:
# 1. Correct proj_head dimension (pred_emb_dim -> target_emb_dim)
# 2. Real diffusion sampling for visualization (not just VAE reconstruction)
# 3. Periodic visualization during training
# 4. Proper dtype handling for VAE stability
# 5. Hybrid Loss Integration (Fixes Spatial Collapse/One-Color Issue)
#
# COARSE RGB CONDITIONING (No VAE for coarse):
# - t image downsampled to 32x32 RGB (NO VAE encoding)
# - 32x32 RGB = 3072 values (vs 256 for 4x4 VAE latent) = 12x more info!
# - Coarse structure from RGB, fine details from IJEPA embeddings
# - Output at full resolution (128x128 or 256x256)

import os
import copy
import logging
import math
import sys
import yaml
from pathlib import Path

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
from sd_models import load_sd_model, save_full_checkpoint, load_full_checkpoint
from sd_joint_loss import compute_sd_loss_and_metrics, diffusion_sample
from embedding_validation import EmbeddingConsistencyValidator


# ============================================================================
# Global Settings
# ============================================================================
log_freq = 10
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ============================================================================
# Hybrid IJEPA Loss - The Fix for Spatial Collapse
# ============================================================================

class HybridIJEPALoss(torch.nn.Module):
    """
    Balanced Loss to ensure both Global Diversity and Local Detail.
    
    Fixes:
    1. Mode Collapse (via Contrastive Loss)
    2. Spatial Collapse (via Spatial Variance Loss & High L1)
    3. Embedding Mismatch (via Feature Regression) [NEW]
    """
    def __init__(
        self,
        l1_weight: float = 20.0,         # UPDATED: High weight to force pixel-perfect match (was 5.0)
        cosine_weight: float = 1.0,      # Direction match
        contrastive_weight: float = 1.0, # UPDATED: Increased slightly (was 0.5)
        spatial_var_weight: float = 10.0,# UPDATED: Forces high variance to kill blur (was 1.0)
        feature_reg_weight: float = 5.0, # UPDATED: Forces raw features to match target encoder (was 2.0)
        temperature: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        self.contrastive_weight = contrastive_weight
        self.spatial_var_weight = spatial_var_weight
        self.feature_reg_weight = feature_reg_weight # [NEW]
        self.temperature = temperature
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        pred_raw: torch.Tensor = None,   # [NEW] High-dim prediction
        target_raw: torch.Tensor = None  # [NEW] High-dim target
    ):
        """
        Args:
            pred: (B, N, 64) predicted embeddings (Projected)
            target: (B, N, 64) target embeddings (Projected)
            pred_raw: (B, N, D) predicted embeddings (High Dim) [NEW]
            target_raw: (B, N, D) target embeddings (High Dim) [NEW]
        """
        B, N, C = pred.shape
        losses = {}
        
        # 1. Reconstruction (L1) - High Weight
        # Forces predictions to match the detailed texture of targets
        l1_loss = F.l1_loss(pred, target) # Changed to pure L1 for sharper gradients
        losses['l1'] = l1_loss
        
        # 2. Cosine Similarity
        pred_norm = F.normalize(pred, dim=-1, eps=1e-6)
        target_norm = F.normalize(target, dim=-1, eps=1e-6)
        cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        cosine_loss = 1.0 - cos_sim
        losses['cosine'] = cosine_loss
        losses['cos_sim'] = cos_sim # Log raw similarity too
        
        # 3. Spatial Variance Loss (The "Anti-Flat" Fix)
        # We calculate the variance *across patches* (dim=1)
        # If prediction is flat (single color), pred_var will be 0.
        # We force pred_var to match target_var (which is high/detailed).
        pred_std = torch.sqrt(pred.var(dim=1) + 1e-6).mean()
        target_std = torch.sqrt(target.var(dim=1) + 1e-6).mean()
        spatial_loss = F.mse_loss(pred_std, target_std)
        losses['spatial'] = spatial_loss
        
        # 4. Contrastive Loss (Global Diversity)
        # Kept but with lower weight to prevent overpowering spatial details
        if B > 1:
            pred_global = pred.mean(dim=1)  
            target_global = target.mean(dim=1)
            
            pred_global = F.normalize(pred_global, dim=-1)
            target_global = F.normalize(target_global, dim=-1)
            
            logits = torch.matmul(pred_global, target_global.T) / self.temperature
            labels = torch.arange(B, device=pred.device)
            contrastive_loss = F.cross_entropy(logits, labels)
            losses['contr'] = contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=pred.device)
            losses['contr'] = contrastive_loss
            
        # 5. Feature Regression Loss [NEW]
        # Forces the raw output (before projection) to match the target encoder's raw output.
        # This is critical for SD generation because SD uses 'pred_raw'.
        if pred_raw is not None and target_raw is not None:
            # Normalize to ensure stable gradients
            p_raw_n = F.layer_norm(pred_raw, (pred_raw.size(-1),))
            t_raw_n = F.layer_norm(target_raw, (target_raw.size(-1),))
            feat_loss = F.l1_loss(p_raw_n, t_raw_n) # Use L1 for sharpness
            losses['feat_reg'] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=pred.device)
            losses['feat_reg'] = feat_loss
        
        # Weighted Sum
        total = (
            self.l1_weight * l1_loss +
            self.cosine_weight * cosine_loss +
            self.spatial_var_weight * spatial_loss +
            self.contrastive_weight * contrastive_loss +
            self.feature_reg_weight * feat_loss
        )
        
        return total, losses


# ============================================================================
# Utility Functions
# ============================================================================

def embedding_to_patches(embs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert spatial embedding (B, C, H, W) to patch tokens (B, N, C).
    Uses average pooling over non-overlapping patch_size x patch_size windows.
    """
    B, C, H, W = embs.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Size ({H}x{W}) must be divisible by patch_size={patch_size}"
    
    Hp, Wp = H // patch_size, W // patch_size
    
    embs = embs.view(B, C, Hp, patch_size, Wp, patch_size)
    embs = embs.mean(dim=(3, 5))  # Average pool within each patch
    embs = embs.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
    
    return embs


def visualize_tokens(tokens: torch.Tensor, size: int) -> torch.Tensor:
    """
    Visualize high-dim tokens using PCA projection to RGB.
    Safe version: Uses float32 and handles potential SVD errors.
    """
    B, N, C = tokens.shape
    h = w = int(math.sqrt(N))
    
    # 1. Flatten and cast to float32 (Safe for VRAM & Stability)
    x = tokens.reshape(-1, C).float()
    
    # 2. Standardize (Mean=0, Std=1) for better PCA
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-5
    x_norm = (x - mean) / std
    
    # 3. PCA: Compute top 3 components
    try:
        # q=3 for RGB. center=False because we already standardized.
        U, S, V = torch.pca_lowrank(x_norm, q=3, center=False, niter=2)
        # Project data: (Batch*Pixels, 768) @ (768, 3) -> (Batch*Pixels, 3)
        pca_proj = torch.matmul(x_norm, V[:, :3])
    except Exception as e:
        # Fallback to first 3 channels if PCA fails
        pca_proj = x_norm[:, :3]

    # 4. Normalize to [0, 1] range for Image Saving (Per Batch Item for Contrast)
    pca_proj = pca_proj.reshape(B, N, 3)
    pca_min = pca_proj.min(dim=1, keepdim=True)[0]
    pca_max = pca_proj.max(dim=1, keepdim=True)[0]
    pca_proj = (pca_proj - pca_min) / (pca_max - pca_min + 1e-6)
    
    # 5. Reshape back to (B, 3, H, W)
    vis = pca_proj.reshape(B, h, w, 3).permute(0, 3, 1, 2)
    
    # 6. Resize to target resolution
    if vis.shape[-2:] != (size, size):
        vis = F.interpolate(vis, size=(size, size), mode='nearest')
    
    return vis.clamp(0, 1)


def save_visualization_grid(
    rgb_t: torch.Tensor,
    emb_gt: torch.Tensor,
    emb_pred: torch.Tensor,
    rgb_tp1_gt: torch.Tensor,
    rgb_tp1_pred: torch.Tensor,
    save_path: str,
    crop_size: int = 128,
    max_samples: int = 4,
):
    """
    Save a 5-column visualization grid:
    [RGB_t | Emb_GT | Emb_Pred | RGB_GT_tp1 | RGB_Pred_tp1]
    """
    B = min(rgb_t.shape[0], max_samples)
    
    # Prepare RGB images
    rgb_t = rgb_t[:B].float().clamp(0, 1)
    rgb_tp1_gt = rgb_tp1_gt[:B].float().clamp(0, 1)
    
    if rgb_tp1_pred is not None:
        rgb_tp1_pred = rgb_tp1_pred[:B].float().clamp(0, 1)
    else:
        rgb_tp1_pred = torch.zeros_like(rgb_tp1_gt)
    
    # Visualize embeddings
    emb_gt_vis = visualize_tokens(emb_gt[:B], crop_size)
    emb_pred_vis = visualize_tokens(emb_pred[:B], crop_size)
    
    # Build grid
    rows = []
    for b in range(B):
        row = torch.cat([
            rgb_t[b],
            emb_gt_vis[b],
            emb_pred_vis[b],
            rgb_tp1_gt[b],
            rgb_tp1_pred[b],
        ], dim=2)  # Horizontal concatenation
        rows.append(row)
    
    grid = torch.cat(rows, dim=1)  # Vertical concatenation
    save_image(grid, save_path)


def run_health_check(sd_state, encoder, predictor, proj_head, train_loader, device):
    """
    Performs a quick functional test of the entire pipeline before starting training.
    """
    print("\n" + "="*30)
    print("STARTING SYSTEM HEALTH CHECK...")
    print("="*30)

    try:
        # 1. Data Loader Check
        print("[Step 1/5] Checking Data Loader...")
        batch = next(iter(train_loader))
        batch_data, masks_enc, masks_pred = batch
        imgs_t = batch_data[0].to(device, non_blocking=True)
        embs_tp1 = batch_data[1].to(device, non_blocking=True)
        rgb_tp1 = batch_data[2].to(device, non_blocking=True)
        print(f"  - Data shapes: RGB={imgs_t.shape}, Embs={embs_tp1.shape}")

        # 2. IJEPA Path Check
        print("[Step 2/5] Checking IJEPA Forward Pass...")
        with torch.no_grad():
            z_enc = encoder(imgs_t, [m.to(device) for m in masks_enc])
            z_pred = predictor(z_enc, [m.to(device) for m in masks_enc], [m.to(device) for m in masks_pred])
            z_proj = proj_head(z_pred)
        print(f"  - IJEPA Predictor (Raw) shape: {z_pred.shape}")
        print(f"  - IJEPA Proj (Loss) shape: {z_proj.shape}")

        # 3. Fixed Prompt & Text Encoder Check
        print("[Step 3/5] Checking Fixed Prompt Embeddings...")
        prompt_embeds = sd_state.get("prompt_embeds")
        if prompt_embeds is None or prompt_embeds.abs().sum() < 1e-6:
            print("  - [ERROR] Prompt embeddings are empty or zeros!")
            print("  - Check Hugging Face token and Text Encoder logs.")
            return False
        print("  - Text conditioning signal verified.")

        # 4. SD3.5 + Adapter Loss Check (COARSE CONDITIONING mode)
        print("[Step 4/5] Checking SD Joint Loss Logic (COARSE CONDITIONING mode)...")
        from sd_joint_loss import compute_sd_loss_and_metrics
        sd_loss, metrics, _ = compute_sd_loss_and_metrics(
            sd_state=sd_state,
            ijepa_tokens=z_pred,
            rgb_target=rgb_tp1.to(torch.float32),
            rgb_source=imgs_t.to(torch.float32),  # Will be downsampled to 32x32
            step=0,
            gt_tokens=None,
            do_diffusion_sample=False
        )
        print(f"  - Joint Loss calculated: {sd_loss.item():.4f}")

        # 5. Memory Check
        print("[Step 5/5] Checking GPU Memory...")
        mem_used = torch.cuda.memory_allocated(device) / 1e9
        print(f"  - GPU Memory Allocated: {mem_used:.2f} GB")

        print("="*30)
        print("HEALTH CHECK PASSED SUCCESSFULLY!")
        print("="*30 + "\n")
        return True

    except Exception as e:
        print("\n" + "!"*30)
        print(f"HEALTH CHECK FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("!"*30 + "\n")
        return False


# ============================================================================
# Main Training Function
# ============================================================================

def main(args, resume_preempt: bool = False):
    
    # ========================================================================
    # Parse Configuration
    # ========================================================================
    
    # Meta
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
    
    # [NEW] Reference dropout probability for training (default 0.15 = 15% dropout - ORIGINAL VALUE)
    ref_dropout_prob = float(args["meta"].get("ref_dropout_prob", 0.15))
    
    # LoRA settings
    use_lora = args["meta"].get("use_lora", True)
    lora_rank = int(args["meta"].get("lora_rank", 16))
    lora_alpha = int(args["meta"].get("lora_alpha", 32))
    
    # Hybrid Loss Weights (Overridable from config)
    # UPDATED DEFAULTS to be Aggressive
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
    # Device Setup
    # ========================================================================
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # ========================================================================
    # Output Directories
    # ========================================================================
    
    os.makedirs(folder, exist_ok=True)
    vis_folder = os.path.join(folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    
    # Save config
    with open(os.path.join(folder, "config.yaml"), "w") as f:
        yaml.dump(args, f)
    
    # ========================================================================
    # Distributed Init
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
        torch.distributed.is_available() and
        torch.distributed.is_initialized() and
        world_size > 1
    )
    
    # ========================================================================
    # Paths
    # ========================================================================
    
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    val_log_file = os.path.join(folder, f"{tag}_val_r{rank}.csv")
    best_path = os.path.join(folder, f"{tag}-best.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file else best_path
    
    # CSV Loggers
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_ijepa"),
        ("%.5f", "loss_sd"),
        ("%.5f", "mask_enc"),
        ("%.5f", "mask_pred"),
        ("%d", "time_ms"),
    )
    
    val_csv_logger = CSVLogger(
        val_log_file,
        ("%d", "epoch"),
        ("%.5f", "val_loss"),
        ("%.5f", "val_loss_ijepa"),
        ("%.5f", "val_loss_sd"),
    )
    
    # ========================================================================
    # Build Models
    # ========================================================================
    
    logger.info("Building models...")
    
    # IJEPA encoder and predictor
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # ========================================================================
    # proj_head dimension (FIXED)
    # ========================================================================
    
    # Get encoder embed_dim (Usually 384 or 768 or 1280)
    if hasattr(encoder, "embed_dim"):
        encoder_embed_dim = encoder.embed_dim
    elif hasattr(encoder, "module") and hasattr(encoder.module, "embed_dim"):
        encoder_embed_dim = encoder.module.embed_dim
    else:
        encoder_embed_dim = 1280
    
    # This projects Predictor -> Loss Dimension (64)
    # NOTE: target_emb_dim comes from config (set it back to 64!)
    proj_head = torch.nn.Linear(encoder_embed_dim, target_emb_dim).to(device)
    logger.info(f"proj_head (For Loss): {encoder_embed_dim} -> {target_emb_dim}")
    
    # Stable Diffusion with LoRA
    logger.info("Loading Stable Diffusion...")
    sd_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    
    # CRITICAL CHANGE: We tell SD adapter to expect the RAW Predictor Dimension (High Dim)
    # instead of the compressed Loss Dimension (64).
    # This gives SD the FULL bandwidth.
    logger.info(f"Setting SD Adapter Input Dimension to Predictor output: {encoder_embed_dim}")
    
    sd_state = load_sd_model(
        device=device,
        checkpoint_dir=sd_checkpoint_dir,
        dtype=sd_dtype,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        hf_token=args["meta"].get("hf_token", None),
        target_emb_dim=encoder_embed_dim  # <--- PASSING HIGH DIM (e.g. 768) HERE!
    )
    
    # Force VAE to float32
    if "vae" in sd_state:
        sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)
    
    # ========================================================================
    # Data Loaders
    # ========================================================================
    
    logger.info("Building data loaders...")
    
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )
    
    full_dataset = S2FutureEmbeddingDataset(
        root_dir=root_path,
        patch_size=crop_size,
        transform=None,
    )
    
    # Train/val split
    num_samples = len(full_dataset)
    indices = np.random.permutation(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=mask_collator,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        collate_fn=mask_collator,
        sampler=val_sampler,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    
    ipe = len(train_loader)
    logger.info(f"Dataset: {num_samples} samples, {ipe} iters/epoch")
    
    # ========================================================================
    # Optimizer
    # ========================================================================
    
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    
    optimizer.add_param_group({"params": proj_head.parameters()})
    
    if "trainable_params" in sd_state and len(sd_state["trainable_params"]) > 0:
        optimizer.add_param_group({"params": sd_state["trainable_params"]})
        logger.info(f"Added {len(sd_state['trainable_params'])} SD trainable params")
    
    # ========================================================================
    # DDP Wrap
    # ========================================================================
    
    if distributed:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        proj_head = DistributedDataParallel(proj_head, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    # ========================================================================
    # LOSS: Hybrid Loss (Spatial Variance + Contrastive + L1)
    # ========================================================================
    
    ijepa_loss_fn = HybridIJEPALoss(
        l1_weight=ijepa_l1_weight,
        cosine_weight=ijepa_cosine_weight,
        contrastive_weight=ijepa_contrastive_weight,
        spatial_var_weight=ijepa_spatial_weight,
        feature_reg_weight=ijepa_feature_reg_weight, # [NEW]
    )
    logger.info(f"Initialized HybridIJEPALoss: L1={ijepa_l1_weight}, Cos={ijepa_cosine_weight}, Contr={ijepa_contrastive_weight}, Spatial={ijepa_spatial_weight}, FeatReg={ijepa_feature_reg_weight}")
    
    # ========================================================================
    # Forward Pass (UPDATED: With configurable reference dropout)
    # ========================================================================
    
    def forward_pass(
        imgs_t: torch.Tensor,
        embs_tp1: torch.Tensor,
        rgb_tp1: torch.Tensor,
        masks_enc,
        masks_pred,
        step: int,
        compute_sd: bool = True,
        history_hidden: list = None,  # NEW: temporal history for training
    ):
        """Standard forward pass with double-stream logic + temporal attention."""
        
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
            
            # === IJEPA Forward (skip if disabled) ===
            if enable_ijepa:
                # === Target: GT embeddings (64 Dim) ===
                with torch.no_grad():
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),)) 
                    
                    num_patches = (crop_size // patch_size) ** 2
                    full_mask = [torch.arange(num_patches, device=imgs_t.device).unsqueeze(0).expand(imgs_t.size(0), -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)
                    
                    B_local = h.size(0)
                    
                    h_masked = apply_masks(h, masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(masks_enc))
                    
                    h_raw_masked = apply_masks(h_raw_all, masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(masks_enc))
                
                # === Prediction: IJEPA forward ===
                z_enc = encoder(imgs_t, masks_enc)
                z_raw = predictor(z_enc, masks_enc, masks_pred)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                z_proj = proj_head(z_raw_norm)
                
                # === IJEPA Loss ===
                loss_ijepa, loss_stats = ijepa_loss_fn(
                    pred=z_proj, 
                    target=h_masked,
                    pred_raw=z_raw_norm,
                    target_raw=h_raw_masked
                )
                
                if distributed:
                    loss_ijepa = AllReduce.apply(loss_ijepa)
            else:
                # IJEPA disabled - use random embeddings for SD
                B = imgs_t.size(0)
                num_patches = (crop_size // patch_size) ** 2
                encoder_embed_dim_local = encoder.module.embed_dim if distributed else encoder.embed_dim
                z_raw_norm = torch.randn(B, num_patches, encoder_embed_dim_local, device=imgs_t.device, dtype=dtype)
                loss_ijepa = torch.tensor(0.0, device=imgs_t.device)
                loss_stats = {'l1': torch.tensor(0.0), 'cosine': torch.tensor(0.0), 
                              'contr': torch.tensor(0.0), 'spatial': torch.tensor(0.0),
                              'feat_reg': torch.tensor(0.0)}
            
            # === SD Loss (with configurable reference dropout) ===
            if compute_sd:
                rgb_for_sd = rgb_tp1.to(device=imgs_t.device, dtype=torch.float32)
                rgb_source_sd = imgs_t.to(device=imgs_t.device, dtype=torch.float32)
                
                # Move temporal history to device if provided
                device_history = None
                if history_hidden is not None and len(history_hidden) > 0:
                    device_history = [h.to(device=imgs_t.device) for h in history_hidden]
                
                sd_loss, sd_metrics, _ = compute_sd_loss_and_metrics(
                    sd_state=sd_state,
                    ijepa_tokens=z_raw_norm, 
                    rgb_target=rgb_for_sd, 
                    rgb_source=rgb_source_sd,
                    gt_tokens=None,
                    step=step,
                    save_vis=False,
                    vis_folder=vis_folder,
                    do_diffusion_sample=False,
                    ssim_weight=ssim_weight,
                    embedding_guidance_weight=0.0,
                    ref_dropout_prob=ref_dropout_prob,  # [NEW] Configurable dropout
                    history_hidden=device_history,  # NEW: temporal history
                )
            else:
                sd_loss = torch.tensor(0.0, device=imgs_t.device)
                sd_metrics = {}
            
            total_loss = loss_ijepa + sd_loss_weight * sd_loss
        
        # Logging stats
        sd_metrics['l1'] = loss_stats['l1'].item() if torch.is_tensor(loss_stats['l1']) else loss_stats['l1']
        sd_metrics['feat_reg'] = loss_stats.get('feat_reg', torch.tensor(0)).item() if torch.is_tensor(loss_stats.get('feat_reg', 0)) else loss_stats.get('feat_reg', 0)
        sd_metrics['cos'] = loss_stats['cosine'].item() if torch.is_tensor(loss_stats['cosine']) else loss_stats['cosine']
        sd_metrics['contr'] = loss_stats['contr'].item() if torch.is_tensor(loss_stats['contr']) else loss_stats['contr']
        sd_metrics['spatial'] = loss_stats['spatial'].item() if torch.is_tensor(loss_stats['spatial']) else loss_stats['spatial']
        
        return total_loss, loss_ijepa, sd_loss, sd_metrics
    
    # ========================================================================
    # Generate Visualization (COARSE CONDITIONING - 32x32 downsample + 0.35 noise)
    # ========================================================================
    
    def generate_visualization(
            imgs_t: torch.Tensor,
            embs_tp1: torch.Tensor,
            rgb_tp1: torch.Tensor,
        ):
        """
        Generate full predictions for visualization (no masking).
        
        COARSE CONDITIONING:
        - ref_rgb (256x256) is downsampled to 32x32 internally by sd_joint_loss
        - This gives only COARSE structure (layout, colors)
        - 0.35 noise is added
        - Fine details must come from IJEPA embeddings
        - Output is full 256x256 resolution
        """
        
        B = imgs_t.size(0)
        num_patches = (crop_size // patch_size) ** 2
        
        full_mask = [
            torch.arange(num_patches, device=imgs_t.device)
            .unsqueeze(0).expand(B, -1)
        ]
        
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                if enable_ijepa:
                    # IJEPA prediction
                    z_enc = encoder(imgs_t, full_mask)
                    z_raw = predictor(z_enc, full_mask, full_mask)
                    z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                    z_proj = proj_head(z_raw_norm)
                else:
                    # Random embeddings when IJEPA disabled
                    encoder_embed_dim_local = encoder.module.embed_dim if distributed else encoder.embed_dim
                    z_raw_norm = torch.randn(B, num_patches, encoder_embed_dim_local, device=imgs_t.device, dtype=dtype)
                    z_proj = torch.randn(B, num_patches, target_emb_dim, device=imgs_t.device, dtype=dtype)
                
                # GT embedding tokens
                h = embedding_to_patches(embs_tp1, patch_size)
                h = F.layer_norm(h, (h.size(-1),))
            
            # Use RAW features for Diffusion
            z_for_sd = z_raw_norm.to(dtype=adapter_dtype)
            
            # =========================================================
            # COARSE CONDITIONING: imgs_t will be downsampled to 32x32
            # internally, then 0.35 noise added. Output is 256x256.
            # =========================================================
            rgb_pred = diffusion_sample(
                unet=sd_state["unet"],
                vae=sd_state["vae"],
                scheduler=sd_state["noise_scheduler"],
                cond_adapter=sd_state["cond_adapter"],
                ijepa_tokens=z_for_sd,
                text_embeds=sd_state.get("prompt_embeds"),
                pooled_text_embeds=sd_state.get("pooled_prompt_embeds"),
                num_steps=diffusion_vis_steps,
                image_size=(crop_size, crop_size),
                device=imgs_t.device,
                ref_rgb=imgs_t,  # Will be downsampled to 32x32 internally
            )
        
        return z_proj, h, rgb_pred
    
    # ========================================================================
    # Forward Pass with Generation (COARSE CONDITIONING)
    # ========================================================================
    
    def forward_pass_with_generation(
        imgs_t: torch.Tensor,
        embs_tp1: torch.Tensor,
        rgb_tp1: torch.Tensor,
        step: int,
    ):
        """
        Forward pass with real diffusion sampling for metrics.
        
        COARSE CONDITIONING:
        - rgb_source (256x256) is downsampled to 32x32 internally
        - 0.35 noise added
        - Output is 256x256, metrics computed at full resolution
        """
        
        B = imgs_t.size(0)
        num_patches = (crop_size // patch_size) ** 2
        
        full_mask = [
            torch.arange(num_patches, device=imgs_t.device)
            .unsqueeze(0).expand(B, -1)
        ]
        
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
            with torch.no_grad():
                if enable_ijepa:
                    z_enc = encoder(imgs_t, full_mask)
                    z_raw = predictor(z_enc, full_mask, full_mask)
                    z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                else:
                    encoder_embed_dim_local = encoder.module.embed_dim if distributed else encoder.embed_dim
                    z_raw_norm = torch.randn(B, num_patches, encoder_embed_dim_local, device=imgs_t.device, dtype=dtype)

                rgb_for_sd = rgb_tp1.to(dtype=torch.float32)
                rgb_source_sd = imgs_t.to(dtype=torch.float32)
                
                # =========================================================
                # COARSE CONDITIONING: rgb_source will be downsampled to 32x32
                # =========================================================
                _, sd_metrics, _ = compute_sd_loss_and_metrics(
                    sd_state=sd_state,
                    ijepa_tokens=z_raw_norm, 
                    rgb_target=rgb_for_sd,
                    rgb_source=rgb_source_sd,  # Will be downsampled to 32x32 internally
                    step=step,
                    save_vis=False,
                    vis_folder=vis_folder,
                    do_diffusion_sample=True,
                    diffusion_steps=diffusion_vis_steps,
                    ssim_weight=ssim_weight,
                )
        
        return None, None, None, sd_metrics
    
    # ========================================================================
    # Load Checkpoint
    # ========================================================================
    
    if load_model and load_path and os.path.exists(load_path):
        logger.info(f"Loading checkpoint: {load_path}")
        (
            encoder, predictor, target_encoder,
            optimizer, scaler, start_epoch
        ) = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        
        # Advance schedulers
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            global_step += 1
    
    # ========================================================================
    # Save Checkpoint (using unified format)
    # ========================================================================
    
    def save_checkpoint(epoch: int, val_loss: float, is_best: bool = False):
        if rank == 0:
            # Save latest
            save_full_checkpoint(
                save_path=latest_path,
                encoder=encoder,
                predictor=predictor,
                proj_head=proj_head,
                target_encoder=target_encoder,
                sd_state=sd_state,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=val_loss,
                config=args,
            )
            
            if is_best:
                save_full_checkpoint(
                    save_path=best_path,
                    encoder=encoder,
                    predictor=predictor,
                    proj_head=proj_head,
                    target_encoder=target_encoder,
                    sd_state=sd_state,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_loss=val_loss,
                    config=args,
                )
                logger.info(f"New best model! val_loss={val_loss:.4f}")

    # ========================================================================
    # Embedding Consistency Validator (UPDATED: PURE NOISE mode)
    # ========================================================================
    
    embedding_validator = None
    if rank == 0:
        emb_val_dir = os.path.join(folder, "embedding_validation")
        
        enc_unwrap = encoder.module if distributed else encoder
        pred_unwrap = predictor.module if distributed else predictor
        tgt_unwrap = target_encoder.module if distributed else target_encoder
        
        embedding_validator = EmbeddingConsistencyValidator(
            encoder=enc_unwrap,
            predictor=pred_unwrap,
            target_encoder=tgt_unwrap,
            sd_state=sd_state,
            device=device,
            output_dir=emb_val_dir,
            diffusion_steps=diffusion_vis_steps,
        )
        logger.info(f"[EmbeddingValidator] Initialized (COARSE RGB + Pure Noise) -> {emb_val_dir}")

    # ========================================================================
    # Health Check
    # ========================================================================

    if rank == 0:
        is_healthy = run_health_check(
            sd_state=sd_state,
            encoder=encoder,
            predictor=predictor,
            proj_head=proj_head,
            train_loader=train_loader,
            device=device
        )
        if not is_healthy:
            print("Shutting down due to Health Check failure.")
            return

    # ========================================================================
    # Training Loop
    # ========================================================================
    
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"IJEPA Training: {'ENABLED' if enable_ijepa else 'DISABLED (only SD trains)'}")
    logger.info(f"Conditioning Mode: COARSE RGB (32x32) + Pure Noise Latent")
    logger.info(f"Visualization every {vis_interval_iters} iterations")
    logger.info(f"Diffusion sampling steps for vis: {diffusion_vis_steps}")
    
    # ========================================================================
    # Temporal History Buffer for Training
    # ========================================================================
    # During training, we simulate temporal context by maintaining a rolling
    # buffer of adapter hidden states from previous iterations.
    # This teaches the temporal attention module to use history effectively.
    # The buffer stores detached tensors to avoid memory leaks.
    temporal_history_buffer = []
    TEMPORAL_MAX_HISTORY = 4  # Match the adapter's max_history
    
    for epoch in range(start_epoch, num_epochs):
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Clear temporal history at epoch start (data is reshuffled)
        temporal_history_buffer = []
        
        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Meters
        loss_meter = AverageMeter()
        loss_ijepa_meter = AverageMeter()
        loss_sd_meter = AverageMeter()
        contr_meter = AverageMeter() # InfoNCE loss tracker
        spatial_meter = AverageMeter() # Spatial Variance tracker
        feat_reg_meter = AverageMeter() # [NEW] Feature Regression Tracker
        time_meter = AverageMeter()
        
        encoder.train()
        predictor.train()
        proj_head.train()
        
        for itr, (batch_data, masks_enc, masks_pred) in enumerate(train_loader):
            
            # Load batch
            imgs_t = batch_data[0].to(device, non_blocking=True)
            embs_tp1 = batch_data[1].to(device, non_blocking=True)
            rgb_tp1 = batch_data[2].to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            
            def train_step():
                nonlocal global_step, temporal_history_buffer
                
                # 1. Update LR and Weight Decay schedules
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()
                
                # 2. Reset gradients (using set_to_none for better performance)
                optimizer.zero_grad(set_to_none=True)
                
                # 3. Forward pass with Flow Matching logic + temporal history
                # Randomly drop temporal history 30% of the time during training
                # so the model also learns to work without it (single-frame fallback)
                use_history = len(temporal_history_buffer) > 0 and torch.rand(1).item() > 0.3
                current_history = temporal_history_buffer if use_history else None
                
                total_loss, loss_ijepa, loss_sd, sd_metrics = forward_pass(
                    imgs_t, embs_tp1, rgb_tp1,
                    masks_enc, masks_pred,
                    step=global_step,
                    compute_sd=True,
                    history_hidden=current_history,
                )
                
                # Update temporal history buffer with current adapter output
                # We need to do a quick forward through adapter to get the hidden states
                with torch.no_grad():
                    adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
                    # Get current frame's IJEPA features for history
                    num_patches_local = (crop_size // patch_size) ** 2
                    full_mask_local = [torch.arange(num_patches_local, device=device).unsqueeze(0).expand(imgs_t.size(0), -1)]
                    z_enc_hist = encoder(imgs_t, full_mask_local)
                    z_pred_hist = predictor(z_enc_hist, full_mask_local, full_mask_local)
                    z_hist_norm = F.layer_norm(z_pred_hist, (z_pred_hist.size(-1),))
                    
                    coarse_rgb_hist = F.interpolate(imgs_t.float(), size=(32, 32), mode='bilinear', align_corners=False)
                    hist_hidden, _ = sd_state["cond_adapter"](
                        z_hist_norm.to(dtype=adapter_dtype), 
                        ref_rgb=coarse_rgb_hist.to(dtype=adapter_dtype),
                        history_hidden=None,  # Don't use history when creating history entry
                    )
                    
                    # Add to buffer (most recent first)
                    temporal_history_buffer.insert(0, hist_hidden.detach().cpu())
                    # Keep only max_history entries
                    if len(temporal_history_buffer) > TEMPORAL_MAX_HISTORY:
                        temporal_history_buffer = temporal_history_buffer[:TEMPORAL_MAX_HISTORY]
                
                # 4. Backward pass and Optimizer step with Gradient Clipping
                if use_bfloat16:
                    total_loss.backward()
                    
                    # Apply gradient clipping to all trainable parameters
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                        
                    optimizer.step()
                else:
                    if scaler:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        
                        # Apply gradient clipping before scaler.step
                        for group in optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                            
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        for group in optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
                        optimizer.step()
                
                # 5. Gradient statistics logging
                grad_stats = grad_logger(encoder.named_parameters())
                
                # 6. EMA (Exponential Moving Average) update for Target Encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for p_q, p_k in zip(encoder.parameters(), target_encoder.parameters()):
                        p_k.data.mul_(m).add_((1 - m) * p_q.detach().data)
                
                global_step += 1
                
                return (
                    total_loss.item(),
                    loss_ijepa.item(),
                    loss_sd.item(),
                    sd_metrics,
                    new_lr,
                    new_wd,
                    grad_stats,
                )
            
            (loss, l_ij, l_sd, sd_m, new_lr, new_wd, grad_stats), elapsed = gpu_timer(train_step)
            
            loss_meter.update(loss)
            loss_ijepa_meter.update(l_ij)
            loss_sd_meter.update(l_sd)
            contr_meter.update(sd_m.get('contr', 0))
            spatial_meter.update(sd_m.get('spatial', 0))
            feat_reg_meter.update(sd_m.get('feat_reg', 0)) # [NEW]
            time_meter.update(elapsed)
            
            # CSV log
            csv_logger.log(
                epoch + 1, itr, loss, l_ij, l_sd,
                len(masks_enc[0][0]), len(masks_pred[0][0]), elapsed
            )
            
            # Console log
            if itr % log_freq == 0 or np.isnan(loss):
                logger.info(
                    f"[{epoch+1},{itr:5d}] loss={loss_meter.avg:.4f} "
                    f"contr={contr_meter.avg:.4f} "
                    f"spatial={spatial_meter.avg:.4f} "
                    f"reg={feat_reg_meter.avg:.4f} " # [NEW]
                    f"lr={new_lr:.2e} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                    f"({time_meter.avg:.0f}ms)"
                )
                
                if sd_m:
                    m_str = " | ".join(f"{k}:{v:.4f}" for k, v in sd_m.items() if not np.isnan(v))
                    logger.info(f"[{epoch+1},{itr:5d}] SD: {m_str}")
                
                if grad_stats:
                    logger.info(
                        f"[{epoch+1},{itr:5d}] grads: first={grad_stats.first_layer:.2e} "
                        f"last={grad_stats.last_layer:.2e} "
                        f"range=({grad_stats.min:.2e}, {grad_stats.max:.2e})"
                    )
            
            # ================================================================
            # Visualization (PURE NOISE diffusion sampling)
            # ================================================================
            
            if rank == 0 and vis_interval_iters > 0 and (itr + 1) % vis_interval_iters == 0:
                logger.info(f"Generating COARSE CONDITIONING visualization with {diffusion_vis_steps}-step diffusion...")
                
                encoder.eval()
                predictor.eval()
                proj_head.eval()
                
                z_pred, h_gt, rgb_pred = generate_visualization(imgs_t, embs_tp1, rgb_tp1)
                
                vis_path = os.path.join(vis_folder, f"train_ep{epoch+1}_iter{itr+1}.png")
                save_visualization_grid(
                    rgb_t=imgs_t,
                    emb_gt=h_gt,
                    emb_pred=z_pred,
                    rgb_tp1_gt=rgb_tp1,
                    rgb_tp1_pred=rgb_pred,
                    save_path=vis_path,
                    crop_size=crop_size,
                )
                logger.info(f"Saved: {vis_path}")
                
                encoder.train()
                predictor.train()
                proj_head.train()
            
            assert not np.isnan(loss), "Loss is NaN!"
        
        # End of epoch
        logger.info(
            f"Epoch {epoch+1} complete. "
            f"Avg loss={loss_meter.avg:.4f} "
            f"(ijepa={loss_ijepa_meter.avg:.4f}, sd={loss_sd_meter.avg:.4f})"
        )
        
        # ====================================================================
        # Validation (every 5 epochs)
        # ====================================================================
        
        if (epoch + 1) % 5 != 0:
            logger.info("Skipping validation this epoch")
            continue
        
        logger.info("Running validation (COARSE CONDITIONING generation)...")
        
        encoder.eval()
        predictor.eval()
        proj_head.eval()
        
        val_loss_meter = AverageMeter()
        val_ijepa_meter = AverageMeter()
        val_sd_meter = AverageMeter()
        
        with torch.no_grad():
            for v_itr, (v_data, v_masks_enc, v_masks_pred) in enumerate(val_loader):
                
                v_imgs_t = v_data[0].to(device, non_blocking=True)
                v_embs_tp1 = v_data[1].to(device, non_blocking=True)
                v_rgb_tp1 = v_data[2].to(device, non_blocking=True)
                v_masks_enc = [m.to(device, non_blocking=True) for m in v_masks_enc]
                v_masks_pred = [m.to(device, non_blocking=True) for m in v_masks_pred]
                
                total_loss, l_ij, l_sd, val_sd_metrics = forward_pass(
                    v_imgs_t, v_embs_tp1, v_rgb_tp1,
                    v_masks_enc, v_masks_pred,
                    step=global_step + v_itr,
                    compute_sd=True,
                )
                
                val_loss_meter.update(total_loss.item())
                val_ijepa_meter.update(l_ij.item())
                val_sd_meter.update(l_sd.item())
                
                # Log generation metrics for first batch
                if v_itr == 0 and rank == 0:
                    # Run with diffusion sampling to get real generation metrics
                    _, _, _, gen_metrics = forward_pass_with_generation(
                        v_imgs_t, v_embs_tp1, v_rgb_tp1,
                        step=global_step,
                    )
                    
                    if gen_metrics:
                        gen_str = " | ".join(f"{k}:{v:.4f}" for k, v in gen_metrics.items() if 'gen_' in k)
                        if gen_str:
                            logger.info(f"[Val] COARSE CONDITIONING Generation metrics: {gen_str}")
                
                # Save validation visualization (first batch only)
                if v_itr == 0 and rank == 0:
                    z_pred, h_gt, rgb_pred = generate_visualization(
                        v_imgs_t, v_embs_tp1, v_rgb_tp1
                    )
                    
                    vis_path = os.path.join(vis_folder, f"val_ep{epoch+1}.png")
                    save_visualization_grid(
                        rgb_t=v_imgs_t,
                        emb_gt=h_gt,
                        emb_pred=z_pred,
                        rgb_tp1_gt=v_rgb_tp1,
                        rgb_tp1_pred=rgb_pred,
                        save_path=vis_path,
                        crop_size=crop_size,
                    )
                    logger.info(f"Saved validation vis: {vis_path}")
        
        # ====================================================================
        # Embedding Consistency Validation
        # ====================================================================
        
        if rank == 0 and embedding_validator is not None:
            logger.info("Running embedding consistency validation (COARSE CONDITIONING)...")
            try:
                emb_results = embedding_validator.validate(
                    dataloader=val_loader,
                    epoch=epoch + 1,
                    max_batches=30,
                )
                logger.info(
                    f"[EmbVal] cos(gen, gt)={emb_results['cos_gen_gt_mean']:.4f}, "
                    f"cos(random, gt)={emb_results['cos_random_gt_mean']:.4f}"
                )
            except Exception as e:
                logger.error(f"[EmbVal] Failed: {e}")
        
        val_loss = val_loss_meter.avg
        logger.info(
            f"Validation: loss={val_loss:.4f} "
            f"(ijepa={val_ijepa_meter.avg:.4f}, sd={val_sd_meter.avg:.4f})"
        )
        
        val_csv_logger.log(epoch + 1, val_loss, val_ijepa_meter.avg, val_sd_meter.avg)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch + 1, val_loss, is_best=is_best)
        
        encoder.train()
        predictor.train()
        proj_head.train()
    
    # ========================================================================
    # Training Complete
    # ========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {folder}")
    logger.info("=" * 60)


if __name__ == "__main__":
    raise RuntimeError(
        "Use main.py as entry point: python main.py --fname config.yaml"
    )