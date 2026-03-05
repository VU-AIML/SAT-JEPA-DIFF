"""
Evaluate Custom IJEPA + Stable Diffusion Model (VRAM Optimized)

This script loads a trained hybrid model (IJEPA + SD Adapter) and evaluates it on the validation set.
It calculates standard metrics (PSNR, SSIM, MSE, L1) and advanced perceptual metrics (FID, LPIPS).
"""

import os
import argparse
import json
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import local modules
from helper import init_model
from sd_models import load_sd_model
from sd_joint_loss import diffusion_sample
from data.data import S2FutureEmbeddingDataset

# Try importing advanced metric libraries
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: 'lpips' library not found. LPIPS metric will be skipped.")
    LPIPS_AVAILABLE = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    print("Warning: 'torchmetrics' library not found. FID metric will be skipped.")
    FID_AVAILABLE = False


# ============================================================================
# Metric Helper Functions
# ============================================================================

def gaussian_kernel(size=11, sigma=1.5, channels=3, device=None):
    """Creates a Gaussian kernel for SSIM calculation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11) -> float:
    """Calculates Structural Similarity Index (SSIM)."""
    # Metrics calculation should be in float32 for accuracy
    img1 = img1.float()
    img2 = img2.float()
    
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    C = img1.size(1)
    kernel = gaussian_kernel(window_size, 1.5, C, img1.device)
    
    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=C)
    
    sigma1_sq = F.conv2d(img1*img1, kernel, padding=window_size//2, groups=C) - mu1**2
    sigma2_sq = F.conv2d(img2*img2, kernel, padding=window_size//2, groups=C) - mu2**2
    sigma12 = F.conv2d(img1*img2, kernel, padding=window_size//2, groups=C) - mu1*mu2
    
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    img1 = img1.float()
    img2 = img2.float()
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def compute_gssim(img1: torch.Tensor, img2: torch.Tensor, scales: list = [1.0, 0.5, 0.25]) -> float:
    """Calculates Gradient-based SSIM."""
    img1 = img1.float()
    img2 = img2.float()
    
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
    
    ssim_scores = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = int(img1.shape[2] * scale), int(img1.shape[3] * scale)
            img1_scaled = F.interpolate(img1, size=(h, w), mode='bilinear', align_corners=False)
            img2_scaled = F.interpolate(img2, size=(h, w), mode='bilinear', align_corners=False)
        else:
            img1_scaled = img1
            img2_scaled = img2
        
        gray1 = img1_scaled.mean(dim=1, keepdim=True)
        gray2 = img2_scaled.mean(dim=1, keepdim=True)
        
        grad1_x = F.conv2d(gray1, sobel_x, padding=1)
        grad1_y = F.conv2d(gray1, sobel_y, padding=1)
        grad2_x = F.conv2d(gray2, sobel_x, padding=1)
        grad2_y = F.conv2d(gray2, sobel_y, padding=1)
        
        grad1 = torch.sqrt(grad1_x**2 + grad1_y**2 + 1e-8)
        grad2 = torch.sqrt(grad2_x**2 + grad2_y**2 + 1e-8)
        
        grad1 = (grad1 - grad1.min()) / (grad1.max() - grad1.min() + 1e-8)
        grad2 = (grad2 - grad2.min()) / (grad2.max() - grad2.min() + 1e-8)
        
        grad1_3ch = grad1.repeat(1, 3, 1, 1)
        grad2_3ch = grad2.repeat(1, 3, 1, 1)
        ssim_scores.append(compute_ssim(grad1_3ch, grad2_3ch))
    
    return np.mean(ssim_scores)

def compute_metrics_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Computes all pixel-based metrics for a batch."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    
    return {
        "l1": F.l1_loss(pred.float(), target.float()).item(),
        "mse": F.mse_loss(pred.float(), target.float()).item(),
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "gssim": compute_gssim(pred, target),
    }

# ============================================================================
# Model Loading
# ============================================================================

def load_full_model_for_eval(checkpoint_path: str, device: torch.device):
    """
    Loads the full IJEPA + SD model state from the checkpoint.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    torch.cuda.empty_cache() # Clear cache before loading
    
    # Load with weights_only=False to allow loading the config dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 1. Recover Configuration
    config = checkpoint.get("config", None)
    if config is None:
        raise ValueError("Checkpoint does not contain 'config'. Cannot reconstruct model.")
    
    print("Model configuration found.")
    
    # Extract params
    patch_size = config["mask"]["patch_size"]
    crop_size = config["data"]["crop_size"]
    pred_depth = config["meta"]["pred_depth"]
    pred_emb_dim = config["meta"]["pred_emb_dim"]
    model_name = config["meta"]["model_name"]
    use_lora = config["meta"].get("use_lora", True)
    lora_rank = int(config["meta"].get("lora_rank", 16))
    lora_alpha = int(config["meta"].get("lora_alpha", 32))
    
    # 2. Initialize IJEPA
    print("Initializing IJEPA...")
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    
    # Determine embedding dimension
    if hasattr(encoder, "embed_dim"):
        encoder_embed_dim = encoder.embed_dim
    else:
        encoder_embed_dim = 1280 # Fallback
    
    # 3. Initialize Stable Diffusion
    print(f"Initializing Stable Diffusion (Target Dim: {encoder_embed_dim})...")
    
    # FIX: Use bfloat16 or float16 to save memory (avoid OOM)
    # Most modern GPUs support bfloat16, fallback to float16 if needed.
    eval_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype for SD: {eval_dtype}")
    
    sd_state = load_sd_model(
        device=device,
        checkpoint_dir=config["meta"].get("sd_checkpoint_dir", "./sd_finetuned"),
        dtype=eval_dtype, 
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_emb_dim=encoder_embed_dim
    )
    
    # VAE can stay in float32 for stability if needed, or cast to eval_dtype for memory
    # SD 3.5 VAE is robust, let's keep it float32 only if we have issues, otherwise eval_dtype
    if "vae" in sd_state:
        # Keeping VAE in float32 is safer for decoding without NaN, 
        # but if OOM persists, change this to eval_dtype
        sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    # 4. Load State Dicts
    print("Loading weights...")
    
    # Load IJEPA weights
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    if "predictor" in checkpoint:
        predictor.load_state_dict(checkpoint["predictor"])
        
    # Load SD Trainable Params (LoRA + Adapter)
    if "sd_state" in checkpoint:
        # Load Adapter
        if "cond_adapter" in sd_state and "cond_adapter" in checkpoint:
             sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"])
             
        # Load UNet (LoRA)
        if "unet" in checkpoint:
            sd_state["unet"].load_state_dict(checkpoint["unet"])
        elif "unet_lora" in checkpoint:
             sd_state["unet"].load_state_dict(checkpoint["unet_lora"], strict=False)

    encoder.to(device).eval()
    predictor.to(device).eval()
    
    # Convert IJEPA to eval_dtype as well to save memory and match SD
    encoder.to(dtype=eval_dtype)
    predictor.to(dtype=eval_dtype)
    
    torch.cuda.empty_cache() # Cleanup after loading
    
    return encoder, predictor, sd_state, config, eval_dtype

# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate(
    checkpoint_path: str,
    root_dir: str,
    batch_size: int = 4,
    device: str = "cuda",
    num_samples: int = 1000,
    diffusion_steps: int = 50,
):
    """
    Main evaluation loop.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model (with memory optimization)
    encoder, predictor, sd_state, config, eval_dtype = load_full_model_for_eval(checkpoint_path, device)
    
    # 2. Setup Metrics
    # LPIPS and FID usually expect float32 inputs, we will handle casting during the loop.
    lpips_fn = None
    if LPIPS_AVAILABLE:
        print("Initializing LPIPS (VGG)...")
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_fn.eval()

    fid_metric = None
    if FID_AVAILABLE:
        print("Initializing FID (InceptionV3)...")
        # feature=64 is faster
        fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    
    # 3. Load Data
    print("Loading dataset...")
    crop_size = config["data"]["crop_size"]
    patch_size = config["mask"]["patch_size"]
    
    # Use the same dataset class as training
    dataset = S2FutureEmbeddingDataset(
        root_dir=root_dir,
        patch_size=crop_size,
        transform=None,
    )
    
    # Create validation split 
    total_len = len(dataset)
    val_indices = list(range(int(0.8 * total_len), total_len))
    val_dataset = Subset(dataset, val_indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Evaluation limit: {num_samples} samples")
    print(f"Diffusion Steps: {diffusion_steps}")
    
    # 4. Evaluation Loop
    total_metrics = {
        "l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "gssim": 0.0, "lpips": 0.0
    }
    num_evaluated = 0
    
    # Helper for full masks
    num_patches = (crop_size // patch_size) ** 2
    
    print("Starting evaluation...")
    with torch.no_grad():
        # Use autocast to handle mixed precision efficiently
        with torch.amp.autocast("cuda", dtype=eval_dtype):
            for batch_data in tqdm(val_loader, desc="Generating"):
                if num_evaluated >= num_samples:
                    break
                
                if isinstance(batch_data, list):
                     imgs_t = batch_data[0].to(device)
                     rgb_tp1 = batch_data[2].to(device)
                else:
                     imgs_t = batch_data[0].to(device)
                     rgb_tp1 = batch_data[2].to(device)

                B = imgs_t.size(0)
                
                # Create Full Mask
                full_mask = [
                    torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)
                ]
                
                # --- Inference Step ---
                
                # 1. IJEPA (latent generation)
                # Ensure input is in correct dtype
                z_enc = encoder(imgs_t.to(dtype=eval_dtype), full_mask)
                z_raw = predictor(z_enc, full_mask, full_mask)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                
                # 2. Stable Diffusion Generation
                # Pass tensors in correct dtype. diffusion_sample handles VAE casting internally.
                rgb_pred = diffusion_sample(
                    unet=sd_state["unet"],
                    vae=sd_state["vae"],
                    scheduler=sd_state["noise_scheduler"],
                    cond_adapter=sd_state["cond_adapter"],
                    ijepa_tokens=z_raw_norm,
                    text_embeds=sd_state.get("prompt_embeds"),
                    pooled_text_embeds=sd_state.get("pooled_prompt_embeds"),
                    num_steps=diffusion_steps,
                    image_size=(crop_size, crop_size),
                    device=device,
                    ref_rgb=imgs_t.to(dtype=torch.float32), # ref_rgb needs to be float32 for interpolation usually
                )
                
                # --- Metrics Calculation (Move to float32 for accuracy) ---
                
                rgb_gt = rgb_tp1.to(dtype=torch.float32).clamp(0, 1)
                rgb_pred = rgb_pred.to(dtype=torch.float32).clamp(0, 1)
                
                current_batch_size = rgb_pred.size(0)
                
                for i in range(current_batch_size):
                    m = compute_metrics_batch(rgb_pred[i], rgb_gt[i])
                    for k, v in m.items():
                        total_metrics[k] += v
                
                # LPIPS
                if lpips_fn is not None:
                    # Scale to [-1, 1]
                    pred_norm = (rgb_pred * 2.0) - 1.0
                    gt_norm = (rgb_gt * 2.0) - 1.0
                    batch_lpips = lpips_fn(pred_norm, gt_norm)
                    total_metrics["lpips"] += batch_lpips.sum().item()
                
                # FID
                if fid_metric is not None:
                    fid_metric.update(rgb_gt, real=True)
                    fid_metric.update(rgb_pred, real=False)
                
                num_evaluated += current_batch_size
            
    # 5. Finalize Results
    avg_metrics = {k: v / max(num_evaluated, 1) for k, v in total_metrics.items()}
    
    fid_score = 0.0
    if fid_metric is not None:
        print("Computing FID score...")
        fid_score = fid_metric.compute().item()
    avg_metrics["fid"] = fid_score
    
    # 6. Report
    print("\n" + "="*40)
    print("Final Evaluation Results")
    print("="*40)
    print(f"Model: {checkpoint_path}")
    print(f"Samples evaluated: {num_evaluated}")
    print("-" * 40)
    print(f"L1 Loss:      {avg_metrics['l1']:.4f}")
    print(f"MSE Loss:     {avg_metrics['mse']:.4f}")
    print(f"PSNR:         {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM:         {avg_metrics['ssim']:.4f}")
    print(f"GSSIM:        {avg_metrics['gssim']:.4f}")
    
    if LPIPS_AVAILABLE:
        print(f"LPIPS:        {avg_metrics['lpips']:.4f} (Lower is better)")
    else:
        print("LPIPS:        N/A")
        
    if FID_AVAILABLE:
        print(f"FID:          {avg_metrics['fid']:.4f} (Lower is better)")
    else:
        print("FID:          N/A")
    print("="*40)

    # Save
    output_path = checkpoint_path.replace(".pth.tar", "_eval_metrics.json")
    if output_path == checkpoint_path:
        output_path = checkpoint_path + "_eval_metrics.json"
        
    with open(output_path, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Custom IJEPA-SD Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--root_dir", type=str, default="downloads", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion sampling steps")
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        diffusion_steps=args.steps
    )