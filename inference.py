"""
Sat-JEPA-Diff Inference Script

Bu script, eğitilmiş modeli kullanarak belirli bölgeler için
t -> t+1 tahminleri yapar ve görselleştirir.

Çıktı: 3 sütunlu görüntü grid'i
  - Sol: Original t
  - Orta: Ground Truth t+1
  - Sağ: Predicted t+1

Kullanım:
    python inference_satjepa.py \
        --checkpoint ./experiments/best_model.pth.tar \
        --output_dir ./inference_results
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm

# Model imports
from helper import init_model
from sd_models import load_sd_model, IJEPAConditioningAdapter
from sd_joint_loss import diffusion_sample, COARSE_SIZE


# =============================================================================
# Region Paths
# =============================================================================

REGION_PATHS = [
    "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Istanbul_Turkey/s2_rgb_Istanbul_Turkey_2017_10km.tif",
    "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Amazon_Deforestation_Arc_Brazil/s2_rgb_Amazon_Deforestation_Arc_Brazil_2017_10km.tif",
    "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Corn_Belt_Iowa_USA/s2_rgb_Corn_Belt_Iowa_USA_2017_10km.tif",
    "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Sahara_Tamanrasset_Algeria/s2_rgb_Sahara_Tamanrasset_Algeria_2017_10km.tif",
    "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Mekong_Delta_Vietnam/s2_rgb_Mekong_Delta_Vietnam_2019_10km.tif"
]


# =============================================================================
# Utilities
# =============================================================================

def extract_info_from_path(path: str) -> Dict:
    """Extract region name and year from file path."""
    basename = os.path.basename(path)
    # Pattern: s2_rgb_{region}_{year}_10km.tif
    match = re.match(r"s2_rgb_(.+)_(\d{4})_10km\.tif", basename)
    if match:
        return {
            "region": match.group(1),
            "year": int(match.group(2)),
            "dir": os.path.dirname(path),
        }
    return {"region": "unknown", "year": 0, "dir": os.path.dirname(path)}


def get_next_year_path(path: str) -> Optional[str]:
    """Get path for t+1 year image."""
    info = extract_info_from_path(path)
    next_year = info["year"] + 1
    next_path = os.path.join(
        info["dir"],
        f"s2_rgb_{info['region']}_{next_year}_10km.tif"
    )
    if os.path.exists(next_path):
        return next_path
    return None


def load_and_resize_tif(path: str, target_size: int = 128, debug: bool = True) -> torch.Tensor:
    """
    Load GeoTIFF, resize to target_size using rasterio.
    Returns tensor with shape (3, H, W).
    
    IMPORTANT: Training uses NO explicit normalization - raw float32 values are used directly.
    The TIF files should already be in [0, 1] range or the model was trained on raw values.
    """
    from rasterio.enums import Resampling
    
    with rasterio.open(path) as src:
        # Read and resize simultaneously (efficient)
        arr = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear
        )
        dtype = src.profile['dtype']
        
        # If image has > 3 channels, take first 3 (RGB)
        if arr.shape[0] > 3:
            arr = arr[:3, :, :]
        
        arr = arr.astype(np.float32)
        
        if debug:
            print(f"    Raw TIF: dtype={dtype}, shape={arr.shape}, min={arr.min():.1f}, max={arr.max():.1f}")
        
        # SAME AS TRAINING: No normalization if already float, else normalize
        # Check if data needs normalization
        max_val = arr.max()
        
        if dtype in ['float32', 'float64']:
            # Already float - check if in reasonable range
            if max_val <= 1.0:
                # Already normalized
                pass
            elif max_val <= 10.0:
                # Might be log-scaled or similar
                pass  
            else:
                # Large float values - normalize
                arr = arr / 10000.0
        elif max_val > 255:
            # 16-bit integer data - Sentinel-2 typical range 0-10000
            arr = arr / 10000.0
        elif max_val > 1.0:
            # 8-bit integer data
            arr = arr / 255.0
        
        # Clamp to [0, 1] as done in training visualization
        arr = np.clip(arr, 0, 1)
        
        if debug:
            print(f"    After processing: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
        
        # To Tensor (C, H, W)
        tensor = torch.from_numpy(arr)
        
        return tensor


def enhance_for_display(img: torch.Tensor, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """
    Enhance image for better display using percentile clipping.
    Same method as PredRNN inference for consistency.
    
    Args:
        img: (C, H, W) tensor in [0, 1] range
        p_low: Low percentile (default 2)
        p_high: High percentile (default 98)
    
    Returns:
        Enhanced numpy array (H, W, C) in [0, 1] range
    """
    img_np = img.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    
    # Calculate robust min and max
    p2, p98 = np.percentile(img_np, (p_low, p_high))
    
    # Clip and scale
    img_np = np.clip(img_np, p2, p98)
    
    # Avoid division by zero
    if p98 > p2:
        img_np = (img_np - p2) / (p98 - p2)
    else:
        img_np = np.zeros_like(img_np)
    
    return img_np.clip(0, 1)


def save_comparison_image(
    img_t: torch.Tensor,
    img_tp1_gt: torch.Tensor,
    img_tp1_pred: torch.Tensor,
    save_path: str,
    titles: Tuple[str, str, str] = ("Input (t)", "Ground Truth (t+1)", "Predicted (t+1)"),
):
    """
    Save 3-column comparison image.
    All inputs should be (C, H, W) tensors in [0, 1] range.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images = [img_t, img_tp1_gt, img_tp1_pred]
    
    for ax, img, title in zip(axes, images, titles):
        img_np = enhance_for_display(img)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_grid_image(
    img_t: torch.Tensor,
    img_tp1_gt: torch.Tensor,
    img_tp1_pred: torch.Tensor,
    save_path: str,
):
    """
    Save simple grid image without matplotlib (just concatenate).
    """
    # Enhance images
    img_t_np = enhance_for_display(img_t)
    img_tp1_gt_np = enhance_for_display(img_tp1_gt)
    img_tp1_pred_np = enhance_for_display(img_tp1_pred)
    
    H, W, _ = img_t_np.shape
    
    # Add small white padding between images
    pad = 4
    padding = np.ones((H, pad, 3))
    
    # Concatenate horizontally
    grid = np.concatenate([
        img_t_np,
        padding,
        img_tp1_gt_np,
        padding,
        img_tp1_pred_np,
    ], axis=1)
    
    # Convert to PIL and save
    grid_uint8 = (grid * 255).astype(np.uint8)
    Image.fromarray(grid_uint8).save(save_path)
    print(f"  Saved: {save_path}")


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
    patch_size: int = 8,
    crop_size: int = 128,
    pred_depth: int = 6,  # Default changed to 6 (common value)
    pred_emb_dim: int = 384,
    model_name: str = "vit_base",
    lora_rank: int = 8,  # Default changed to 8
    lora_alpha: int = 16,  # Default changed to 16
):
    """Load all model components from checkpoint."""
    
    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)
    
    # 1. First load checkpoint to get config
    print(f"[1/4] Loading checkpoint config: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to get config from checkpoint
    config = checkpoint.get("config", {})
    
    # Override parameters from checkpoint config if available
    if config:
        patch_size = config.get("patch_size", patch_size)
        crop_size = config.get("crop_size", crop_size)
        pred_depth = config.get("pred_depth", pred_depth)
        pred_emb_dim = config.get("pred_emb_dim", pred_emb_dim)
        model_name = config.get("model_name", model_name)
        lora_rank = config.get("lora_rank", lora_rank)
        lora_alpha = config.get("lora_alpha", lora_alpha)
        print(f"  Config from checkpoint:")
        print(f"    patch_size: {patch_size}")
        print(f"    crop_size: {crop_size}")
        print(f"    pred_depth: {pred_depth}")
        print(f"    pred_emb_dim: {pred_emb_dim}")
        print(f"    model_name: {model_name}")
        print(f"    lora_rank: {lora_rank}")
        print(f"    lora_alpha: {lora_alpha}")
    else:
        # Try to infer pred_depth from checkpoint keys
        predictor_keys = [k for k in checkpoint.get("predictor", {}).keys() if "predictor_blocks" in k]
        if predictor_keys:
            block_nums = [int(k.split(".")[1]) for k in predictor_keys if k.split(".")[1].isdigit()]
            if block_nums:
                inferred_depth = max(block_nums) + 1
                print(f"  Inferred pred_depth from checkpoint: {inferred_depth}")
                pred_depth = inferred_depth
        
        # Try to infer lora_rank from checkpoint lora weights
        lora_state = checkpoint.get("lora_state_dict", {})
        if lora_state:
            for key, tensor in lora_state.items():
                if "lora_A" in key:
                    inferred_rank = tensor.shape[0]
                    print(f"  Inferred lora_rank from checkpoint: {inferred_rank}")
                    lora_rank = inferred_rank
                    lora_alpha = inferred_rank * 2  # Common default
                    break
    
    # 2. Initialize IJEPA encoder and predictor with correct config
    print("[2/4] Initializing IJEPA...")
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    
    # Get encoder dimension
    encoder_embed_dim = getattr(encoder, "embed_dim", 768)
    print(f"  Encoder embed_dim: {encoder_embed_dim}")
    
    # 3. Load SD model with correct LoRA config
    print(f"[3/4] Loading Stable Diffusion 3.5 (LoRA rank={lora_rank}, alpha={lora_alpha})...")
    sd_state = load_sd_model(
        device=device,
        dtype=torch.float16,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_emb_dim=encoder_embed_dim,
        load_text_encoders=False,  # Skip text encoders for inference
    )
    
    # Force VAE to float32 for stability
    sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)
    
    # 4. Load weights from checkpoint
    print(f"[4/4] Loading model weights...")
    
    # Load encoder
    encoder.load_state_dict(checkpoint["encoder"])
    print("  ✓ Encoder loaded")
    
    # Load predictor
    predictor.load_state_dict(checkpoint["predictor"])
    print("  ✓ Predictor loaded")
    
    # Load conditioning adapter
    sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"])
    print("  ✓ Conditioning adapter loaded")
    
    # Load LoRA weights if available
    if checkpoint.get("use_lora") and "lora_state_dict" in checkpoint:
        sd_state["unet"].load_state_dict(checkpoint["lora_state_dict"], strict=False)
        print("  ✓ LoRA weights loaded")
    
    # Load prompt embeddings if available
    if "prompt_embeds" in checkpoint:
        unet_dtype = next(sd_state["unet"].parameters()).dtype
        sd_state["prompt_embeds"] = checkpoint["prompt_embeds"].to(device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = checkpoint["pooled_prompt_embeds"].to(device=device, dtype=unet_dtype)
        print("  ✓ Prompt embeddings loaded")
    else:
        # Use zero embeddings
        unet_dtype = next(sd_state["unet"].parameters()).dtype
        sd_state["prompt_embeds"] = torch.zeros(1, 77, 4096, device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = torch.zeros(1, 2048, device=device, dtype=unet_dtype)
        print("  ⚠ Using zero prompt embeddings")
    
    epoch = checkpoint.get("epoch", "unknown")
    print(f"\n  Checkpoint epoch: {epoch}")
    
    # Set to eval mode
    encoder.eval()
    predictor.eval()
    sd_state["cond_adapter"].eval()
    sd_state["unet"].eval()
    sd_state["vae"].eval()
    
    return encoder, predictor, sd_state, encoder_embed_dim


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_next_frame(
    rgb_t: torch.Tensor,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    sd_state: Dict,
    device: torch.device,
    patch_size: int = 8,
    num_diffusion_steps: int = 20,
    noise_strength: float = 0.35,
    debug: bool = True,
) -> torch.Tensor:
    """
    Predict t+1 image from t image.
    
    Args:
        rgb_t: Input image at time t, shape (3, H, W), range [0, 1]
        encoder: IJEPA encoder
        predictor: IJEPA predictor
        sd_state: Stable Diffusion state dict
        device: torch device
        patch_size: ViT patch size
        num_diffusion_steps: Number of diffusion sampling steps
        noise_strength: img2img noise strength
    
    Returns:
        Predicted image at t+1, shape (3, H, W), range [0, 1]
    """
    # Add batch dimension
    rgb_t = rgb_t.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    B, C, H, W = rgb_t.shape
    num_patches = (H // patch_size) ** 2
    
    if debug:
        print(f"    Input rgb_t: shape={rgb_t.shape}, min={rgb_t.min():.3f}, max={rgb_t.max():.3f}")
    
    # Get encoder dtype
    enc_dtype = next(encoder.parameters()).dtype
    rgb_t_enc = rgb_t.to(dtype=enc_dtype)
    
    # Create full mask (attend to all patches)
    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0)]
    
    # IJEPA forward: encode t, predict t+1 embedding
    z_enc = encoder(rgb_t_enc, full_mask)
    z_pred = predictor(z_enc, full_mask, full_mask)
    
    if debug:
        print(f"    z_enc: shape={z_enc.shape}, min={z_enc.min():.3f}, max={z_enc.max():.3f}")
        print(f"    z_pred (before norm): shape={z_pred.shape}, min={z_pred.min():.3f}, max={z_pred.max():.3f}")
    
    # Normalize for SD
    z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
    
    if debug:
        print(f"    z_pred (after norm): min={z_pred.min():.3f}, max={z_pred.max():.3f}")
    
    # Get adapter dtype
    adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
    z_pred = z_pred.to(dtype=adapter_dtype)
    
    # Generate image using diffusion
    gen_rgb = diffusion_sample(
        unet=sd_state["unet"],
        vae=sd_state["vae"],
        scheduler=sd_state["noise_scheduler"],
        cond_adapter=sd_state["cond_adapter"],
        ijepa_tokens=z_pred,
        text_embeds=sd_state.get("prompt_embeds"),
        pooled_text_embeds=sd_state.get("pooled_prompt_embeds"),
        num_steps=num_diffusion_steps,
        image_size=(H, W),
        device=device,
        ref_rgb=rgb_t.to(torch.float32),
        coarse_size=COARSE_SIZE,
        noise_strength=noise_strength,
    )
    
    if debug:
        print(f"    gen_rgb: shape={gen_rgb.shape}, min={gen_rgb.min():.3f}, max={gen_rgb.max():.3f}")
    
    return gen_rgb.squeeze(0)  # (3, H, W)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sat-JEPA-Diff Inference")
    
    # Paths
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", "-o", type=str, default="./inference_results",
                        help="Output directory for results")
    
    # Model config
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--pred_depth", type=int, default=6)
    parser.add_argument("--pred_emb_dim", type=int, default=384)
    parser.add_argument("--model_name", type=str, default="vit_base")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    # Inference config
    parser.add_argument("--diffusion_steps", type=int, default=20,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--noise_strength", type=float, default=0.35,
                        help="Img2img noise strength")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    encoder, predictor, sd_state, encoder_embed_dim = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    
    # Process each region
    print("\n" + "=" * 60)
    print("Running Inference...")
    print("=" * 60)
    
    results = []
    
    for region_path in REGION_PATHS:
        info = extract_info_from_path(region_path)
        region_name = info["region"]
        year_t = info["year"]
        
        print(f"\n[{region_name}] Year {year_t} -> {year_t + 1}")
        
        # Check if t+1 exists
        next_year_path = get_next_year_path(region_path)
        if next_year_path is None:
            print(f"  ⚠ Skipping: No t+1 image found")
            continue
        
        # Load images
        print(f"  Loading images...")
        try:
            rgb_t = load_and_resize_tif(region_path, args.crop_size)
            rgb_tp1_gt = load_and_resize_tif(next_year_path, args.crop_size)
        except Exception as e:
            print(f"  ⚠ Error loading images: {e}")
            continue
        
        # Run prediction
        print(f"  Predicting t+1...")
        rgb_tp1_pred = predict_next_frame(
            rgb_t=rgb_t,
            encoder=encoder,
            predictor=predictor,
            sd_state=sd_state,
            device=device,
            patch_size=args.patch_size,
            num_diffusion_steps=args.diffusion_steps,
            noise_strength=args.noise_strength,
        )
        
        # Compute metrics
        rgb_tp1_pred_cpu = rgb_tp1_pred.cpu()
        l1 = F.l1_loss(rgb_tp1_pred_cpu, rgb_tp1_gt).item()
        mse = F.mse_loss(rgb_tp1_pred_cpu, rgb_tp1_gt).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        print(f"  Metrics: L1={l1:.4f}, MSE={mse:.4f}, PSNR={psnr:.2f}")
        
        results.append({
            "region": region_name,
            "year_t": year_t,
            "l1": l1,
            "mse": mse,
            "psnr": psnr,
        })
        
        # Save comparison image
        save_name = f"{region_name}_{year_t}_to_{year_t+1}.png"
        save_path = os.path.join(args.output_dir, save_name)
        
        save_comparison_image(
            img_t=rgb_t,
            img_tp1_gt=rgb_tp1_gt,
            img_tp1_pred=rgb_tp1_pred_cpu,
            save_path=save_path,
            titles=(
                f"Input ({year_t})",
                f"Ground Truth ({year_t + 1})",
                f"Predicted ({year_t + 1})"
            ),
        )
        
        # Also save simple grid
        grid_path = os.path.join(args.output_dir, f"{region_name}_{year_t}_grid.png")
        save_grid_image(rgb_t, rgb_tp1_gt, rgb_tp1_pred_cpu, grid_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        avg_l1 = np.mean([r["l1"] for r in results])
        avg_mse = np.mean([r["mse"] for r in results])
        avg_psnr = np.mean([r["psnr"] for r in results])
        
        print(f"\nAverage Metrics ({len(results)} regions):")
        print(f"  L1:   {avg_l1:.4f}")
        print(f"  MSE:  {avg_mse:.4f}")
        print(f"  PSNR: {avg_psnr:.2f}")
        
        print("\nPer-Region Results:")
        for r in results:
            print(f"  {r['region']}: L1={r['l1']:.4f}, PSNR={r['psnr']:.2f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()