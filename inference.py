"""
Sat-JEPA-Diff Inference Script (Multi-Caption Architecture)

Bu script, eğitilmiş modeli kullanarak belirli bölgeler için
t -> t+1 tahminleri yapar ve görselleştirir.

Mimari:
- IJEPA encoder → visual tokens at t+1
- CaptionForecaster → semantic caption at t+1
- SD3.5 conditioned on: informative(t) + geometric(t) + semantic(t+1) + IJEPA tokens
- No coarse RGB — pure embedding + caption conditioning

Çıktı: 3 sütunlu görüntü grid'i
  - Sol: Original t
  - Orta: Ground Truth t+1
  - Sağ: Predicted t+1

Kullanım:
    python inference.py \\
        --checkpoint ./experiments/best_model.pth.tar \\
        --output_dir ./inference_results

    # Caption'larla birlikte (daha kaliteli tahmin):
    python inference.py \\
        --checkpoint ./experiments/best_model.pth.tar \\
        --caption_dir ./captions \\
        --output_dir ./inference_results
"""

import os
import re
import json
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
from sd_models import load_sd_model, encode_caption_batch
from sd_joint_loss import diffusion_sample
from caption_forecaster import CaptionForecaster


# =============================================================================
# Region Paths
# =============================================================================

REGION_PATHS = [
    "/media/kursat/Toshiba_2tb/projects/satellite/ijepa/downloads/Istanbul_Turkey/s2_rgb_Istanbul_Turkey_2017_10km.tif",
    "/media/kursat/Toshiba_2tb/projects/satellite/ijepa/downloads/Amazon_Deforestation_Arc_Brazil/s2_rgb_Amazon_Deforestation_Arc_Brazil_2017_10km.tif",
    "/media/kursat/Toshiba_2tb/projects/satellite/ijepa/downloads/Corn_Belt_Iowa_USA/s2_rgb_Corn_Belt_Iowa_USA_2017_10km.tif",
    "/media/kursat/Toshiba_2tb/projects/satellite/ijepa/downloads/Sahara_Tamanrasset_Algeria/s2_rgb_Sahara_Tamanrasset_Algeria_2017_10km.tif",
    "/media/kursat/Toshiba_2tb/projects/satellite/ijepa/downloads/Mekong_Delta_Vietnam/s2_rgb_Mekong_Delta_Vietnam_2019_10km.tif"
]


# =============================================================================
# Utilities
# =============================================================================

def extract_info_from_path(path: str) -> Dict:
    """Extract region name and year from file path."""
    basename = os.path.basename(path)
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
    """
    with rasterio.open(path) as src:
        arr = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear
        )
        dtype = src.profile['dtype']

        if arr.shape[0] > 3:
            arr = arr[:3, :, :]

        arr = arr.astype(np.float32)

        if debug:
            print(f"    Raw TIF: dtype={dtype}, shape={arr.shape}, min={arr.min():.1f}, max={arr.max():.1f}")

        max_val = arr.max()

        if dtype in ['float32', 'float64']:
            if max_val <= 1.0:
                pass
            elif max_val <= 10.0:
                pass
            else:
                arr = arr / 10000.0
        elif max_val > 255:
            arr = arr / 10000.0
        elif max_val > 1.0:
            arr = arr / 255.0

        arr = np.clip(arr, 0, 1)

        if debug:
            print(f"    After processing: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

        tensor = torch.from_numpy(arr)
        return tensor


def enhance_for_display(img: torch.Tensor, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Enhance image for display using percentile clipping."""
    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    p2, p98 = np.percentile(img_np, (p_low, p_high))
    img_np = np.clip(img_np, p2, p98)
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
    caption_info: str = None,
):
    """Save 3-column comparison image with optional caption info."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [img_t, img_tp1_gt, img_tp1_pred]

    for ax, img, title in zip(axes, images, titles):
        img_np = enhance_for_display(img)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    if caption_info:
        fig.suptitle(caption_info, fontsize=9, color='gray', y=0.02)

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
    """Save simple grid image (just concatenate)."""
    img_t_np = enhance_for_display(img_t)
    img_tp1_gt_np = enhance_for_display(img_tp1_gt)
    img_tp1_pred_np = enhance_for_display(img_tp1_pred)

    H, W, _ = img_t_np.shape
    pad = 4
    padding = np.ones((H, pad, 3))

    grid = np.concatenate([
        img_t_np, padding,
        img_tp1_gt_np, padding,
        img_tp1_pred_np,
    ], axis=1)

    grid_uint8 = (grid * 255).astype(np.uint8)
    Image.fromarray(grid_uint8).save(save_path)
    print(f"  Saved: {save_path}")


def load_captions_for_region(caption_dir: str, region_name: str, year_t: int) -> Optional[Dict]:
    """
    Load captions for a given region and year from caption directory.
    Looks for JSON files with caption text.

    Returns dict with keys: informative, geometric, semantic
    or None if not found.
    """
    if caption_dir is None:
        return None

    # Try several naming patterns
    patterns = [
        f"{region_name}_{year_t}.json",
        f"captions_{region_name}_{year_t}.json",
        f"{region_name}/{year_t}.json",
    ]

    for pattern in patterns:
        path = os.path.join(caption_dir, pattern)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            captions = {
                "informative": data.get("informative", data.get("informative_caption", "")),
                "geometric": data.get("geometric", data.get("geometric_caption", "")),
                "semantic": data.get("semantic", data.get("semantic_caption", "")),
            }
            if any(v for v in captions.values()):
                print(f"  Loaded captions from: {path}")
                return captions

    # Try precomputed embeddings
    emb_patterns = [
        f"caption_embeddings/{region_name}_{year_t}.pt",
        f"{region_name}/caption_embeddings/{year_t}.pt",
    ]
    for pattern in emb_patterns:
        path = os.path.join(caption_dir, pattern)
        if os.path.exists(path):
            print(f"  Loaded precomputed embeddings from: {path}")
            return {"_precomputed_path": path}

    return None


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
    caption_dir: str = None,
):
    """
    Load all model components from checkpoint.
    Auto-detects config from checkpoint.

    Returns:
        encoder, predictor, proj_head, sd_state, caption_forecaster,
        config, eval_dtype
    """

    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)

    # 1. Load checkpoint config
    print(f"[1/5] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})

    # Parse config (handle both flat and nested formats)
    if "mask" in config:
        # New nested format from train.py
        patch_size = config["mask"]["patch_size"]
        crop_size = config["data"]["crop_size"]
        pred_depth = config["meta"]["pred_depth"]
        pred_emb_dim = config["meta"]["pred_emb_dim"]
        model_name = config["meta"]["model_name"]
        use_lora = config["meta"].get("use_lora", True)
        lora_rank = int(config["meta"].get("lora_rank", 16))
        lora_alpha = int(config["meta"].get("lora_alpha", 32))
        target_emb_dim = config["meta"].get("target_emb_dim", 64)
        use_captions = config["meta"].get("use_captions", True)
        caption_forecaster_layers = int(config["meta"].get("caption_forecaster_layers", 3))
        caption_forecaster_hidden = int(config["meta"].get("caption_forecaster_hidden", 1024))
    else:
        # Old flat format (backward compat)
        patch_size = config.get("patch_size", 8)
        crop_size = config.get("crop_size", 128)
        pred_depth = config.get("pred_depth", 6)
        pred_emb_dim = config.get("pred_emb_dim", 384)
        model_name = config.get("model_name", "vit_base")
        use_lora = config.get("use_lora", True)
        lora_rank = int(config.get("lora_rank", 16))
        lora_alpha = int(config.get("lora_alpha", 32))
        target_emb_dim = config.get("target_emb_dim", 64)
        use_captions = config.get("use_captions", False)
        caption_forecaster_layers = 3
        caption_forecaster_hidden = 1024

        # Infer pred_depth from checkpoint keys if needed
        predictor_keys = [k for k in checkpoint.get("predictor", {}).keys()
                          if "predictor_blocks" in k]
        if predictor_keys:
            block_nums = [int(k.split(".")[1]) for k in predictor_keys
                          if k.split(".")[1].isdigit()]
            if block_nums:
                pred_depth = max(block_nums) + 1

        # Infer lora_rank from checkpoint
        lora_state = checkpoint.get("lora_state_dict", {})
        if lora_state:
            for key, tensor in lora_state.items():
                if "lora_A" in key:
                    lora_rank = tensor.shape[0]
                    lora_alpha = lora_rank * 2
                    break

    print(f"  Config: patch={patch_size}, crop={crop_size}, model={model_name}")
    print(f"  pred_depth={pred_depth}, pred_emb_dim={pred_emb_dim}")
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
    print(f"  Captions: {use_captions}")

    # 2. Initialize IJEPA
    print("[2/5] Initializing IJEPA...")
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )

    encoder_embed_dim = getattr(encoder, "embed_dim", 768)
    print(f"  Encoder embed_dim: {encoder_embed_dim}")

    # Projection head
    proj_head = torch.nn.Linear(encoder_embed_dim, target_emb_dim).to(device)

    # 3. Load SD model
    eval_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[3/5] Loading Stable Diffusion 3.5 (dtype={eval_dtype})...")

    # Check for precomputed embeddings
    _caption_dir = caption_dir or (config.get("meta", {}).get("caption_dir", None) if "meta" in config else None)
    _has_precomputed = _caption_dir and os.path.isdir(
        os.path.join(_caption_dir, "caption_embeddings")
    )
    _load_text_enc = use_captions and not _has_precomputed

    sd_state = load_sd_model(
        device=device,
        checkpoint_dir=config.get("meta", {}).get("sd_checkpoint_dir", "./sd_finetuned") if "meta" in config else "./sd_finetuned",
        dtype=eval_dtype,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_emb_dim=encoder_embed_dim,
        hf_token=config.get("meta", {}).get("hf_token", None) if "meta" in config else None,
        load_text_encoders=_load_text_enc,
    )

    sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    # 4. Caption Forecaster
    caption_forecaster = None
    if use_captions:
        print("[4/5] Initializing Caption Forecaster...")
        caption_forecaster = CaptionForecaster(
            ijepa_dim=encoder_embed_dim,
            text_dim=4096,
            hidden_dim=caption_forecaster_hidden,
            num_layers=caption_forecaster_layers,
        ).to(device)
    else:
        print("[4/5] Skipping Caption Forecaster (captions disabled)")

    # 5. Load weights
    print("[5/5] Loading model weights...")
    unet_dtype = next(sd_state["unet"].parameters()).dtype

    encoder.load_state_dict(checkpoint["encoder"])
    print("  ✓ Encoder loaded")

    predictor.load_state_dict(checkpoint["predictor"])
    print("  ✓ Predictor loaded")

    if "proj_head" in checkpoint:
        proj_head.load_state_dict(checkpoint["proj_head"])
        print("  ✓ Projection head loaded")

    try:
        sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"])
        print("  ✓ Multi-Caption Conditioning adapter loaded")
    except RuntimeError as e:
        print(f"  ⚠ Adapter mismatch (re-initialized): {e}")
    sd_state["cond_adapter"].to(dtype=unet_dtype)

    if checkpoint.get("use_lora") and "lora_state_dict" in checkpoint:
        sd_state["unet"].load_state_dict(checkpoint["lora_state_dict"], strict=False)
        print("  ✓ LoRA weights loaded")

    if "prompt_embeds" in checkpoint:
        sd_state["prompt_embeds"] = checkpoint["prompt_embeds"].to(device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = checkpoint["pooled_prompt_embeds"].to(device=device, dtype=unet_dtype)
        print("  ✓ Prompt embeddings loaded")

    if caption_forecaster is not None and "caption_forecaster" in checkpoint:
        caption_forecaster.load_state_dict(checkpoint["caption_forecaster"])
        print("  ✓ Caption Forecaster loaded")

    epoch = checkpoint.get("epoch", "unknown")
    print(f"\n  Checkpoint epoch: {epoch}")

    # Set eval mode
    encoder.eval()
    predictor.eval()
    proj_head.eval()
    sd_state["cond_adapter"].eval()
    sd_state["unet"].eval()
    sd_state["vae"].eval()
    if caption_forecaster is not None:
        caption_forecaster.eval()

    # Cast to eval_dtype for memory efficiency
    encoder.to(dtype=eval_dtype)
    predictor.to(dtype=eval_dtype)
    proj_head.to(dtype=eval_dtype)
    if caption_forecaster is not None:
        caption_forecaster.to(dtype=eval_dtype)

    torch.cuda.empty_cache()

    return encoder, predictor, proj_head, sd_state, caption_forecaster, config, eval_dtype


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_next_frame(
    rgb_t: torch.Tensor,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    proj_head: torch.nn.Module,
    sd_state: Dict,
    device: torch.device,
    patch_size: int = 8,
    num_diffusion_steps: int = 20,
    noise_strength: float = 0.35,
    # Caption conditioning
    caption_forecaster: torch.nn.Module = None,
    captions_t: Dict[str, List[str]] = None,
    cap_emb_t: Dict[str, torch.Tensor] = None,
    eval_dtype: torch.dtype = torch.bfloat16,
    debug: bool = True,
) -> torch.Tensor:
    """
    Predict t+1 image from t image with multi-caption conditioning.

    Args:
        rgb_t: Input image at time t, shape (3, H, W), range [0, 1]
        encoder: IJEPA encoder
        predictor: IJEPA predictor
        proj_head: Projection head (encoder_dim -> target_dim)
        sd_state: Stable Diffusion state dict
        device: torch device
        patch_size: ViT patch size
        num_diffusion_steps: Number of diffusion sampling steps
        noise_strength: img2img noise strength
        caption_forecaster: CaptionForecaster module (optional)
        captions_t: Dict with informative/geometric/semantic text lists (optional)
        cap_emb_t: Dict with precomputed caption embeddings (optional)
        eval_dtype: dtype for computation
        debug: Print debug info

    Returns:
        Predicted image at t+1, shape (3, H, W), range [0, 1]
    """
    rgb_t = rgb_t.unsqueeze(0).to(device)
    B, C, H, W = rgb_t.shape
    num_patches = (H // patch_size) ** 2

    if debug:
        print(f"    Input rgb_t: shape={rgb_t.shape}, min={rgb_t.min():.3f}, max={rgb_t.max():.3f}")

    enc_dtype = next(encoder.parameters()).dtype
    rgb_t_enc = rgb_t.to(dtype=enc_dtype)

    # Full mask (attend to all patches)
    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0)]

    # IJEPA forward
    z_enc = encoder(rgb_t_enc, full_mask)
    z_pred = predictor(z_enc, full_mask, full_mask)
    z_pred_norm = F.layer_norm(z_pred, (z_pred.size(-1),))

    if debug:
        print(f"    z_pred (after norm): shape={z_pred_norm.shape}, "
              f"min={z_pred_norm.min():.3f}, max={z_pred_norm.max():.3f}")

    # Get adapter dtype
    adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
    z_for_sd = z_pred_norm.to(dtype=adapter_dtype)
    unet_dtype = next(sd_state["unet"].parameters()).dtype

    # Caption conditioning
    info_h, info_p = None, None
    geo_h, geo_p = None, None
    sem_h, sem_p = None, None

    if cap_emb_t is not None:
        # Precomputed embeddings
        info_h = cap_emb_t["informative_hidden"].to(device=device, dtype=unet_dtype)
        info_p = cap_emb_t["informative_pooled"].to(device=device, dtype=unet_dtype)
        geo_h = cap_emb_t["geometric_hidden"].to(device=device, dtype=unet_dtype)
        geo_p = cap_emb_t["geometric_pooled"].to(device=device, dtype=unet_dtype)
        sem_t_h = cap_emb_t["semantic_hidden"].to(device=device, dtype=unet_dtype)
        sem_t_p = cap_emb_t["semantic_pooled"].to(device=device, dtype=unet_dtype)

        if caption_forecaster is not None:
            sem_h = caption_forecaster(z_pred_norm, sem_t_h)
            sem_p = sem_t_p
            if debug:
                print(f"    Caption forecast applied (precomputed embeddings)")
        else:
            sem_h, sem_p = sem_t_h, sem_t_p

    elif captions_t is not None and sd_state.get("caption_encoder") is not None:
        # Runtime text encoding
        cap_enc = sd_state["caption_encoder"]

        info_h, info_p = encode_caption_batch(
            cap_enc, captions_t["informative"], device, unet_dtype
        )
        geo_h, geo_p = encode_caption_batch(
            cap_enc, captions_t["geometric"], device, unet_dtype
        )
        sem_t_h, sem_t_p = encode_caption_batch(
            cap_enc, captions_t["semantic"], device, unet_dtype
        )

        if caption_forecaster is not None and sem_t_h is not None:
            sem_h = caption_forecaster(z_pred_norm, sem_t_h)
            sem_p = sem_t_p
            if debug:
                print(f"    Caption forecast applied (runtime encoding)")
        else:
            sem_h, sem_p = sem_t_h, sem_t_p

        if debug:
            captions_used = sum([info_h is not None, geo_h is not None, sem_h is not None])
            print(f"    Captions used: {captions_used}/3")

    else:
        if debug:
            print(f"    No captions — IJEPA-only conditioning")

    # Diffusion sampling with multi-caption conditioning
    gen_rgb = diffusion_sample(
        unet=sd_state["unet"],
        vae=sd_state["vae"],
        scheduler=sd_state["noise_scheduler"],
        cond_adapter=sd_state["cond_adapter"],
        ijepa_tokens=z_for_sd,
        num_steps=num_diffusion_steps,
        image_size=(H, W),
        device=device,
        noise_strength=noise_strength,
        informative_hidden=info_h,
        informative_pooled=info_p,
        geometric_hidden=geo_h,
        geometric_pooled=geo_p,
        semantic_hidden=sem_h,
        semantic_pooled=sem_p,
        ref_rgb=rgb_t.to(torch.float32),
    )

    if debug:
        print(f"    gen_rgb: shape={gen_rgb.shape}, min={gen_rgb.min():.3f}, max={gen_rgb.max():.3f}")

    return gen_rgb.squeeze(0)


@torch.no_grad()
def predict_multi_step(
    rgb_t: torch.Tensor,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    proj_head: torch.nn.Module,
    sd_state: Dict,
    device: torch.device,
    num_steps: int = 3,
    patch_size: int = 8,
    num_diffusion_steps: int = 20,
    noise_strength: float = 0.35,
    caption_forecaster: torch.nn.Module = None,
    captions_t: Dict[str, List[str]] = None,
    eval_dtype: torch.dtype = torch.bfloat16,
    debug: bool = True,
) -> List[torch.Tensor]:
    """
    Autoregressive multi-step prediction: t -> t+1 -> t+2 -> ...
    Each step feeds the predicted image back as input.

    Returns list of predicted frames [t+1, t+2, ..., t+num_steps].
    """
    predictions = []
    current_input = rgb_t

    for step in range(num_steps):
        if debug:
            print(f"\n  [Rollout Step {step+1}/{num_steps}]")

        pred = predict_next_frame(
            rgb_t=current_input,
            encoder=encoder,
            predictor=predictor,
            proj_head=proj_head,
            sd_state=sd_state,
            device=device,
            patch_size=patch_size,
            num_diffusion_steps=num_diffusion_steps,
            noise_strength=noise_strength,
            caption_forecaster=caption_forecaster,
            captions_t=captions_t,
            eval_dtype=eval_dtype,
            debug=debug,
        )

        predictions.append(pred.cpu())
        current_input = pred.cpu()

        # Clear captions after first step (we only have captions for t, not t+1)
        captions_t = None

    return predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sat-JEPA-Diff Inference (Multi-Caption)")

    # Paths
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", "-o", type=str, default="./inference_results",
                        help="Output directory for results")
    parser.add_argument("--caption_dir", type=str, default=None,
                        help="Directory with caption files (optional)")

    # Inference config
    parser.add_argument("--diffusion_steps", type=int, default=20,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--noise_strength", type=float, default=0.35,
                        help="Img2img noise strength")
    parser.add_argument("--multi_step", type=int, default=1,
                        help="Number of autoregressive rollout steps (1 = single step)")

    # Region selection
    parser.add_argument("--regions", type=str, default=None,
                        help="Comma-separated region indices (e.g. '0,2,4') or 'all'")
    parser.add_argument("--input_tif", type=str, default=None,
                        help="Path to a custom input TIF (overrides region list)")
    parser.add_argument("--gt_tif", type=str, default=None,
                        help="Path to ground truth TIF for custom input")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    (encoder, predictor, proj_head, sd_state,
     caption_forecaster, config, eval_dtype) = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        caption_dir=args.caption_dir,
    )

    # Extract config values
    if "mask" in config:
        patch_size = config["mask"]["patch_size"]
        crop_size = config["data"]["crop_size"]
        use_captions = config["meta"].get("use_captions", True)
    else:
        patch_size = config.get("patch_size", 8)
        crop_size = config.get("crop_size", 128)
        use_captions = config.get("use_captions", False)

    caption_dir = args.caption_dir

    # Build region list
    if args.input_tif:
        region_paths = [args.input_tif]
    else:
        region_paths = REGION_PATHS
        if args.regions and args.regions != "all":
            indices = [int(x.strip()) for x in args.regions.split(",")]
            region_paths = [region_paths[i] for i in indices]

    # Process each region
    print("\n" + "=" * 60)
    print("Running Inference...")
    print("=" * 60)

    results = []

    for region_path in region_paths:
        if args.input_tif:
            info = {"region": Path(region_path).stem, "year": 0}
        else:
            info = extract_info_from_path(region_path)

        region_name = info["region"]
        year_t = info["year"]

        print(f"\n[{region_name}] Year {year_t} -> {year_t + 1}")

        # Check paths
        if not os.path.exists(region_path):
            print(f"  ⚠ Skipping: Input file not found: {region_path}")
            continue

        if args.input_tif:
            next_year_path = args.gt_tif
        else:
            next_year_path = get_next_year_path(region_path)

        has_gt = next_year_path is not None and os.path.exists(next_year_path)
        if not has_gt:
            print(f"  ⚠ No ground truth found — will generate prediction only")

        # Load images
        print(f"  Loading images...")
        try:
            rgb_t = load_and_resize_tif(region_path, crop_size)
        except Exception as e:
            print(f"  ⚠ Error loading input: {e}")
            continue

        rgb_tp1_gt = None
        if has_gt:
            try:
                rgb_tp1_gt = load_and_resize_tif(next_year_path, crop_size)
            except Exception as e:
                print(f"  ⚠ Error loading GT: {e}")
                has_gt = False

        # Load captions for this region
        captions_t = None
        cap_emb_t = None
        caption_info_str = "No captions"

        if use_captions and caption_dir:
            cap_data = load_captions_for_region(caption_dir, region_name, year_t)
            if cap_data is not None:
                if "_precomputed_path" in cap_data:
                    # Load precomputed embeddings
                    cap_emb_t = torch.load(cap_data["_precomputed_path"], map_location=device)
                    # Add batch dimension if missing
                    for k, v in cap_emb_t.items():
                        if v.dim() == 2:
                            cap_emb_t[k] = v.unsqueeze(0)
                    caption_info_str = "Precomputed embeddings"
                else:
                    captions_t = {k: [v] for k, v in cap_data.items()}  # Wrap as batch of 1
                    caption_info_str = f"Info: {cap_data['informative'][:60]}..."
            else:
                print(f"  No captions found for {region_name}_{year_t}")

        # Run prediction
        if args.multi_step > 1:
            print(f"  Predicting {args.multi_step} steps (autoregressive)...")
            predictions = predict_multi_step(
                rgb_t=rgb_t,
                encoder=encoder,
                predictor=predictor,
                proj_head=proj_head,
                sd_state=sd_state,
                device=device,
                num_steps=args.multi_step,
                patch_size=patch_size,
                num_diffusion_steps=args.diffusion_steps,
                noise_strength=args.noise_strength,
                caption_forecaster=caption_forecaster,
                captions_t=captions_t,
                eval_dtype=eval_dtype,
            )

            # Save each step
            for step_idx, pred in enumerate(predictions):
                step_year = year_t + step_idx + 1
                save_name = f"{region_name}_{year_t}_to_{step_year}_step{step_idx+1}.png"
                save_path = os.path.join(args.output_dir, save_name)

                gt_for_step = rgb_tp1_gt if step_idx == 0 and has_gt else torch.zeros_like(pred)
                save_comparison_image(
                    img_t=rgb_t, img_tp1_gt=gt_for_step, img_tp1_pred=pred,
                    save_path=save_path,
                    titles=(f"Input ({year_t})", f"GT ({step_year})", f"Pred ({step_year})"),
                    caption_info=caption_info_str if step_idx == 0 else None,
                )

            rgb_tp1_pred = predictions[0]  # For metrics, use first step

        else:
            print(f"  Predicting t+1...")
            rgb_tp1_pred = predict_next_frame(
                rgb_t=rgb_t,
                encoder=encoder,
                predictor=predictor,
                proj_head=proj_head,
                sd_state=sd_state,
                device=device,
                patch_size=patch_size,
                num_diffusion_steps=args.diffusion_steps,
                noise_strength=args.noise_strength,
                caption_forecaster=caption_forecaster,
                captions_t=captions_t,
                cap_emb_t=cap_emb_t,
                eval_dtype=eval_dtype,
            ).cpu()

        # Compute metrics
        result = {"region": region_name, "year_t": year_t}

        if has_gt:
            l1 = F.l1_loss(rgb_tp1_pred, rgb_tp1_gt).item()
            mse = F.mse_loss(rgb_tp1_pred, rgb_tp1_gt).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            print(f"  Metrics: L1={l1:.4f}, MSE={mse:.4f}, PSNR={psnr:.2f}")
            result.update({"l1": l1, "mse": mse, "psnr": psnr})
        else:
            print(f"  No GT — metrics not computed")

        results.append(result)

        # Save comparison image
        if args.multi_step <= 1:
            save_name = f"{region_name}_{year_t}_to_{year_t+1}.png"
            save_path = os.path.join(args.output_dir, save_name)

            gt_img = rgb_tp1_gt if has_gt else torch.zeros_like(rgb_tp1_pred)
            gt_title = f"Ground Truth ({year_t + 1})" if has_gt else f"No GT ({year_t + 1})"

            save_comparison_image(
                img_t=rgb_t, img_tp1_gt=gt_img, img_tp1_pred=rgb_tp1_pred,
                save_path=save_path,
                titles=(f"Input ({year_t})", gt_title, f"Predicted ({year_t + 1})"),
                caption_info=caption_info_str,
            )

            grid_path = os.path.join(args.output_dir, f"{region_name}_{year_t}_grid.png")
            save_grid_image(rgb_t, gt_img, rgb_tp1_pred, grid_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results_with_metrics = [r for r in results if "l1" in r]

    if results_with_metrics:
        avg_l1 = np.mean([r["l1"] for r in results_with_metrics])
        avg_mse = np.mean([r["mse"] for r in results_with_metrics])
        avg_psnr = np.mean([r["psnr"] for r in results_with_metrics])

        print(f"\nAverage Metrics ({len(results_with_metrics)} regions):")
        print(f"  L1:   {avg_l1:.4f}")
        print(f"  MSE:  {avg_mse:.4f}")
        print(f"  PSNR: {avg_psnr:.2f}")

        print("\nPer-Region Results:")
        for r in results_with_metrics:
            print(f"  {r['region']}: L1={r['l1']:.4f}, PSNR={r['psnr']:.2f}")

    # Save results JSON
    results_json = os.path.join(args.output_dir, "inference_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Metrics JSON: {results_json}")


if __name__ == "__main__":
    main()