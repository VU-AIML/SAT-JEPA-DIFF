"""
Autoregressive Rollout - Parameter Sweep Version (Multi-Caption Architecture)

Generates a grid visualization:
  Row 0: Ground Truth for all years
  Row 1..N: Rollout predictions for each (noise, latent_anchor) combo

Each cell shows the image + PSNR vs GT.

Architecture:
- IJEPA encoder → visual tokens at t+1
- CaptionForecaster → semantic caption at t+1 (optional)
- SD3.5 conditioned on: informative(t) + geometric(t) + semantic(t+1) + IJEPA tokens
- No coarse RGB, no color matching — pure embedding + caption conditioning

Usage:
  python autoregressive_rollout.py --checkpoint model.pth
  python autoregressive_rollout.py --checkpoint model.pth --steps 30 --output sweep.png
  python autoregressive_rollout.py --checkpoint model.pth --caption_dir ./captions
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from helper import init_model
from sd_models import load_sd_model, encode_caption_batch
from caption_forecaster import CaptionForecaster

REGION_DIR = "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Rio_de_Janeiro_Coast_Brazil"
FILENAME_TEMPLATE = "s2_rgb_Rio_de_Janeiro_Coast_Brazil_{year}_10km.tif"
YEARS = range(2018, 2025)
IMG_SIZE = 128

# ============================================================================
# Parameter Sweep Grid
# ============================================================================
SWEEP_CONFIGS = [
    {"noise": 0.35, "latent_anchor": 0.00},
    {"noise": 0.35, "latent_anchor": 0.10},
    {"noise": 0.30, "latent_anchor": 0.00},
    {"noise": 0.30, "latent_anchor": 0.10},
    {"noise": 0.25, "latent_anchor": 0.00},
    {"noise": 0.25, "latent_anchor": 0.10},
    {"noise": 0.20, "latent_anchor": 0.00},
    {"noise": 0.20, "latent_anchor": 0.10},
]


def normalize_for_display(img_tensor):
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    p2, p98 = np.percentile(img_np, (2, 98))
    if p98 > p2:
        img_np = (img_np - p2) / (p98 - p2)
    return np.clip(img_np, 0, 1)


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1.float(), img2.float()).item()
    if mse < 1e-10:
        return 100.0
    return -10 * np.log10(mse)


def load_and_preprocess(path, size=IMG_SIZE):
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            img_np = src.read(out_shape=(src.count, size, size), resampling=Resampling.bilinear)
            if img_np.shape[0] > 3:
                img_np = img_np[:3, :, :]
            img_np = img_np.astype(np.float32)
            max_val = img_np.max()
            if max_val > 1.0:
                img_np = img_np / (10000.0 if max_val > 255.0 else 255.0)
            return torch.from_numpy(np.clip(img_np, 0, 1))
    except Exception as e:
        print(f"[Error] {path}: {e}")
        return None


def load_captions_for_year(caption_dir, region_name, year):
    """Load captions or precomputed embeddings for a given year."""
    if caption_dir is None:
        return None, None

    # Try precomputed embeddings
    emb_path = os.path.join(caption_dir, "caption_embeddings", f"{region_name}_{year}.pt")
    if os.path.exists(emb_path):
        return torch.load(emb_path, map_location="cpu"), None

    # Try JSON captions
    for pattern in [f"{region_name}_{year}.json", f"captions_{region_name}_{year}.json"]:
        path = os.path.join(caption_dir, pattern)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            captions = {
                "informative": [data.get("informative", data.get("informative_caption", ""))],
                "geometric": [data.get("geometric", data.get("geometric_caption", ""))],
                "semantic": [data.get("semantic", data.get("semantic_caption", ""))],
            }
            if any(v[0] for v in captions.values()):
                return None, captions

    return None, None


# ============================================================================
# Model Loading
# ============================================================================

def load_full_model(checkpoint_path, device, caption_dir=None):
    """Load model from checkpoint — auto-detects config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Parse config (nested vs flat)
    if "mask" in config:
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
        patch_size = config.get("patch_size", 8)
        crop_size = config.get("crop_size", 128)
        pred_depth = config.get("pred_depth", 6)
        pred_emb_dim = config.get("pred_emb_dim", 384)
        model_name = config.get("model_name", "vit_base")
        use_lora = config.get("use_lora", True)
        lora_rank = int(config.get("lora_rank", 8))
        lora_alpha = int(config.get("lora_alpha", 16))
        target_emb_dim = config.get("target_emb_dim", 64)
        use_captions = False
        caption_forecaster_layers = 3
        caption_forecaster_hidden = 1024

    global IMG_SIZE
    IMG_SIZE = crop_size

    # IJEPA
    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    encoder_embed_dim = getattr(encoder, "embed_dim", 768)

    # Projection head
    proj_head = torch.nn.Linear(encoder_embed_dim, target_emb_dim).to(device)

    # SD model
    eval_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    _caption_dir = caption_dir or (config.get("meta", {}).get("caption_dir", None) if "meta" in config else None)
    _has_precomputed = _caption_dir and os.path.isdir(
        os.path.join(_caption_dir, "caption_embeddings")
    )
    _load_text_enc = use_captions and not _has_precomputed

    sd_state = load_sd_model(
        device=device, dtype=eval_dtype, use_lora=use_lora,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        target_emb_dim=encoder_embed_dim,
        hf_token=config.get("meta", {}).get("hf_token", None) if "meta" in config else None,
        load_text_encoders=_load_text_enc,
    )
    sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    # Caption Forecaster
    caption_forecaster = None
    if use_captions:
        caption_forecaster = CaptionForecaster(
            ijepa_dim=encoder_embed_dim,
            text_dim=4096,
            hidden_dim=caption_forecaster_hidden,
            num_layers=caption_forecaster_layers,
        ).to(device)

    # Load weights
    unet_dtype = next(sd_state["unet"].parameters()).dtype

    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])

    if "proj_head" in checkpoint:
        proj_head.load_state_dict(checkpoint["proj_head"])

    try:
        sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"], strict=True)
    except RuntimeError:
        sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"], strict=False)
    sd_state["cond_adapter"].to(dtype=unet_dtype)

    if checkpoint.get("use_lora") and "lora_state_dict" in checkpoint:
        sd_state["unet"].load_state_dict(checkpoint["lora_state_dict"], strict=False)

    if "prompt_embeds" in checkpoint:
        sd_state["prompt_embeds"] = checkpoint["prompt_embeds"].to(device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = checkpoint["pooled_prompt_embeds"].to(device=device, dtype=unet_dtype)

    if caption_forecaster is not None and "caption_forecaster" in checkpoint:
        caption_forecaster.load_state_dict(checkpoint["caption_forecaster"])

    # Eval mode + dtype
    for m in [encoder, predictor, proj_head, sd_state["cond_adapter"],
              sd_state["unet"], sd_state["vae"]]:
        m.eval()
    if caption_forecaster is not None:
        caption_forecaster.eval()
        caption_forecaster.to(dtype=eval_dtype)
    encoder.to(dtype=eval_dtype)
    predictor.to(dtype=eval_dtype)
    proj_head.to(dtype=eval_dtype)

    torch.cuda.empty_cache()

    return (encoder, predictor, proj_head, sd_state,
            caption_forecaster, config, use_captions, eval_dtype)


# ============================================================================
# Single-Step Prediction
# ============================================================================

def predict_one_step(
    encoder, predictor, proj_head, sd_state, input_img, anchor_img,
    device, steps, noise, latent_anchor_weight=0.0,
    caption_forecaster=None,
    cap_emb_t=None,
    captions_t=None,
    eval_dtype=torch.bfloat16,
):
    """
    Generate next frame with multi-step Euler denoising + latent anchor.
    Uses multi-caption conditioning (no coarse RGB).
    """
    patch_size = 8
    num_patches = (IMG_SIZE // patch_size) ** 2

    vae = sd_state["vae"]
    unet = sd_state["unet"]
    cond_adapter = sd_state["cond_adapter"]

    vae_dtype = next(vae.parameters()).dtype
    unet_dtype = next(unet.parameters()).dtype

    scale = getattr(vae.config, "scaling_factor", 1.5305)
    shift_val = getattr(vae.config, "shift_factor", 0.0609)

    rgb_t = input_img.unsqueeze(0).to(device)
    enc_dtype = next(encoder.parameters()).dtype
    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0)]

    with torch.no_grad():
        # --- IJEPA ---
        z_enc = encoder(rgb_t.to(dtype=enc_dtype), full_mask)
        z_pred = predictor(z_enc, full_mask, full_mask)
        z_pred_norm = F.layer_norm(z_pred, (z_pred.size(-1),))

        adapter_dtype = next(cond_adapter.parameters()).dtype
        z_for_adapter = z_pred_norm.to(dtype=adapter_dtype)

        # --- Caption Conditioning ---
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

            # Add batch dim if missing
            if info_h.dim() == 2:
                info_h = info_h.unsqueeze(0)
                info_p = info_p.unsqueeze(0)
                geo_h = geo_h.unsqueeze(0)
                geo_p = geo_p.unsqueeze(0)
                sem_t_h = sem_t_h.unsqueeze(0)
                sem_t_p = sem_t_p.unsqueeze(0)

            if caption_forecaster is not None:
                sem_h = caption_forecaster(z_pred_norm, sem_t_h)
                sem_p = sem_t_p
            else:
                sem_h, sem_p = sem_t_h, sem_t_p

        elif captions_t is not None and sd_state.get("caption_encoder") is not None:
            # Runtime text encoding
            cap_enc = sd_state["caption_encoder"]
            info_h, info_p = encode_caption_batch(cap_enc, captions_t["informative"], device, unet_dtype)
            geo_h, geo_p = encode_caption_batch(cap_enc, captions_t["geometric"], device, unet_dtype)
            sem_t_h, sem_t_p = encode_caption_batch(cap_enc, captions_t["semantic"], device, unet_dtype)

            if caption_forecaster is not None and sem_t_h is not None:
                sem_h = caption_forecaster(z_pred_norm, sem_t_h)
                sem_p = sem_t_p
            else:
                sem_h, sem_p = sem_t_h, sem_t_p

        # --- Adapter forward (multi-caption, no coarse RGB) ---
        enc_hidden, pooled_proj = cond_adapter(
            z_for_adapter,
            informative_hidden=info_h,
            informative_pooled=info_p,
            geometric_hidden=geo_h,
            geometric_pooled=geo_p,
            semantic_hidden=sem_h,
            semantic_pooled=sem_p,
        )
        enc_hidden = enc_hidden.to(dtype=unet_dtype)
        pooled_proj = pooled_proj.to(dtype=unet_dtype)

        # --- Encode input -> latent ---
        ref_norm = rgb_t.float() * 2 - 1
        ref_latent = vae.encode(ref_norm.to(vae_dtype)).latent_dist.sample()
        ref_latent = (ref_latent - shift_val) * scale
        clean_latent = ref_latent.to(unet_dtype)

        # --- Add noise ---
        rand_noise = torch.randn_like(clean_latent)
        latents = (1.0 - noise) * clean_latent + noise * rand_noise

        # --- Multi-step Euler denoise ---
        ts = torch.linspace(noise, 0.0, steps + 1, device=device)

        for i in range(steps):
            dt = ts[i + 1] - ts[i]
            t_batch = torch.full((1,), ts[i].item() * 1000, device=device, dtype=unet_dtype)
            out = unet(
                hidden_states=latents, timestep=t_batch,
                encoder_hidden_states=enc_hidden,
                pooled_projections=pooled_proj,
                return_dict=False,
            )
            vel = out[0] if isinstance(out, tuple) else out
            latents = latents + dt * vel

        # --- Latent-space anchoring ---
        if latent_anchor_weight > 0 and anchor_img is not None:
            a_rgb = anchor_img.unsqueeze(0).to(device)
            a_norm = a_rgb.float() * 2 - 1
            a_latent = vae.encode(a_norm.to(vae_dtype)).latent_dist.sample()
            a_latent = (a_latent - shift_val) * scale
            a_latent = a_latent.to(latents.dtype)
            latents = (1.0 - latent_anchor_weight) * latents + latent_anchor_weight * a_latent

        # --- Decode ---
        latents = latents.to(dtype=vae_dtype)
        latents_unscaled = (latents / scale) + shift_val
        image = vae.decode(latents_unscaled, return_dict=False)[0]
        result = ((image.float() + 1) / 2).clamp(0, 1)

    return result.squeeze(0).cpu()


# ============================================================================
# Rollout
# ============================================================================

def run_single_rollout(
    encoder, predictor, proj_head, sd_state, gt_images, device,
    steps, noise, latent_anchor,
    caption_forecaster=None,
    caption_dir=None,
    region_name=None,
    use_captions=False,
    eval_dtype=torch.bfloat16,
):
    """Run one rollout with given parameters. Returns predictions + PSNR list."""
    predictions = [gt_images[0]]
    psnr_values = [None]
    current = gt_images[0]
    anchor = gt_images[0]

    years_list = list(YEARS)

    for i in range(len(years_list) - 1):
        year_t = years_list[i]

        # Load captions for current year
        cap_emb_t = None
        captions_t = None
        if use_captions and caption_dir and region_name:
            cap_emb_t, captions_t = load_captions_for_year(caption_dir, region_name, year_t)

        pred = predict_one_step(
            encoder, predictor, proj_head, sd_state,
            input_img=current, anchor_img=anchor,
            device=device, steps=steps, noise=noise,
            latent_anchor_weight=latent_anchor,
            caption_forecaster=caption_forecaster,
            cap_emb_t=cap_emb_t,
            captions_t=captions_t,
            eval_dtype=eval_dtype,
        )

        predictions.append(pred)
        current = pred

        # PSNR vs GT
        psnr = None
        if gt_images[i + 1] is not None:
            psnr = compute_psnr(pred, gt_images[i + 1])
        psnr_values.append(psnr)

    return predictions, psnr_values


# ============================================================================
# Parameter Sweep
# ============================================================================

def run_parameter_sweep(checkpoint_path, output_path, device_name="cuda",
                        steps=30, sweep_configs=None,
                        caption_dir=None, region_dir=None,
                        region_name=None):
    """Run parameter sweep and generate grid visualization."""
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    if sweep_configs is None:
        sweep_configs = SWEEP_CONFIGS

    # Load model once
    (encoder, predictor, proj_head, sd_state,
     caption_forecaster, config, use_captions, eval_dtype) = load_full_model(
        checkpoint_path, device, caption_dir=caption_dir
    )

    # Region dir
    _region_dir = region_dir or REGION_DIR
    _region_name = region_name
    if _region_name is None:
        _region_name = os.path.basename(_region_dir)

    # Load GT images
    gt_images = []
    for year in YEARS:
        path = os.path.join(_region_dir, FILENAME_TEMPLATE.format(year=year))
        gt_images.append(load_and_preprocess(path))

    if gt_images[0] is None:
        print("Error: First year image missing!")
        return

    num_years = len(list(YEARS))
    num_configs = len(sweep_configs)
    num_rows = 1 + num_configs

    # Caption info
    caption_status = "with captions" if (use_captions and caption_dir) else "no captions"
    print(f"Conditioning: {caption_status}")
    if caption_forecaster is not None:
        print("Caption Forecaster: enabled")

    # Run all rollouts
    all_results = []
    for cfg_idx, cfg in enumerate(sweep_configs):
        n = cfg["noise"]
        la = cfg["latent_anchor"]
        label = f"noise={n:.2f}, anchor={la:.2f}"
        print(f"\n[{cfg_idx+1}/{num_configs}] {label}")

        preds, psnrs = run_single_rollout(
            encoder, predictor, proj_head, sd_state, gt_images, device,
            steps=steps, noise=n, latent_anchor=la,
            caption_forecaster=caption_forecaster,
            caption_dir=caption_dir,
            region_name=_region_name,
            use_captions=use_captions,
            eval_dtype=eval_dtype,
        )

        valid_psnrs = [p for p in psnrs if p is not None]
        if valid_psnrs:
            psnr_str = ", ".join(f"{p:.1f}" for p in valid_psnrs)
            print(f"  PSNR: [{psnr_str}]  avg={np.mean(valid_psnrs):.1f}dB")

        all_results.append({
            "config": cfg, "label": label,
            "preds": preds, "psnrs": psnrs,
        })

    # --- Plot Grid ---
    fig, axes = plt.subplots(num_rows, num_years, figsize=(num_years * 2.5, num_rows * 2.5))

    # Row 0: Ground Truth
    for col, year in enumerate(YEARS):
        ax = axes[0, col]
        if gt_images[col] is not None:
            ax.imshow(normalize_for_display(gt_images[col]))
        ax.set_title(f"{year}", fontsize=9)
        ax.axis('off')

    # Rows 1..N: Sweep results
    for row_idx, result in enumerate(all_results):
        cfg = result["config"]
        preds = result["preds"]
        psnrs = result["psnrs"]
        row = row_idx + 1

        for col in range(num_years):
            ax = axes[row, col]
            ax.imshow(normalize_for_display(preds[col]))
            ax.axis('off')

            if col > 0 and psnrs[col] is not None:
                ax.set_title(f"{psnrs[col]:.1f}dB", fontsize=8, color='red')

        axes[row, 0].set_ylabel(
            f"n={cfg['noise']:.2f}\na={cfg['latent_anchor']:.2f}",
            fontsize=7, rotation=0, labelpad=50, va='center',
        )

    plt.suptitle(
        f"Parameter Sweep (steps={steps}, {caption_status})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved: {output_path}")

    # Print summary table
    print(f"\n{'Config':<30} {'Avg PSNR':>10} {'Final PSNR':>12}")
    print("-" * 55)
    for result in all_results:
        valid = [p for p in result["psnrs"] if p is not None]
        avg = np.mean(valid) if valid else 0
        final = valid[-1] if valid else 0
        print(f"{result['label']:<30} {avg:>8.1f}dB {final:>10.1f}dB")

    # Save results JSON
    json_results = []
    for result in all_results:
        json_results.append({
            "config": result["config"],
            "psnrs": [p if p is not None else None for p in result["psnrs"]],
            "avg_psnr": float(np.mean([p for p in result["psnrs"] if p is not None])) if any(p is not None for p in result["psnrs"]) else 0,
        })
    json_path = output_path.rsplit(".", 1)[0] + "_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autoregressive Rollout — Multi-Caption Architecture")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="parameter_sweep.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--caption_dir", type=str, default=None,
                   help="Directory with caption files / precomputed embeddings")
    p.add_argument("--region_dir", type=str, default=None,
                   help="Override region directory")
    p.add_argument("--region_name", type=str, default=None,
                   help="Region name for caption lookup")
    args = p.parse_args()

    run_parameter_sweep(
        args.checkpoint, args.output,
        steps=args.steps,
        caption_dir=args.caption_dir,
        region_dir=args.region_dir,
        region_name=args.region_name,
    )