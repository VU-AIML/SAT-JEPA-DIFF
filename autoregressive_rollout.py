"""
Autoregressive Rollout - Parameter Sweep Version

Generates a grid visualization:
  Row 0: Ground Truth for all years
  Row 1..N: Rollout predictions for each (noise, latent_anchor) combo

Each cell shows the image + PSNR vs GT.
Title: "Parameter Sweep (steps=X, color_fix=ON/OFF)"

Usage:
  python autoregressive_rollout.py --checkpoint model.pth
  python autoregressive_rollout.py --checkpoint model.pth --steps 30 --output sweep.png
"""

import os
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
from sd_models import load_sd_model
from sd_joint_loss import match_color_statistics, downsample_to_coarse
COARSE_SIZE = 32

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


def load_full_model(checkpoint_path, device, patch_size=8, crop_size=128):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=config.get("pred_depth", 6),
        pred_emb_dim=config.get("pred_emb_dim", 384),
        model_name=config.get("model_name", "vit_base"),
    )
    encoder_embed_dim = getattr(encoder, "embed_dim", 768)

    sd_state = load_sd_model(
        device=device, dtype=torch.float16, use_lora=True,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_emb_dim=encoder_embed_dim, load_text_encoders=False,
    )
    sd_state["vae"] = sd_state["vae"].to(dtype=torch.float32)

    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    try:
        sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"], strict=True)
    except RuntimeError:
        sd_state["cond_adapter"].load_state_dict(checkpoint["cond_adapter"], strict=False)
    if checkpoint.get("use_lora") and "lora_state_dict" in checkpoint:
        sd_state["unet"].load_state_dict(checkpoint["lora_state_dict"], strict=False)

    for m in [encoder, predictor, sd_state["cond_adapter"], sd_state["unet"], sd_state["vae"]]:
        m.eval()
    return encoder, predictor, sd_state


def predict_one_step(
    encoder, predictor, sd_state, input_img, anchor_img,
    device, steps, noise, latent_anchor_weight=0.0,
):
    """
    Generate next frame with multi-step Euler denoising + latent anchor.
    """
    patch_size = 8
    num_patches = (IMG_SIZE // patch_size) ** 2

    vae = sd_state["vae"]
    unet = sd_state["unet"]

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
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

        adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
        z_pred = z_pred.to(dtype=adapter_dtype)

        # --- Conditioning ---
        coarse_rgb = downsample_to_coarse(rgb_t.float(), COARSE_SIZE)
        try:
            h, p = sd_state["cond_adapter"](z_pred, ref_rgb=coarse_rgb)
        except TypeError:
            h, p = sd_state["cond_adapter"](z_pred, ref_rgb=coarse_rgb, history_hidden=None)
        h = h.to(dtype=unet_dtype)
        p = p.to(dtype=unet_dtype)

        text_embeds = sd_state.get("prompt_embeds")
        pooled_text = sd_state.get("pooled_prompt_embeds")

        if text_embeds is not None:
            th = text_embeds.expand(1, -1, -1).to(device=device, dtype=unet_dtype)
            tp = pooled_text.expand(1, -1).to(device=device, dtype=unet_dtype)
            enc_hidden = torch.cat([th, h], dim=1)
            pooled_proj = (tp + p) / 2
        else:
            enc_hidden = h
            pooled_proj = p

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


def run_single_rollout(encoder, predictor, sd_state, gt_images, device,
                       steps, noise, latent_anchor, color_fix):
    """Run one rollout with given parameters. Returns list of predictions + PSNR list."""
    predictions = [gt_images[0]]
    psnr_values = [None]  # No PSNR for first year (it's the input)
    current = gt_images[0]
    anchor = gt_images[0]

    for i in range(len(YEARS) - 1):
        pred = predict_one_step(
            encoder, predictor, sd_state,
            input_img=current, anchor_img=anchor,
            device=device, steps=steps, noise=noise,
            latent_anchor_weight=latent_anchor,
        )

        if color_fix:
            pred = match_color_statistics(
                pred.unsqueeze(0), anchor.unsqueeze(0)
            ).squeeze(0)

        predictions.append(pred)
        current = pred

        # Compute PSNR vs GT
        psnr = None
        if gt_images[i + 1] is not None:
            psnr = compute_psnr(pred, gt_images[i + 1])
        psnr_values.append(psnr)

    return predictions, psnr_values


def run_parameter_sweep(checkpoint_path, output_path, device_name="cuda",
                        steps=30, color_fix=True, sweep_configs=None):
    """Run parameter sweep and generate grid visualization."""
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    if sweep_configs is None:
        sweep_configs = SWEEP_CONFIGS

    # Load model once
    encoder, predictor, sd_state = load_full_model(checkpoint_path, device)

    # Load GT images
    gt_images = []
    for year in YEARS:
        path = os.path.join(REGION_DIR, FILENAME_TEMPLATE.format(year=year))
        gt_images.append(load_and_preprocess(path))

    if gt_images[0] is None:
        print("Error: First year image missing!")
        return

    num_years = len(list(YEARS))
    num_configs = len(sweep_configs)
    num_rows = 1 + num_configs  # GT row + one row per config

    # Run all rollouts
    all_results = []
    for cfg_idx, cfg in enumerate(sweep_configs):
        n = cfg["noise"]
        la = cfg["latent_anchor"]
        label = f"noise={n:.2f}, anchor={la:.2f}"
        print(f"\n[{cfg_idx+1}/{num_configs}] {label}")

        preds, psnrs = run_single_rollout(
            encoder, predictor, sd_state, gt_images, device,
            steps=steps, noise=n, latent_anchor=la, color_fix=color_fix,
        )

        # Print PSNR summary
        valid_psnrs = [p for p in psnrs if p is not None]
        if valid_psnrs:
            psnr_str = ", ".join(f"{p:.1f}" for p in valid_psnrs)
            print(f"  PSNR: [{psnr_str}]  avg={np.mean(valid_psnrs):.1f}dB")

        all_results.append({"config": cfg, "label": label, "preds": preds, "psnrs": psnrs})

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

            # Show PSNR below each cell (except first column = input)
            if col > 0 and psnrs[col] is not None:
                ax.set_title(f"{psnrs[col]:.1f}dB", fontsize=8, color='red')

        # Row label on the left (first column)
        axes[row, 0].set_ylabel(
            f"n={cfg['noise']:.2f}\na={cfg['latent_anchor']:.2f}",
            fontsize=7, rotation=0, labelpad=50, va='center',
        )

    color_str = "ON" if color_fix else "OFF"
    plt.suptitle(
        f"Parameter Sweep (steps={steps}, color_fix={color_str})",
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="brazil_parameter_sweep_only_sd.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--no_color_fix", action="store_true")
    args = p.parse_args()

    run_parameter_sweep(
        args.checkpoint, args.output,
        steps=args.steps,
        color_fix=not args.no_color_fix,
    )