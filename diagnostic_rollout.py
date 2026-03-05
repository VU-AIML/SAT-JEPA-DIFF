"""
Parameter Sweep: Find optimal noise + latent_anchor combination

Runs autoregressive rollout with multiple parameter combinations and outputs:
1. A comparison grid (one row per config)
2. PSNR table for each config x year
3. Recommendation for best config

This saves you from manually testing --noise 0.15 --latent_anchor 0.15, etc.
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
from sd_joint_loss import match_color_statistics, downsample_to_coarse, COARSE_SIZE

REGION_DIR = "/media/kursat/TOSHIBA_EXT/projects/satellite/ijepa/downloads/Cairo_Nile_Corridor_Egypt"
FILENAME_TEMPLATE = "s2_rgb_Cairo_Nile_Corridor_Egypt_{year}_10km.tif"
YEARS = list(range(2017, 2025))
IMG_SIZE = 128


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
    except:
        return None


def load_full_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    encoder, predictor = init_model(
        device=device, patch_size=8, crop_size=IMG_SIZE,
        pred_depth=config.get("pred_depth", 6),
        pred_emb_dim=config.get("pred_emb_dim", 384),
        model_name=config.get("model_name", "vit_base"),
    )
    embed_dim = getattr(encoder, "embed_dim", 768)
    sd_state = load_sd_model(
        device=device, dtype=torch.float16, use_lora=True,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_emb_dim=embed_dim, load_text_encoders=False,
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


def predict_one(encoder, predictor, sd_state, input_img, anchor_img,
                device, steps, noise, latent_anchor):
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
        z_enc = encoder(rgb_t.to(dtype=enc_dtype), full_mask)
        z_pred = predictor(z_enc, full_mask, full_mask)
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
        adapter_dtype = next(sd_state["cond_adapter"].parameters()).dtype
        z_pred = z_pred.to(dtype=adapter_dtype)

        coarse_rgb = downsample_to_coarse(rgb_t.float(), COARSE_SIZE)
        try:
            h, p = sd_state["cond_adapter"](z_pred, ref_rgb=coarse_rgb)
        except TypeError:
            h, p = sd_state["cond_adapter"](z_pred, ref_rgb=coarse_rgb, history_hidden=None)
        h = h.to(dtype=unet_dtype)
        p = p.to(dtype=unet_dtype)

        text_e = sd_state.get("prompt_embeds")
        pooled_t = sd_state.get("pooled_prompt_embeds")
        if text_e is not None:
            th = text_e.expand(1, -1, -1).to(device=device, dtype=unet_dtype)
            tp = pooled_t.expand(1, -1).to(device=device, dtype=unet_dtype)
            enc_h = torch.cat([th, h], dim=1)
            pool_p = (tp + p) / 2
        else:
            enc_h = h
            pool_p = p

        ref_norm = rgb_t.float() * 2 - 1
        ref_lat = vae.encode(ref_norm.to(vae_dtype)).latent_dist.sample()
        ref_lat = (ref_lat - shift_val) * scale
        clean = ref_lat.to(unet_dtype)

        n = torch.randn_like(clean)
        latents = (1.0 - noise) * clean + noise * n
        ts = torch.linspace(noise, 0.0, steps + 1, device=device)
        for i in range(steps):
            dt = ts[i + 1] - ts[i]
            tb = torch.full((1,), ts[i].item() * 1000, device=device, dtype=unet_dtype)
            out = unet(hidden_states=latents, timestep=tb,
                       encoder_hidden_states=enc_h, pooled_projections=pool_p,
                       return_dict=False)
            v = out[0] if isinstance(out, tuple) else out
            latents = latents + dt * v

        if latent_anchor > 0 and anchor_img is not None:
            a_rgb = anchor_img.unsqueeze(0).to(device)
            a_norm = a_rgb.float() * 2 - 1
            a_lat = vae.encode(a_norm.to(vae_dtype)).latent_dist.sample()
            a_lat = (a_lat - shift_val) * scale
            a_lat = a_lat.to(latents.dtype)
            latents = (1.0 - latent_anchor) * latents + latent_anchor * a_lat

        latents = latents.to(dtype=vae_dtype)
        lu = (latents / scale) + shift_val
        img = vae.decode(lu, return_dict=False)[0]
        result = ((img.float() + 1) / 2).clamp(0, 1)
    return result.squeeze(0).cpu()


def run_sweep(checkpoint_path, output_path, device_name="cuda", steps=30):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    encoder, predictor, sd_state = load_full_model(checkpoint_path, device)

    gt_images = []
    for year in YEARS:
        path = os.path.join(REGION_DIR, FILENAME_TEMPLATE.format(year=year))
        gt_images.append(load_and_preprocess(path))

    if gt_images[0] is None:
        print("Error: 2017 missing!")
        return

    # Parameter grid
    configs = [
        {"noise": 0.25, "latent_anchor": 0.05, "label": "n=0.25 a=0.05"},
        {"noise": 0.20, "latent_anchor": 0.10, "label": "n=0.20 a=0.10"},
        {"noise": 0.15, "latent_anchor": 0.10, "label": "n=0.15 a=0.10"},
        {"noise": 0.15, "latent_anchor": 0.15, "label": "n=0.15 a=0.15"},
        {"noise": 0.10, "latent_anchor": 0.15, "label": "n=0.10 a=0.15"},
        {"noise": 0.10, "latent_anchor": 0.20, "label": "n=0.10 a=0.20"},
    ]

    all_results = {}
    all_psnrs = {}

    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*50}")
        print(f"Config: {label}")
        print(f"{'='*50}")

        preds = [gt_images[0]]
        current = gt_images[0]
        anchor = gt_images[0]
        psnrs = []

        for i in range(len(YEARS) - 1):
            pred = predict_one(
                encoder, predictor, sd_state,
                input_img=current, anchor_img=anchor,
                device=device, steps=steps,
                noise=cfg["noise"], latent_anchor=cfg["latent_anchor"],
            )
            # Color fix
            pred = match_color_statistics(
                pred.unsqueeze(0), anchor.unsqueeze(0)
            ).squeeze(0)

            preds.append(pred)
            current = pred

            psnr = 0.0
            if gt_images[i + 1] is not None:
                psnr = compute_psnr(pred, gt_images[i + 1])
            psnrs.append(psnr)
            print(f"  {YEARS[i+1]}: PSNR={psnr:.1f}dB")

        all_results[label] = preds
        all_psnrs[label] = psnrs
        avg_psnr = np.mean(psnrs)
        print(f"  AVG PSNR: {avg_psnr:.1f}dB")

    # ====================================================================
    # PSNR Summary Table
    # ====================================================================
    print(f"\n{'='*70}")
    print("PSNR COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Config':<20}" + "".join(f"{y:>8}" for y in YEARS[1:]) + f"{'AVG':>8}"
    print(header)
    print("-" * len(header))

    best_label = None
    best_avg = 0

    for label, psnrs in all_psnrs.items():
        avg = np.mean(psnrs)
        row = f"{label:<20}" + "".join(f"{p:>8.1f}" for p in psnrs) + f"{avg:>8.1f}"
        print(row)
        if avg > best_avg:
            best_avg = avg
            best_label = label

    print(f"\nBEST: {best_label} (avg PSNR={best_avg:.1f}dB)")

    # ====================================================================
    # Visual comparison
    # ====================================================================
    n_configs = len(configs)
    n_years = len(YEARS)
    fig, axes = plt.subplots(1 + n_configs, n_years, figsize=(22, 3 * (1 + n_configs)))

    # Row 0: GT
    for i, year in enumerate(YEARS):
        ax = axes[0, i]
        if gt_images[i] is not None:
            ax.imshow(normalize_for_display(gt_images[i]))
        ax.set_title(f"{year}", fontsize=9)
        ax.axis('off')
    axes[0, 0].set_ylabel("GT", fontsize=11, rotation=0, labelpad=30, va='center')

    # Remaining rows: each config
    for row_idx, cfg in enumerate(configs):
        label = cfg["label"]
        preds = all_results[label]
        avg = np.mean(all_psnrs[label])
        for i, year in enumerate(YEARS):
            ax = axes[1 + row_idx, i]
            ax.imshow(normalize_for_display(preds[i]))
            if i > 0:
                p = all_psnrs[label][i - 1]
                ax.set_title(f"{p:.1f}dB", fontsize=8)
            ax.axis('off')
        axes[1 + row_idx, 0].set_ylabel(
            f"{label}\navg={avg:.1f}",
            fontsize=9, rotation=0, labelpad=55, va='center'
        )

    plt.tight_layout()
    plt.suptitle(f"Parameter Sweep (steps={steps}, color_fix=ON)", fontsize=14, y=1.01)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="parameter_sweep.png")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    run_sweep(args.checkpoint, args.output, steps=args.steps)