"""
SD Joint Loss Module - Multi-Caption Conditioning (No Coarse RGB)
================================================================
All conditioning comes from:
1. IJEPA predicted tokens (visual embeddings)
2. Informative caption (t) - encoded text
3. Geometric caption (t) - encoded text
4. Semantic caption (t+1) - forecasted encoded text
"""

import torch
import torch.nn.functional as F
from metrics import compute_all_metrics


def gaussian_kernel(size=11, sigma=1.5, channels=3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1, img2, window_size=11, sigma=1.5):
    C = img1.size(1)
    kernel = gaussian_kernel(window_size, sigma, C, img1.device)
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=C)
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_loss(img1, img2):
    return 1.0 - ssim(img1, img2)


def diffusion_sample(
    unet, vae, scheduler, cond_adapter, ijepa_tokens,
    num_steps=20, image_size=(128, 128), device=None,
    noise_strength=0.35,
    # Caption conditioning
    informative_hidden=None, informative_pooled=None,
    geometric_hidden=None, geometric_pooled=None,
    semantic_hidden=None, semantic_pooled=None,
    # Optional: reference image for img2img style init
    ref_rgb=None,
):
    """
    Diffusion sampling with multi-caption + IJEPA conditioning.
    No coarse RGB. Conditioning is purely from embeddings and captions.
    
    If ref_rgb is provided, it is used ONLY for VAE-encoding a starting
    latent (img2img style). It is NOT passed to the adapter.
    """
    if device is None:
        device = ijepa_tokens.device

    B = ijepa_tokens.shape[0]
    H, W = image_size

    unet_dtype = next(unet.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype

    scale = getattr(vae.config, "scaling_factor", 1.5305)
    shift = getattr(vae.config, "shift_factor", 0.0609)

    latent_h, latent_w = H // 8, W // 8
    latent_ch = getattr(vae.config, "latent_channels", 16)

    # Starting latent: from ref image or pure noise
    if ref_rgb is not None:
        ref_norm = ref_rgb * 2 - 1
        with torch.no_grad():
            ref_latent = vae.encode(ref_norm.to(vae_dtype)).latent_dist.sample()
            ref_latent = (ref_latent - shift) * scale
        latents = ref_latent.to(unet_dtype)
    else:
        latents = torch.randn(B, latent_ch, latent_h, latent_w, device=device, dtype=unet_dtype)

    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - noise_strength) * latents + noise_strength * noise

    # Get conditioning from adapter (captions + IJEPA only)
    encoder_hidden, pooled_proj = cond_adapter(
        ijepa_tokens,
        informative_hidden=informative_hidden,
        informative_pooled=informative_pooled,
        geometric_hidden=geometric_hidden,
        geometric_pooled=geometric_pooled,
        semantic_hidden=semantic_hidden,
        semantic_pooled=semantic_pooled,
    )
    encoder_hidden = encoder_hidden.to(dtype=unet_dtype)
    pooled_proj = pooled_proj.to(dtype=unet_dtype)

    # Single denoise step
    t_batch = torch.full((B,), noise_strength * 1000, device=device, dtype=unet_dtype)

    with torch.no_grad():
        out = unet(
            hidden_states=noisy_latents,
            timestep=t_batch,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled_proj,
            return_dict=False,
        )
    velocity = out[0] if isinstance(out, tuple) else out
    denoised_latents = noisy_latents - noise_strength * velocity

    # Decode
    denoised_latents = denoised_latents.to(dtype=vae_dtype)
    latents_unscaled = (denoised_latents / scale) + shift

    with torch.no_grad():
        image = vae.decode(latents_unscaled, return_dict=False)[0]

    return ((image.float() + 1) / 2).clamp(0, 1)


def compute_sd_loss_and_metrics(
    sd_state,
    ijepa_tokens,
    rgb_target,
    step=0,
    gt_tokens=None,
    save_vis=False,
    vis_folder="./vis",
    do_diffusion_sample=False,
    diffusion_steps=20,
    ssim_weight=0.1,
    # Caption conditioning
    informative_hidden=None, informative_pooled=None,
    geometric_hidden=None, geometric_pooled=None,
    semantic_hidden=None, semantic_pooled=None,
    caption_dropout_prob=0.1,
    # Optional ref for img2img init during sampling only
    rgb_source=None,
    **kwargs,
):
    """
    Compute Flow Matching Loss with Multi-Caption Conditioning.
    No coarse RGB. All conditioning from captions + IJEPA embeddings.
    """
    unet = sd_state["unet"]
    vae = sd_state["vae"]
    scheduler = sd_state["noise_scheduler"]
    cond_adapter = sd_state["cond_adapter"]

    device = rgb_target.device
    B = rgb_target.size(0)
    H, W = rgb_target.shape[-2:]

    unet_dtype = next(unet.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype

    scale = getattr(vae.config, "scaling_factor", 1.5305)
    shift = getattr(vae.config, "shift_factor", 0.0609)

    # 1. Encode target
    target_norm = rgb_target * 2 - 1
    with torch.no_grad():
        target_latents = (vae.encode(target_norm.to(vae_dtype)).latent_dist.sample() - shift) * scale
    target_latents = target_latents.to(unet_dtype)

    # 2. Caption dropout during training
    train_info_h, train_info_p = informative_hidden, informative_pooled
    train_geo_h, train_geo_p = geometric_hidden, geometric_pooled
    train_sem_h, train_sem_p = semantic_hidden, semantic_pooled

    if unet.training:
        if informative_hidden is not None and torch.rand(1).item() < caption_dropout_prob:
            train_info_h, train_info_p = None, None
        if geometric_hidden is not None and torch.rand(1).item() < caption_dropout_prob:
            train_geo_h, train_geo_p = None, None
        if semantic_hidden is not None and torch.rand(1).item() < caption_dropout_prob:
            train_sem_h, train_sem_p = None, None

    # 3. Adapter forward (captions + IJEPA only)
    encoder_hidden, pooled_proj = cond_adapter(
        ijepa_tokens,
        informative_hidden=train_info_h,
        informative_pooled=train_info_p,
        geometric_hidden=train_geo_h,
        geometric_pooled=train_geo_p,
        semantic_hidden=train_sem_h,
        semantic_pooled=train_sem_p,
        history_hidden=kwargs.get("history_hidden"),
    )
    encoder_hidden = encoder_hidden.to(unet_dtype)
    pooled_proj = pooled_proj.to(unet_dtype)

    actual_b = encoder_hidden.size(0)
    num_masks = actual_b // B
    if num_masks > 1:
        target_latents = target_latents.repeat_interleave(num_masks, dim=0)

    # 4. Flow matching
    noise = torch.randn_like(target_latents).to(unet_dtype)
    timesteps = torch.rand((actual_b,), device=device).to(unet_dtype)
    sigmas = timesteps.view(-1, 1, 1, 1)
    noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise

    # 5. UNet forward
    model_output = unet(
        hidden_states=noisy_latents,
        timestep=timesteps * 1000.0,
        encoder_hidden_states=encoder_hidden,
        pooled_projections=pooled_proj,
        return_dict=False
    )[0]

    # 6. Loss
    target_velocity = noise - target_latents
    mse_loss = F.mse_loss(model_output.float(), target_velocity.float())

    # 7. VAE reconstruction metrics
    with torch.no_grad():
        target_latents_first = target_latents[::num_masks] if num_masks > 1 else target_latents
        latents_unscaled = (target_latents_first / scale) + shift
        rec = vae.decode(latents_unscaled.to(vae_dtype), return_dict=False)[0]
        rec = ((rec.float() + 1) / 2).clamp(0, 1)

    ssim_val = ssim_loss(rec, rgb_target)
    total_sd_loss = mse_loss + ssim_weight * ssim_val

    # 8. Diffusion sampling (optional)
    gen_img = None
    if do_diffusion_sample:
        with torch.no_grad():
            gen_img = diffusion_sample(
                unet, vae, scheduler, cond_adapter, ijepa_tokens,
                num_steps=diffusion_steps,
                image_size=(H, W),
                device=device,
                informative_hidden=informative_hidden,
                informative_pooled=informative_pooled,
                geometric_hidden=geometric_hidden,
                geometric_pooled=geometric_pooled,
                semantic_hidden=semantic_hidden,
                semantic_pooled=semantic_pooled,
                ref_rgb=rgb_source,  # Only for latent init, NOT adapter
            )

    # 9. Metrics
    vae_m = compute_all_metrics(rec.detach(), rgb_target.detach())
    metrics = {f"vae_{k}": v for k, v in vae_m.items()}
    metrics["mse_loss"] = mse_loss.item()
    metrics["ssim_loss"] = ssim_val.item()
    metrics["captions_used"] = sum([
        train_info_h is not None,
        train_geo_h is not None,
        train_sem_h is not None,
    ])

    if gen_img is not None:
        gen_m = compute_all_metrics(gen_img.detach(), rgb_target.detach())
        metrics.update({f"gen_{k}": v for k, v in gen_m.items()})

    output = {"rec": rec, "target": rgb_target}
    if gen_img is not None:
        output["gen"] = gen_img

    return total_sd_loss, metrics, output