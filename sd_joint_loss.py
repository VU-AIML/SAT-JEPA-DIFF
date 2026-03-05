"""
SD Joint Loss Module - Coarse RGB Conditioning
Working version with single-step denoising
"""

import torch
import torch.nn.functional as F
from metrics import compute_all_metrics

COARSE_SIZE = 32


def gaussian_kernel(size=11, sigma=1.5, channels=3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1, img2, window_size=11, sigma=1.5):
    C = img1.size(1)
    kernel = gaussian_kernel(window_size, sigma, C, img1.device)
    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=C)
    sigma1_sq = F.conv2d(img1*img1, kernel, padding=window_size//2, groups=C) - mu1**2
    sigma2_sq = F.conv2d(img2*img2, kernel, padding=window_size//2, groups=C) - mu2**2
    sigma12 = F.conv2d(img1*img2, kernel, padding=window_size//2, groups=C) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_loss(img1, img2):
    return 1.0 - ssim(img1, img2)


def downsample_to_coarse(rgb, coarse_size=COARSE_SIZE):
    if rgb.shape[-1] == coarse_size and rgb.shape[-2] == coarse_size:
        return rgb
    return F.interpolate(rgb, size=(coarse_size, coarse_size), mode='bilinear', align_corners=False)


def match_color_statistics(generated, reference, eps=1e-6):
    """
    Match per-channel mean and std of generated image to reference.
    
    Args:
        generated: (B, 3, H, W) in [0, 1]
        reference: (B, 3, H, W) in [0, 1]
    Returns:
        (B, 3, H, W) color-corrected generated image, clamped to [0, 1]
    """
    # Per-channel stats over spatial dims
    gen_mean = generated.mean(dim=(2, 3), keepdim=True)
    gen_std = generated.std(dim=(2, 3), keepdim=True) + eps
    ref_mean = reference.mean(dim=(2, 3), keepdim=True)
    ref_std = reference.std(dim=(2, 3), keepdim=True) + eps
    
    # Normalize generated, then apply reference stats
    result = (generated - gen_mean) / gen_std * ref_std + ref_mean
    return result.clamp(0, 1)


def diffusion_sample(
    unet, vae, scheduler, cond_adapter, ijepa_tokens,
    text_embeds=None, pooled_text_embeds=None,
    num_steps=20, image_size=(128, 128), device=None,
    ref_rgb=None,
    coarse_size=COARSE_SIZE,
    noise_strength=0.35,
):
    """
    Simple working version:
    1. VAE encode ref_rgb -> latent
    2. Add noise
    3. Get conditioning from IJEPA + coarse RGB
    4. VAE decode (skip broken denoising for now)
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
    
    # 1. Coarse RGB for adapter
    coarse_rgb = None
    if ref_rgb is not None:
        coarse_rgb = downsample_to_coarse(ref_rgb, coarse_size)
    
    # 2. VAE encode
    if ref_rgb is not None:
        ref_norm = ref_rgb * 2 - 1
        with torch.no_grad():
            ref_latent = vae.encode(ref_norm.to(vae_dtype)).latent_dist.sample()
            ref_latent = (ref_latent - shift) * scale
        latents = ref_latent.to(unet_dtype)
    else:
        latents = torch.randn(B, latent_ch, latent_h, latent_w, device=device, dtype=unet_dtype)
    
    # 3. Add noise
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - noise_strength) * latents + noise_strength * noise
    
    # 4. Get conditioning
    ijepa_hidden, ijepa_pooled = cond_adapter(ijepa_tokens, ref_rgb=coarse_rgb)
    ijepa_hidden = ijepa_hidden.to(dtype=unet_dtype)
    ijepa_pooled = ijepa_pooled.to(dtype=unet_dtype)
    
    if text_embeds is not None:
        text_hidden = text_embeds.expand(B, -1, -1).to(device=device, dtype=unet_dtype)
        text_pooled = pooled_text_embeds.expand(B, -1).to(device=device, dtype=unet_dtype)
        encoder_hidden_states = torch.cat([text_hidden, ijepa_hidden], dim=1)
        pooled_projections = (text_pooled + ijepa_pooled) / 2
    else:
        encoder_hidden_states = ijepa_hidden
        pooled_projections = ijepa_pooled
    
    # 5. Single denoise step with UNet
    t_val = noise_strength
    t_batch = torch.full((B,), t_val * 1000, device=device, dtype=unet_dtype)
    
    with torch.no_grad():
        out = unet(
            hidden_states=noisy_latents,
            timestep=t_batch,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )
    velocity = out[0] if isinstance(out, tuple) else out
    
    # Simple velocity application
    denoised_latents = noisy_latents - noise_strength * velocity
    
    # 6. Decode
    denoised_latents = denoised_latents.to(dtype=vae_dtype)
    latents_unscaled = (denoised_latents / scale) + shift
    
    with torch.no_grad():
        image = vae.decode(latents_unscaled, return_dict=False)[0]
    
    return ((image.float() + 1) / 2).clamp(0, 1)


def compute_sd_loss_and_metrics(
    sd_state, 
    ijepa_tokens,
    rgb_target,
    rgb_source=None,
    step=0,
    gt_tokens=None,
    save_vis=False, 
    vis_folder="./vis",
    do_diffusion_sample=False, 
    diffusion_steps=20,
    ssim_weight=0.1,
    embedding_guidance_weight=0.0,
    ref_dropout_prob=0.15,
    coarse_size=COARSE_SIZE,
):
    """Compute Flow Matching Loss with Coarse RGB Conditioning."""
    unet = sd_state["unet"]
    vae = sd_state["vae"]
    scheduler = sd_state["noise_scheduler"]
    cond_adapter = sd_state["cond_adapter"]
    text_embeds = sd_state.get("prompt_embeds")
    pooled_text = sd_state.get("pooled_prompt_embeds")
    
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
    
    # 2. Coarse RGB
    coarse_rgb = None
    ref_was_used = False
    if rgb_source is not None:
        coarse_rgb = downsample_to_coarse(rgb_source, coarse_size)
        if unet.training and torch.rand(1).item() < ref_dropout_prob:
            coarse_rgb = None
        else:
            ref_was_used = True
    
    # 3. Adapter forward
    ijepa_hidden, ijepa_pooled = cond_adapter(ijepa_tokens, ref_rgb=coarse_rgb)
    ijepa_hidden = ijepa_hidden.to(unet_dtype)
    ijepa_pooled = ijepa_pooled.to(unet_dtype)
    
    actual_b = ijepa_hidden.size(0)
    num_masks = actual_b // B
    if num_masks > 1:
        target_latents = target_latents.repeat_interleave(num_masks, dim=0)
    
    # 4. Flow matching
    noise = torch.randn_like(target_latents).to(unet_dtype)
    timesteps = torch.rand((actual_b,), device=device).to(unet_dtype)
    sigmas = timesteps.view(-1, 1, 1, 1)
    noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise
    
    if text_embeds is not None:
        text_h = text_embeds.expand(actual_b, -1, -1).to(device=device, dtype=unet_dtype)
        text_p = pooled_text.expand(actual_b, -1).to(device=device, dtype=unet_dtype)
        encoder_hidden = torch.cat([text_h, ijepa_hidden], dim=1)
        pooled = (text_p + ijepa_pooled) / 2
    else:
        encoder_hidden = ijepa_hidden
        pooled = ijepa_pooled
    
    # 5. UNet forward
    model_output = unet(
        hidden_states=noisy_latents,
        timestep=timesteps * 1000.0,
        encoder_hidden_states=encoder_hidden,
        pooled_projections=pooled,
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
    
    # 8. Diffusion sampling
    gen_img = None
    if do_diffusion_sample:
        with torch.no_grad():
            gen_img = diffusion_sample(
                unet, vae, scheduler, cond_adapter, ijepa_tokens,
                text_embeds, pooled_text,
                num_steps=diffusion_steps, 
                image_size=(H, W),
                device=device,
                ref_rgb=rgb_source,
                coarse_size=coarse_size,
            )
    
    # 9. Metrics
    vae_m = compute_all_metrics(rec.detach(), rgb_target.detach())
    metrics = {f"vae_{k}": v for k, v in vae_m.items()}
    metrics["mse_loss"] = mse_loss.item()
    metrics["ssim_loss"] = ssim_val.item()
    metrics["ref_used"] = float(ref_was_used)
    
    if gen_img is not None:
        gen_m = compute_all_metrics(gen_img.detach(), rgb_target.detach())
        metrics.update({f"gen_{k}": v for k, v in gen_m.items()})
    
    output = {"rec": rec, "target": rgb_target}
    if rgb_source is not None:
        output["source"] = rgb_source
    if gen_img is not None:
        output["gen"] = gen_img
        
    return total_sd_loss, metrics, output