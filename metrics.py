# metrics.py
# Image quality metrics for satellite prediction / generation.
#
# Provided functions:
#   - compute_all_metrics(pred, target, device=None, use_lpips=False)
#       returns a dict with keys:
#         'l1', 'mse', 'psnr', 'ssim', 'gssim', 'lpips'
#
# Assumptions:
#   - pred, target: tensors of shape (B, C, H, W) with values in [0, 1]
#   - C is usually 3 (RGB). If not, SSIM is computed per-channel and averaged.

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

_LPIPS_MODEL = None
_HAS_LPIPS = None  # lazily determined


def _init_lpips(device: torch.device):
    """
    Lazy-init LPIPS model if 'lpips' package is available.
    """
    global _LPIPS_MODEL, _HAS_LPIPS
    if _HAS_LPIPS is not None:
        return _LPIPS_MODEL, _HAS_LPIPS

    try:
        import lpips  # type: ignore

        _LPIPS_MODEL = lpips.LPIPS(net="vgg").to(device)
        _LPIPS_MODEL.eval()
        _HAS_LPIPS = True
    except Exception:
        _LPIPS_MODEL = None
        _HAS_LPIPS = False

    return _LPIPS_MODEL, _HAS_LPIPS


def _to_0_1(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is in [0, 1]. If it looks like [-1, 1], rescale.
    """
    with torch.no_grad():
        mx = x.max()
        mn = x.min()
        # heuristic: if in [-1,1], map to [0,1]
        if mn >= -1.0 - 1e-3 and mx <= 1.0 + 1e-3 and (mn < 0.0 or mx > 1.0):
            return (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)


def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device):
    """
    Create a 2D Gaussian window for SSIM, shape (C, 1, K, K).
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # outer product -> 2D kernel
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    window_2d = window_2d.view(1, 1, window_size, window_size)
    window_2d = window_2d.repeat(channel, 1, 1, 1)  # (C,1,K,K)
    return window_2d


def _ssim_per_channel(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    C1: float,
    C2: float,
) -> torch.Tensor:
    """
    Compute SSIM per-channel and per-image, then average over spatial dims.
    img1, img2: (B, C, H, W) in [0,1].
    window: (C, 1, K, K).
    Returns: (B, C) tensor with SSIM per image per channel.
    """
    B, C, H, W = img1.shape
    padding = window.shape[-1] // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=C)
    mu2 = F.conv2d(img2, window, padding=padding, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=C) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    # average over H,W -> (B,C)
    ssim_val = ssim_map.flatten(2).mean(dim=-1)
    return ssim_val


def _compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    SSIM in [0,1], averaged over batch and channels.
    pred, target: (B, C, H, W) in [0,1].
    """
    device = pred.device
    B, C, H, W = pred.shape

    window = _gaussian_window(window_size, sigma, C, device)

    L = 1.0  # pixel range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_vals = _ssim_per_channel(pred, target, window, C1, C2)  # (B,C)
    return float(ssim_vals.mean().item())


def _gaussian_blur(x: torch.Tensor, window_size: int = 11, sigma: float = 2.0) -> torch.Tensor:
    """
    Simple Gaussian blur using same kernel as SSIM.
    """
    device = x.device
    B, C, H, W = x.shape
    window = _gaussian_window(window_size, sigma, C, device)
    padding = window_size // 2
    return F.conv2d(x, window, padding=padding, groups=C)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Optional[torch.device] = None,
    use_lpips: bool = False,
) -> Dict[str, float]:
    """
    Compute a set of image similarity metrics between pred and target.

    Args:
        pred:   (B, C, H, W) predicted images, typically in [0, 1] or [-1,1]
        target: (B, C, H, W) ground truth images, same size as pred
        device: optional torch.device for LPIPS model; if None, inferred from pred
        use_lpips: whether to try computing LPIPS (requires 'lpips' package)

    Returns:
        dict with keys:
            'l1', 'mse', 'psnr', 'ssim', 'gssim', 'lpips'
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in metrics: pred={pred.shape}, target={target.shape}"
        )

    if device is None:
        device = pred.device

    # ensure both in [0, 1]
    pred = _to_0_1(pred.detach())
    target = _to_0_1(target.detach())

    # ---- basic errors ----
    l1 = torch.mean(torch.abs(pred - target)).item()
    mse = torch.mean((pred - target) ** 2).item()

    if mse > 0:
        psnr = 10.0 * math.log10(1.0 / mse)
    else:
        psnr = float("inf")

    # ---- SSIM & G-SSIM ----
    ssim = _compute_ssim(pred, target)

    # G-SSIM: SSIM computed on Gaussian-blurred images
    pred_blur = _gaussian_blur(pred)
    target_blur = _gaussian_blur(target)
    gssim = _compute_ssim(pred_blur, target_blur)

    # ---- LPIPS (optional, default OFF) ----
    lpips_val = float("nan")
    if use_lpips:
        model, has_lp = _init_lpips(device)
        if has_lp and model is not None:
            # LPIPS expects inputs in [-1, 1]
            pred_lp = pred * 2.0 - 1.0
            target_lp = target * 2.0 - 1.0
            with torch.no_grad():
                val = model(pred_lp, target_lp)
                lpips_val = float(val.mean().item())

    metrics = {
        "l1": float(l1),
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "gssim": float(gssim),
        "lpips": float(lpips_val),
    }
    return metrics
