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


# ============================================================================
# Caption Metrics: BLEU 1-4, ROUGE-L, CIDEr
# ============================================================================

_HAS_NLTK = None
_HAS_ROUGE = None


def _ensure_nltk():
    global _HAS_NLTK
    if _HAS_NLTK is not None:
        return _HAS_NLTK
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _HAS_NLTK = True
    except ImportError:
        _HAS_NLTK = False
    return _HAS_NLTK


def _ensure_rouge():
    global _HAS_ROUGE
    if _HAS_ROUGE is not None:
        return _HAS_ROUGE
    try:
        from rouge_score import rouge_scorer  # noqa
        _HAS_ROUGE = True
    except ImportError:
        _HAS_ROUGE = False
    return _HAS_ROUGE


def _tokenize_caption(text: str):
    """Tokenize a caption string into words."""
    if _ensure_nltk():
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    else:
        # Fallback: simple split
        return text.lower().split()


def compute_bleu(pred_tokens, ref_tokens, max_n=4):
    """
    Compute BLEU 1-4 for a single prediction vs single reference.
    Returns dict: {bleu_1, bleu_2, bleu_3, bleu_4}
    """
    from collections import Counter

    scores = {}
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        # Clipped counts
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(sum(pred_ngrams.values()), 1)

        precision = clipped / total
        scores[f"bleu_{n}"] = precision

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    for k in scores:
        scores[k] *= bp

    return scores


def compute_rouge_l(pred_tokens, ref_tokens):
    """
    Compute ROUGE-L F1 score using longest common subsequence.
    """
    m, n = len(ref_tokens), len(pred_tokens)
    if m == 0 or n == 0:
        return 0.0

    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_cider_single(pred_tokens, ref_tokens):
    """
    Simplified CIDEr-like score for a single pred/ref pair.
    Uses TF-IDF weighted n-gram overlap (n=1..4).
    This is a lightweight approximation — full CIDEr needs corpus-level IDF.
    """
    from collections import Counter

    score = 0.0
    for n in range(1, 5):
        pred_ng = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ng = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))

        # Cosine similarity between n-gram TF vectors
        common = set(pred_ng.keys()) & set(ref_ng.keys())
        if not common:
            continue

        dot = sum(pred_ng[ng] * ref_ng[ng] for ng in common)
        norm_p = math.sqrt(sum(v ** 2 for v in pred_ng.values()))
        norm_r = math.sqrt(sum(v ** 2 for v in ref_ng.values()))

        if norm_p > 0 and norm_r > 0:
            score += dot / (norm_p * norm_r)

    return score / 4.0  # Average over n=1..4


def compute_caption_metrics(
    pred_captions: list,
    ref_captions: list,
) -> Dict[str, float]:
    """
    Compute caption quality metrics for a batch of predictions vs references.

    Args:
        pred_captions: list of predicted caption strings (length B)
        ref_captions:  list of reference caption strings (length B)

    Returns:
        dict with keys: bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, cider
    """
    assert len(pred_captions) == len(ref_captions), \
        f"Batch size mismatch: {len(pred_captions)} vs {len(ref_captions)}"

    B = len(pred_captions)
    if B == 0:
        return {f"bleu_{n}": 0.0 for n in range(1, 5)} | {"rouge_l": 0.0, "cider": 0.0}

    # Accumulate per-sample scores
    bleu_accum = {f"bleu_{n}": 0.0 for n in range(1, 5)}
    rouge_accum = 0.0
    cider_accum = 0.0

    for pred_str, ref_str in zip(pred_captions, ref_captions):
        # Skip empty
        if not pred_str.strip() or not ref_str.strip():
            continue

        pred_tok = _tokenize_caption(pred_str)
        ref_tok = _tokenize_caption(ref_str)

        if len(pred_tok) == 0 or len(ref_tok) == 0:
            continue

        # BLEU
        bleu = compute_bleu(pred_tok, ref_tok)
        for k, v in bleu.items():
            bleu_accum[k] += v

        # ROUGE-L
        rouge_accum += compute_rouge_l(pred_tok, ref_tok)

        # CIDEr (simplified)
        cider_accum += compute_cider_single(pred_tok, ref_tok)

    # Average
    metrics = {}
    for k, v in bleu_accum.items():
        metrics[k] = v / B
    metrics["rouge_l"] = rouge_accum / B
    metrics["cider"] = cider_accum / B

    return metrics


def compute_caption_metrics_rouge_score(
    pred_captions: list,
    ref_captions: list,
) -> Dict[str, float]:
    """
    Alternative: use the `rouge-score` package for ROUGE-L (more accurate).
    Falls back to compute_caption_metrics() if package not available.

    Install: pip install rouge-score
    """
    if not _ensure_rouge():
        return compute_caption_metrics(pred_captions, ref_captions)

    from rouge_score import rouge_scorer

    # Start with BLEU + CIDEr from our implementation
    metrics = compute_caption_metrics(pred_captions, ref_captions)

    # Override ROUGE-L with the official implementation
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for pred_str, ref_str in zip(pred_captions, ref_captions):
        if pred_str.strip() and ref_str.strip():
            result = scorer.score(ref_str, pred_str)
            rouge_scores.append(result["rougeL"].fmeasure)

    if rouge_scores:
        metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

    return metrics