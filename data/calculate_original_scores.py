"""
Calculate Baseline Scores Between Consecutive Years

Bu script, yıl t ve yıl t+1 RGB görüntüleri arasındaki 
L1, MSE, PSNR, SSIM, GSSIM, LPIPS ve FID değerlerini hesaplar.

Böylece modelin baseline'dan ne kadar iyi/kötü olduğunu görebilirsin.

Gerekli paketler:
    pip install lpips pytorch-fid torchmetrics
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import json
from datetime import datetime

# LPIPS için
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARNING] lpips not installed. Install with: pip install lpips")

# FID için
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("[WARNING] torchmetrics not installed. Install with: pip install torchmetrics")


# ============== Metrics ==============

def gaussian_kernel(size=11, sigma=1.5, channels=3, device=None):
    """SSIM için Gaussian kernel oluştur."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11, sigma=1.5) -> float:
    """
    SSIM hesapla.
    img1, img2: (B, C, H, W) veya (C, H, W) tensor, [0, 1] aralığında
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
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
    
    return ssim_map.mean().item()


def compute_gssim(img1: torch.Tensor, img2: torch.Tensor, 
                  scales: List[float] = [1.0, 0.5, 0.25]) -> float:
    """
    Gradient-based SSIM (multi-scale edge-aware SSIM).
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Sobel kernels for gradient
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
        
        # Grayscale
        gray1 = img1_scaled.mean(dim=1, keepdim=True)
        gray2 = img2_scaled.mean(dim=1, keepdim=True)
        
        # Gradients
        grad1_x = F.conv2d(gray1, sobel_x, padding=1)
        grad1_y = F.conv2d(gray1, sobel_y, padding=1)
        grad2_x = F.conv2d(gray2, sobel_x, padding=1)
        grad2_y = F.conv2d(gray2, sobel_y, padding=1)
        
        grad1 = torch.sqrt(grad1_x**2 + grad1_y**2 + 1e-8)
        grad2 = torch.sqrt(grad2_x**2 + grad2_y**2 + 1e-8)
        
        # Normalize gradients to [0, 1]
        grad1 = (grad1 - grad1.min()) / (grad1.max() - grad1.min() + 1e-8)
        grad2 = (grad2 - grad2.min()) / (grad2.max() - grad2.min() + 1e-8)
        
        # SSIM on gradients
        grad1_3ch = grad1.repeat(1, 3, 1, 1)
        grad2_3ch = grad2.repeat(1, 3, 1, 1)
        ssim_scores.append(compute_ssim(grad1_3ch, grad2_3ch))
    
    return np.mean(ssim_scores)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val=1.0) -> float:
    """PSNR hesapla."""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0  # Neredeyse aynı
    return 10 * np.log10(max_val**2 / mse)


class LPIPSCalculator:
    """LPIPS hesaplayıcı - lazy loading ile."""
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        self.device = device
        self.net = net
        self._model = None
    
    @property
    def model(self):
        """Lazy load LPIPS model."""
        if self._model is None:
            if not LPIPS_AVAILABLE:
                raise RuntimeError("lpips not installed")
            self._model = lpips.LPIPS(net=self.net).to(self.device)
            self._model.eval()
        return self._model
    
    @torch.no_grad()
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        LPIPS hesapla.
        img1, img2: (C, H, W) veya (B, C, H, W), [0, 1] aralığında
        """
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # LPIPS expects [-1, 1] range
        img1_norm = img1 * 2.0 - 1.0
        img2_norm = img2 * 2.0 - 1.0
        
        # LPIPS requires at least 64x64 images
        if img1_norm.shape[-1] < 64 or img1_norm.shape[-2] < 64:
            img1_norm = F.interpolate(img1_norm, size=(64, 64), mode='bilinear', align_corners=False)
            img2_norm = F.interpolate(img2_norm, size=(64, 64), mode='bilinear', align_corners=False)
        
        lpips_val = self.model(img1_norm, img2_norm)
        return lpips_val.mean().item()


class FIDCalculator:
    """
    FID hesaplayıcı.
    
    FID, tek görüntü çifti için değil, distribution bazında hesaplanır.
    Bu yüzden tüm görüntüleri toplar ve sonunda hesaplar.
    """
    
    def __init__(self, feature_dim: int = 2048, device: str = 'cuda'):
        self.device = device
        self.feature_dim = feature_dim
        self._fid = None
        self.reset()
    
    @property
    def fid(self):
        """Lazy load FID calculator."""
        if self._fid is None:
            if not FID_AVAILABLE:
                raise RuntimeError("torchmetrics not installed")
            self._fid = FrechetInceptionDistance(
                feature=self.feature_dim,
                normalize=True  # Expects [0, 1] input
            ).to(self.device)
        return self._fid
    
    def reset(self):
        """Reset FID state for new calculation."""
        if self._fid is not None:
            self._fid.reset()
    
    @torch.no_grad()
    def update(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor):
        """
        Add batch of images to FID calculation.
        real_imgs, fake_imgs: (B, C, H, W) in [0, 1] range
        """
        # FID requires at least 64x64 images (for InceptionV3)
        if real_imgs.shape[-1] < 64 or real_imgs.shape[-2] < 64:
            real_imgs = F.interpolate(real_imgs, size=(64, 64), mode='bilinear', align_corners=False)
            fake_imgs = F.interpolate(fake_imgs, size=(64, 64), mode='bilinear', align_corners=False)
        
        # FID expects uint8 [0, 255] or float [0, 1] with normalize=True
        # Convert to uint8 for stability
        real_uint8 = (real_imgs.clamp(0, 1) * 255).to(torch.uint8)
        fake_uint8 = (fake_imgs.clamp(0, 1) * 255).to(torch.uint8)
        
        self.fid.update(real_uint8, real=True)
        self.fid.update(fake_uint8, real=False)
    
    def compute(self) -> float:
        """Compute FID score from accumulated images."""
        try:
            fid_val = self.fid.compute().item()
            return fid_val
        except Exception as e:
            print(f"[WARNING] FID computation failed: {e}")
            return float('nan')


def compute_all_metrics(
    img1: torch.Tensor, 
    img2: torch.Tensor,
    lpips_calc: Optional[LPIPSCalculator] = None,
) -> Dict[str, float]:
    """Tüm metrikleri hesapla (FID hariç - o ayrı hesaplanır)."""
    # Ensure [0, 1] range
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    l1 = F.l1_loss(img1, img2).item()
    mse = F.mse_loss(img1, img2).item()
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    gssim = compute_gssim(img1, img2)
    
    metrics = {
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "gssim": gssim,
    }
    
    # LPIPS (per-image metric)
    if lpips_calc is not None:
        try:
            lpips_val = lpips_calc.compute(img1, img2)
            metrics["lpips"] = lpips_val
        except Exception as e:
            print(f"[WARNING] LPIPS failed: {e}")
            metrics["lpips"] = float('nan')
    
    return metrics


# ============== Dataset ==============

YEAR_REGEX = re.compile(r"_(\d{4})_")


def extract_year(path: str) -> Optional[int]:
    m = YEAR_REGEX.search(os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


class ConsecutiveYearDataset(Dataset):
    """
    Ardışık yıllar arasındaki RGB görüntü çiftlerini yükleyen dataset.
    
    Return: rgb_t, rgb_tp1, meta
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 128,
        rgb_prefix: str = "s2_rgb_",
        allowed_regions: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.rgb_prefix = rgb_prefix
        self.allowed_regions = set(allowed_regions) if allowed_regions else None
        self.year_range = year_range
        self.max_samples = max_samples

        self.samples: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        root = self.root_dir
        if not os.path.isdir(root):
            raise ValueError(f"root_dir does not exist: {root}")

        for region_name in sorted(os.listdir(root)):
            region_dir = os.path.join(root, region_name)
            if not os.path.isdir(region_dir):
                continue
            if self.allowed_regions is not None and region_name not in self.allowed_regions:
                continue

            # Collect RGB files by year
            rgb_by_year: Dict[int, str] = {}

            for fname in os.listdir(region_dir):
                if not fname.lower().endswith(".tif"):
                    continue
                fpath = os.path.join(region_dir, fname)
                year = extract_year(fname)
                if year is None:
                    continue

                if fname.startswith(self.rgb_prefix):
                    rgb_by_year[year] = fpath

            if len(rgb_by_year) < 2:
                continue

            # Create pairs for consecutive years
            all_years = sorted(rgb_by_year.keys())
            for year_t in all_years:
                year_tp1 = year_t + 1

                if year_tp1 not in rgb_by_year:
                    continue

                if self.year_range is not None:
                    min_y, max_y = self.year_range
                    if not (min_y <= year_t <= max_y):
                        continue

                rgb_t_path = rgb_by_year[year_t]
                rgb_tp1_path = rgb_by_year[year_tp1]

                # Get dimensions
                try:
                    with rasterio.open(rgb_t_path) as src:
                        width, height = src.width, src.height
                except Exception as e:
                    print(f"[WARN] Cannot open {rgb_t_path}: {e}")
                    continue

                ps = self.patch_size
                n_patches_x = width // ps
                n_patches_y = height // ps

                if n_patches_x == 0 or n_patches_y == 0:
                    continue

                for iy in range(n_patches_y):
                    for ix in range(n_patches_x):
                        x0 = ix * ps
                        y0 = iy * ps

                        sample = {
                            "rgb_t_path": rgb_t_path,
                            "rgb_tp1_path": rgb_tp1_path,
                            "region": region_name,
                            "year_t": year_t,
                            "year_tp1": year_tp1,
                            "x": x0,
                            "y": y0,
                        }
                        self.samples.append(sample)

                        if self.max_samples and len(self.samples) >= self.max_samples:
                            print(f"[ConsecutiveYearDataset] Reached max_samples={self.max_samples}")
                            return

        print(f"[ConsecutiveYearDataset] Found {len(self.samples)} patch samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_patch(self, path: str, x: int, y: int) -> np.ndarray:
        ps = self.patch_size
        window = Window(x, y, ps, ps)

        try:
            with rasterio.open(path) as src:
                arr = src.read(window=window)
            return arr.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Failed to read patch from {path}: {e}")
            return np.zeros((3, ps, ps), dtype=np.float32)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        rgb_t_np = self._load_patch(sample["rgb_t_path"], sample["x"], sample["y"])
        rgb_tp1_np = self._load_patch(sample["rgb_tp1_path"], sample["x"], sample["y"])

        # Normalize to [0, 1] if needed (assuming uint8 or uint16)
        if rgb_t_np.max() > 1.0:
            if rgb_t_np.max() > 255:
                rgb_t_np = rgb_t_np / 65535.0
                rgb_tp1_np = rgb_tp1_np / 65535.0
            else:
                rgb_t_np = rgb_t_np / 255.0
                rgb_tp1_np = rgb_tp1_np / 255.0

        rgb_t = torch.from_numpy(rgb_t_np).float()
        rgb_tp1 = torch.from_numpy(rgb_tp1_np).float()

        meta = {
            "region": sample["region"],
            "year_t": sample["year_t"],
            "year_tp1": sample["year_tp1"],
            "x": sample["x"],
            "y": sample["y"],
        }

        return rgb_t, rgb_tp1, meta


# ============== Main Calculation ==============

def calculate_baseline_scores(
    root_dir: str,
    patch_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    output_file: str = "baseline_scores.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compute_lpips: bool = True,
    compute_fid: bool = True,
    lpips_net: str = 'alex',  # 'alex', 'vgg', or 'squeeze'
):
    """
    Tüm ardışık yıl çiftleri için baseline skorları hesapla.
    """
    print("=" * 60)
    print("BASELINE SCORE CALCULATION")
    print(f"Root dir: {root_dir}")
    print(f"Patch size: {patch_size}")
    print(f"Device: {device}")
    print(f"Compute LPIPS: {compute_lpips and LPIPS_AVAILABLE}")
    print(f"Compute FID: {compute_fid and FID_AVAILABLE}")
    print("=" * 60)

    # Dataset
    dataset = ConsecutiveYearDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize LPIPS calculator
    lpips_calc = None
    if compute_lpips and LPIPS_AVAILABLE:
        print(f"[INFO] Loading LPIPS model ({lpips_net})...")
        lpips_calc = LPIPSCalculator(net=lpips_net, device=device)

    # Initialize FID calculator
    fid_calc = None
    if compute_fid and FID_AVAILABLE:
        print("[INFO] Initializing FID calculator...")
        fid_calc = FIDCalculator(device=device)

    # Accumulate metrics
    all_metrics = {
        "l1": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
        "gssim": [],
    }
    if lpips_calc is not None:
        all_metrics["lpips"] = []
    
    # Per-region and per-year metrics
    region_metrics: Dict[str, Dict[str, List[float]]] = {}
    year_metrics: Dict[str, Dict[str, List[float]]] = {}

    print(f"\nProcessing {len(dataset)} samples...")
    
    for batch in tqdm(dataloader, desc="Computing metrics"):
        rgb_t, rgb_tp1, meta = batch
        
        rgb_t = rgb_t.to(device)
        rgb_tp1 = rgb_tp1.to(device)

        # Update FID with batch
        if fid_calc is not None:
            fid_calc.update(rgb_tp1, rgb_t)  # real=t+1, fake=t (baseline is "predicting" t as t+1)

        # Batch işleme for per-sample metrics
        B = rgb_t.size(0)
        for i in range(B):
            metrics = compute_all_metrics(rgb_t[i], rgb_tp1[i], lpips_calc=lpips_calc)
            
            region = meta["region"][i]
            year_pair = f"{meta['year_t'][i].item()}->{meta['year_tp1'][i].item()}"
            
            # Global metrics
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
            
            # Per-region
            if region not in region_metrics:
                region_metrics[region] = {k: [] for k in all_metrics.keys()}
            for k, v in metrics.items():
                if k not in region_metrics[region]:
                    region_metrics[region][k] = []
                region_metrics[region][k].append(v)
            
            # Per-year pair
            if year_pair not in year_metrics:
                year_metrics[year_pair] = {k: [] for k in all_metrics.keys()}
            for k, v in metrics.items():
                if k not in year_metrics[year_pair]:
                    year_metrics[year_pair][k] = []
                year_metrics[year_pair][k].append(v)

    # Compute FID (global only - requires distribution)
    fid_score = float('nan')
    if fid_calc is not None:
        print("\n[INFO] Computing FID score...")
        fid_score = fid_calc.compute()
        print(f"[INFO] FID = {fid_score:.4f}")

    # Aggregate results
    def aggregate(metric_dict):
        result = {}
        for k, v in metric_dict.items():
            # Filter out NaN values
            v_clean = [x for x in v if not np.isnan(x)]
            if len(v_clean) > 0:
                result[k] = {
                    "mean": float(np.mean(v_clean)),
                    "std": float(np.std(v_clean)),
                    "min": float(np.min(v_clean)),
                    "max": float(np.max(v_clean)),
                    "count": len(v_clean),
                }
            else:
                result[k] = {
                    "mean": float('nan'),
                    "std": float('nan'),
                    "min": float('nan'),
                    "max": float('nan'),
                    "count": 0,
                }
        return result

    results = {
        "metadata": {
            "root_dir": root_dir,
            "patch_size": patch_size,
            "total_samples": len(dataset),
            "timestamp": datetime.now().isoformat(),
            "lpips_net": lpips_net if lpips_calc else None,
        },
        "global": aggregate(all_metrics),
        "fid": fid_score,  # FID is global only
        "by_region": {r: aggregate(m) for r, m in region_metrics.items()},
        "by_year_pair": {y: aggregate(m) for y, m in year_metrics.items()},
    }

    # Print summary
    print("\n" + "=" * 60)
    print("GLOBAL BASELINE SCORES (Year t vs Year t+1)")
    print("=" * 60)
    for metric, stats in results["global"].items():
        print(f"{metric.upper():>8}: {stats['mean']:.6f} ± {stats['std']:.6f} "
              f"(min={stats['min']:.6f}, max={stats['max']:.6f})")
    
    if not np.isnan(fid_score):
        print(f"{'FID':>8}: {fid_score:.4f} (global distribution metric)")

    print("\n" + "-" * 60)
    print("BY YEAR PAIR:")
    print("-" * 60)
    for year_pair, metrics in sorted(results["by_year_pair"].items()):
        print(f"\n{year_pair}:")
        for metric, stats in metrics.items():
            print(f"  {metric.upper():>8}: {stats['mean']:.6f} ± {stats['std']:.6f}")

    print("\n" + "-" * 60)
    print("BY REGION (top 5 by sample count):")
    print("-" * 60)
    sorted_regions = sorted(
        results["by_region"].items(), 
        key=lambda x: x[1]["l1"]["count"], 
        reverse=True
    )[:5]
    for region, metrics in sorted_regions:
        print(f"\n{region} (n={metrics['l1']['count']}):")
        for metric, stats in metrics.items():
            print(f"  {metric.upper():>8}: {stats['mean']:.6f} ± {stats['std']:.6f}")

    # Save results
    # Handle NaN for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    results_clean = clean_for_json(results)
    
    with open(output_file, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate baseline scores between consecutive year RGB images"
    )
    parser.add_argument(
        "--root_dir", "-r",
        type=str,
        default="downloads",
        help="Root directory containing region subfolders"
    )
    parser.add_argument(
        "--patch_size", "-p",
        type=int,
        default=128,
        help="Patch size (default: 128)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--max_samples", "-m",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="baseline_scores.json",
        help="Output JSON file (default: baseline_scores.json)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage"
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip LPIPS calculation"
    )
    parser.add_argument(
        "--no-fid",
        action="store_true",
        help="Skip FID calculation"
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS network backbone (default: alex)"
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    calculate_baseline_scores(
        root_dir=args.root_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        output_file=args.output,
        device=device,
        compute_lpips=not args.no_lpips,
        compute_fid=not args.no_fid,
        lpips_net=args.lpips_net,
    )


if __name__ == "__main__":
    main()