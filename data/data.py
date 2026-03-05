import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window


YEAR_REGEX = re.compile(r"_(\d{4})_")


def extract_year(path: str) -> Optional[int]:
    """Extract 4-digit year from filename using pattern _YYYY_."""
    m = YEAR_REGEX.search(os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


def check_bad_tifs(root_dir: str) -> List[str]:
    """
    Scan all .tif files under root_dir and return a list of files
    that cannot be opened by rasterio.
    """
    bad_files: List[str] = []

    if not os.path.isdir(root_dir):
        print(f"[check_bad_tifs] root_dir is not a directory: {root_dir}")
        return bad_files

    for region in os.listdir(root_dir):
        rpath = os.path.join(root_dir, region)
        if not os.path.isdir(rpath):
            continue

        for fname in os.listdir(rpath):
            if not fname.lower().endswith(".tif"):
                continue
            fpath = os.path.join(rpath, fname)

            try:
                with rasterio.open(fpath) as src:
                    _ = src.profile
            except Exception:
                print("BAD FILE (open failed):", fpath)
                bad_files.append(fpath)

    print("\nTOTAL BAD FILES (open failed):", len(bad_files))
    return bad_files


class S2FutureEmbeddingDataset(Dataset):
    """
    Dataset for Sentinel-2 future embedding + RGB prediction.

    For each region directory, expects files like:
        embedding_<region>_2017_10km.tif
        s2_rgb_<region>_2017_10km.tif

    It will create samples of the form:
        RGB(year t,  patch) -> (EMBEDDING(year t+1, patch), RGB(year t+1, patch))

    __getitem__ return:
        rgb_t, emb_tp1, rgb_tp1, meta
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 128,
        rgb_prefix: str = "s2_rgb_",
        emb_prefix: str = "embedding_",
        transform: Optional[Callable] = None,
        allowed_regions: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            root_dir: Path to the folder containing region subfolders.
            patch_size: Size of square patches (in pixels).
            rgb_prefix: Filename prefix for RGB tif files.
            emb_prefix: Filename prefix for embedding tif files.
            transform: Optional callable (rgb_t, emb_tp1, rgb_tp1, meta)
                       -> (rgb_t, emb_tp1, rgb_tp1, meta).
            allowed_regions: If not None, only use these region folder names.
            year_range: Optional (min_year, max_year) filter for year_t.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.rgb_prefix = rgb_prefix
        self.emb_prefix = emb_prefix
        self.transform = transform
        self.allowed_regions = set(allowed_regions) if allowed_regions else None
        self.year_range = year_range

        self.samples: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan folders and build list of patch-level samples."""
        root = self.root_dir
        if not os.path.isdir(root):
            raise ValueError(f"root_dir does not exist or is not a directory: {root}")

        for region_name in sorted(os.listdir(root)):
            region_dir = os.path.join(root, region_name)
            if not os.path.isdir(region_dir):
                continue
            if self.allowed_regions is not None and region_name not in self.allowed_regions:
                continue

            # Collect per-year file paths for this region
            rgb_by_year: Dict[int, str] = {}
            emb_by_year: Dict[int, str] = {}

            for fname in os.listdir(region_dir):
                if not fname.lower().endswith(".tif"):
                    continue
                fpath = os.path.join(region_dir, fname)
                year = extract_year(fname)
                if year is None:
                    continue

                if fname.startswith(self.rgb_prefix):
                    rgb_by_year[year] = fpath
                elif fname.startswith(self.emb_prefix):
                    emb_by_year[year] = fpath

            if not rgb_by_year or not emb_by_year:
                continue

            # For each year t, we want: RGB_t, EMB_{t+1}, RGB_{t+1}
            all_years = sorted(rgb_by_year.keys())
            for year_t in all_years:
                year_tp1 = year_t + 1

                if year_t not in rgb_by_year:
                    continue
                if year_tp1 not in emb_by_year:
                    continue
                if year_tp1 not in rgb_by_year:
                    # we need RGB_{t+1} for diffusion loss
                    continue

                if self.year_range is not None:
                    min_y, max_y = self.year_range
                    if not (min_y <= year_t <= max_y):
                        continue

                rgb_t_path = rgb_by_year[year_t]
                emb_tp1_path = emb_by_year[year_tp1]
                rgb_tp1_path = rgb_by_year[year_tp1]

                # Determine patch grid for this triple
                with rasterio.open(rgb_t_path) as src_rgb_t, \
                     rasterio.open(emb_tp1_path) as src_emb_tp1, \
                     rasterio.open(rgb_tp1_path) as src_rgb_tp1:

                    if not (
                        src_rgb_t.width == src_emb_tp1.width == src_rgb_tp1.width
                        and src_rgb_t.height == src_emb_tp1.height == src_rgb_tp1.height
                    ):
                        raise ValueError(
                            f"Size mismatch in region {region_name}, year {year_t}: "
                            f"rgb_t={src_rgb_t.width}x{src_rgb_t.height}, "
                            f"emb_tp1={src_emb_tp1.width}x{src_emb_tp1.height}, "
                            f"rgb_tp1={src_rgb_tp1.width}x{src_rgb_tp1.height}"
                        )

                    width, height = src_rgb_t.width, src_rgb_t.height

                ps = self.patch_size
                n_patches_x = width // ps
                n_patches_y = height // ps

                if n_patches_x == 0 or n_patches_y == 0:
                    raise ValueError(
                        f"Image too small for patch_size={ps}: {rgb_t_path} "
                        f"({width}x{height})"
                    )

                for iy in range(n_patches_y):
                    for ix in range(n_patches_x):
                        x0 = ix * ps
                        y0 = iy * ps

                        sample = {
                            "rgb_t_path": rgb_t_path,
                            "emb_tp1_path": emb_tp1_path,
                            "rgb_tp1_path": rgb_tp1_path,
                            "region": region_name,
                            "year_t": year_t,
                            "year_tp1": year_tp1,
                            "x": x0,
                            "y": y0,
                        }
                        self.samples.append(sample)

        if not self.samples:
            raise RuntimeError("No samples found. Check root_dir and file naming pattern.")

        print(f"[S2FutureEmbeddingDataset] Found {len(self.samples)} patch samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_patch(self, path: str, x: int, y: int) -> np.ndarray:
        """
        Load a patch (C, H, W) from a TIFF file using rasterio.

        If reading fails (corrupted tile, etc.), return a zero patch with a
        reasonable number of channels instead of raising, to avoid crashing
        the training loop.
        """
        ps = self.patch_size
        window = Window(x, y, ps, ps)

        try:
            with rasterio.open(path) as src:
                arr = src.read(window=window)  # (bands, H, W)
            return arr
        except Exception as e:
            print(f"[WARN] Failed to read patch from {path} at (x={x}, y={y}): {e}")

            # Heuristic: choose number of channels based on filename
            fname = os.path.basename(path).lower()
            if "embedding_" in fname:
                nbands = 64
            elif "s2_rgb_" in fname:
                nbands = 3
            else:
                nbands = 3

            # Return zeros so that training can continue
            arr = np.zeros((nbands, ps, ps), dtype=np.float32)
            return arr

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        rgb_t_np = self._load_patch(sample["rgb_t_path"], sample["x"], sample["y"])
        emb_tp1_np = self._load_patch(sample["emb_tp1_path"], sample["x"], sample["y"])
        rgb_tp1_np = self._load_patch(sample["rgb_tp1_path"], sample["x"], sample["y"])

        # Convert to float32 tensors
        rgb_t = torch.from_numpy(rgb_t_np).float()      # (C_rgb, H, W)
        emb_tp1 = torch.from_numpy(emb_tp1_np).float()  # (C_emb=64, H, W) expected
        rgb_tp1 = torch.from_numpy(rgb_tp1_np).float()  # (C_rgb, H, W)

        meta = {
            "region": sample["region"],
            "year_t": sample["year_t"],
            "year_tp1": sample["year_tp1"],
            "x": sample["x"],
            "y": sample["y"],
            "rgb_t_path": sample["rgb_t_path"],
            "emb_tp1_path": sample["emb_tp1_path"],
            "rgb_tp1_path": sample["rgb_tp1_path"],
        }

        if self.transform is not None:
            rgb_t, emb_tp1, rgb_tp1, meta = self.transform(rgb_t, emb_tp1, rgb_tp1, meta)

        return rgb_t, emb_tp1, rgb_tp1, meta


if __name__ == "__main__":
    # Optional quick check
    root = "downloads"
    print("Checking for bad GeoTIFF files under:", root)
    check_bad_tifs(root)

    dataset = S2FutureEmbeddingDataset(
        root_dir=root,
        patch_size=128,
    )
    rgb_t, emb_tp1, rgb_tp1, meta = dataset[0]
    print("RGB_t shape:", rgb_t.shape)
    print("EMB_tp1 shape:", emb_tp1.shape)
    print("RGB_tp1 shape:", rgb_tp1.shape)
    print("Meta:", meta)
