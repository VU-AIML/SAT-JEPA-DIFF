"""
Caption Generation with EarthDial — FAST HYBRID
================================================
1) Try single-call (1 inference → 3 captions, 3x faster)
2) Parse & quality check: informative must be >100 chars
3) If bad → fallback to 3 separate calls (guaranteed quality)

This way: ~70% of patches use fast path, ~30% use slow path.
Average ~6s/patch → 29K in ~48 hours (~2 days).
"""

import os
import re
import json
import time
import subprocess
import argparse
import warnings
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import rasterio
from rasterio.windows import Window

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from earthdial.model.internvl_chat import InternVLChatModel
    from earthdial.train.dataset import build_transform
    HAS_EARTHDIAL = True
except ImportError:
    HAS_EARTHDIAL = False
    print("[ERROR] earthdial not found. Install: cd EarthDial && pip install -e .")

# =============================================================================
# Prompts
# =============================================================================

SINGLE_PROMPT = (
    "Describe this satellite image in three separate sections. "
    "Write each section after its header. Be detailed.\n\n"
    "### INFORMATIVE\n"
    "Describe this satellite image in great detail as a remote sensing expert. "
    "What land cover types are visible and what percentage of the image does each cover? "
    "What is the vegetation type, density, and health? "
    "Are there buildings, roads, or man-made structures? Describe their density and type. "
    "Are there water bodies? Describe their type and condition. "
    "Is there agricultural activity? Describe crop patterns, field sizes, irrigation. "
    "What does exposed soil look like? "
    "What direction are shadows pointing? What season does vegetation suggest? "
    "Are there unusual features? Write a thorough paragraph.\n\n"
    "### GEOMETRIC\n"
    "Describe only the geometric shapes and spatial layout. "
    "What shapes do parcels and boundaries form? What are their orientations? "
    "What is the road network pattern? What shapes are building footprints? "
    "Are there linear or circular features? How are objects clustered or dispersed? "
    "Do not mention colors or land use types.\n\n"
    "### SEMANTIC\n"
    "In 2-3 sentences, what is the main land use type, development density, and apparent season?"
)

FALLBACK_PROMPTS = {
    "informative": {
        "max_tokens": 512,
        "prompt": (
            "Describe this satellite image in great detail as a remote sensing expert would. "
            "What land cover types are visible and what percentage of the image does each cover? "
            "What is the vegetation type, density, and health condition? "
            "Are there any buildings, roads, or other man-made structures? If so, describe their density and type. "
            "Are there any water bodies? If so, describe their type and condition. "
            "Is there any agricultural activity? If so, describe the crop patterns, field sizes, and irrigation. "
            "What does the exposed soil look like? "
            "What direction are the shadows pointing and what does this tell about the sun position? "
            "What season does the vegetation suggest? "
            "Are there any unusual or anomalous features? "
            "Please be very thorough and write a long, detailed paragraph covering all visible features."
        ),
    },
    "geometric": {
        "max_tokens": 384,
        "prompt": (
            "Describe the geometric shapes and spatial layout you see in this satellite image. "
            "What shapes do the land parcels and field boundaries form? What are their orientations? "
            "What is the road network pattern - is it a grid, radial, or organic? What are the intersection angles? "
            "What shapes are the building footprints? How are they aligned and spaced? "
            "Are there any linear features like rivers, canals, or railways? Are they straight or curved? "
            "Are there any circular structures like center-pivot irrigation or roundabouts? "
            "How are objects clustered or dispersed across the image? "
            "Only describe shapes, geometry, and spatial arrangement. Do not describe colors or land use types."
        ),
    },
    "semantic": {
        "max_tokens": 150,
        "prompt": (
            "Describe the overall meaning of this satellite image in 2-3 sentences. "
            "What is the main land use type - urban, rural, agricultural, forest, water, or barren? "
            "How densely developed is the area? What season does the vegetation indicate?"
        ),
    },
}

MIN_INFORMATIVE_LEN = 100  # If shorter, fallback to 3 calls

YEAR_REGEX = re.compile(r"_(\d{4})_")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_year(path):
    m = YEAR_REGEX.search(os.path.basename(path))
    return int(m.group(1)) if m else None


# =============================================================================
# Thermal
# =============================================================================

def get_gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def thermal_gate(target=72, limit=79):
    temp = get_gpu_temp()
    if temp == 0:
        time.sleep(0.2)
        return
    if temp <= target:
        return
    if temp > limit:
        tqdm.write(f"\n[THERMAL] {temp}°C — cooling to {target}°C...")
        while get_gpu_temp() > target:
            time.sleep(5)
        tqdm.write(f"[THERMAL] Resumed at {get_gpu_temp()}°C")
        return
    ratio = (temp - target) / max(limit - target, 1)
    time.sleep(0.2 + ratio * 1.5)


# =============================================================================
# Model
# =============================================================================

def load_earthdial(model_path):
    if not HAS_EARTHDIAL:
        raise RuntimeError("earthdial not installed")
    print(f"[EarthDial] Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    print(f"[EarthDial] Ready. GPU: {get_gpu_temp()}°C")
    return model, tokenizer


_tf = {}

def get_transform(sz=448):
    if sz not in _tf:
        try:
            _tf[sz] = build_transform(is_train=False, input_size=sz)
        except Exception:
            _tf[sz] = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
    return _tf[sz]


def _infer(model, tokenizer, pv_cpu, prompt, max_tokens):
    """Single inference call. GPU used only here."""
    dtype = next(model.parameters()).dtype
    pv = pv_cpu.to(dtype=dtype, device="cuda")
    with torch.inference_mode():
        resp = model.chat(
            tokenizer=tokenizer,
            pixel_values=pv,
            question=prompt,
            generation_config=dict(max_new_tokens=max_tokens, do_sample=False),
        )
    del pv
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return resp.strip()


def parse_single_response(text):
    """Parse ### INFORMATIVE / ### GEOMETRIC / ### SEMANTIC sections."""
    result = {"informative": "", "geometric": "", "semantic": ""}

    # Split by ### headers
    parts = re.split(r'###\s*(?:INFORMATIVE|GEOMETRIC|SEMANTIC)\s*\n?', text, flags=re.IGNORECASE)

    # parts[0] = preamble (before first ###), parts[1..] = sections
    sections = [p.strip() for p in parts[1:] if p.strip()]

    if len(sections) >= 3:
        result["informative"] = sections[0]
        result["geometric"] = sections[1]
        result["semantic"] = sections[2]
    elif len(sections) == 2:
        result["informative"] = sections[0]
        result["geometric"] = sections[1]
    elif len(sections) == 1:
        result["informative"] = sections[0]
    else:
        # No ### headers found — try [HEADER] style as backup
        parts2 = re.split(r'\[(?:INFORMATIVE|GEOMETRIC|SEMANTIC)\]', text, flags=re.IGNORECASE)
        sections2 = [p.strip() for p in parts2[1:] if p.strip()]
        if len(sections2) >= 3:
            result["informative"] = sections2[0]
            result["geometric"] = sections2[1]
            result["semantic"] = sections2[2]
        else:
            # Total parse failure — dump everything as informative
            result["informative"] = text.strip()

    return result


def is_good_caption(caps):
    """Quality check: informative must be substantial."""
    info = caps.get("informative", "")
    geo = caps.get("geometric", "")
    sem = caps.get("semantic", "")
    return len(info) >= MIN_INFORMATIVE_LEN and len(geo) > 20 and len(sem) > 20


def generate_captions(model, tokenizer, image, target_temp, max_temp):
    """
    HYBRID: Try single call first. If quality check fails, do 3 separate calls.
    Returns dict with informative, geometric, semantic.
    """
    pv_cpu = get_transform(448)(image).unsqueeze(0)

    # ── Fast path: single call ──
    resp = _infer(model, tokenizer, pv_cpu, SINGLE_PROMPT, max_tokens=1024)
    thermal_gate(target=target_temp, limit=max_temp)

    caps = parse_single_response(resp)

    if is_good_caption(caps):
        del pv_cpu
        return caps

    # ── Slow path: 3 separate calls ──
    result = {}
    for cap_type, cfg in FALLBACK_PROMPTS.items():
        result[cap_type] = _infer(model, tokenizer, pv_cpu, cfg["prompt"], cfg["max_tokens"])
        thermal_gate(target=target_temp, limit=max_temp)

    del pv_cpu
    return result


# =============================================================================
# Patches
# =============================================================================

def extract_patch(tif_path, x, y, ps):
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(window=Window(x, y, ps, ps))
            if arr.shape[0] >= 3:
                rgb = arr[:3]
            elif arr.shape[0] == 1:
                rgb = np.repeat(arr, 3, axis=0)
            else:
                return None
            if rgb.dtype in (np.float32, np.float64):
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                elif rgb.max() <= 255:
                    rgb = rgb.clip(0, 255).astype(np.uint8)
                else:
                    for c in range(3):
                        lo, hi = np.percentile(rgb[c], [2, 98])
                        rgb[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6) * 255, 0, 255)
                    rgb = rgb.astype(np.uint8)
            elif rgb.dtype == np.uint16:
                for c in range(3):
                    lo, hi = np.percentile(rgb[c], [2, 98])
                    rgb[c] = np.clip((rgb[c].astype(float) - lo) / (hi - lo + 1e-6) * 255, 0, 255)
                rgb = rgb.astype(np.uint8)
            return Image.fromarray(np.transpose(rgb, (1, 2, 0)), "RGB")
    except Exception:
        return None


def build_index(root, ps=128, prefix="s2_rgb_"):
    patches = []
    for rn in sorted(os.listdir(root)):
        rd = os.path.join(root, rn)
        if not os.path.isdir(rd):
            continue
        for fn in os.listdir(rd):
            if not fn.lower().endswith(".tif") or not fn.startswith(prefix):
                continue
            yr = extract_year(fn)
            if yr is None:
                continue
            fp = os.path.join(rd, fn)
            try:
                with rasterio.open(fp) as s:
                    w, h = s.width, s.height
            except Exception:
                continue
            for iy in range(h // ps):
                for ix in range(w // ps):
                    patches.append({"region": rn, "year": yr, "rgb_path": fp,
                                    "x": ix * ps, "y": iy * ps})
    print(f"[Index] {len(patches)} patches")
    return patches


def key(r, y, x, yy):
    return f"{r}_{y}_x{x:04d}_y{yy:04d}"


# =============================================================================
# Main
# =============================================================================

def run(root_dir, output_dir, model_path, patch_size=128,
        rgb_prefix="s2_rgb_", resume=True, target_temp=72, max_temp=79):

    os.makedirs(output_dir, exist_ok=True)
    patches = build_index(root_dir, patch_size, rgb_prefix)
    if not patches:
        return

    cf = os.path.join(output_dir, "captions.json")
    done = {}
    if resume and os.path.exists(cf):
        with open(cf) as f:
            for e in json.load(f):
                done[key(e["region"], e["year"], e["x"], e["y"])] = e
        print(f"Done: {len(done)}")

    todo = [p for p in patches if key(p["region"], p["year"], p["x"], p["y"]) not in done]
    print(f"Todo: {len(todo)}")
    if not todo:
        print("All done!")
        return

    print(f"[THERMAL] target={target_temp}°C limit={max_temp}°C now={get_gpu_temp()}°C")
    print(f"[MODE] Hybrid: single-call → quality check → fallback if bad")

    model, tokenizer = load_earthdial(model_path)

    all_caps = list(done.values())
    pd = os.path.join(output_dir, "captions_by_patch")
    fast_count = 0
    slow_count = 0

    for i, p in enumerate(tqdm(todo, desc="EarthDial")):
        img = extract_patch(p["rgb_path"], p["x"], p["y"], patch_size)
        if img is None:
            continue
        try:
            caps = generate_captions(model, tokenizer, img, target_temp, max_temp)
            del img

            # Track which path was used
            if is_good_caption(caps):
                fast_count += 1
            else:
                slow_count += 1

            entry = {"region": p["region"], "year": p["year"],
                     "x": p["x"], "y": p["y"], "rgb_path": p["rgb_path"], **caps}
            all_caps.append(entry)

            rd = os.path.join(pd, p["region"])
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, f"{p['year']}_x{p['x']:04d}_y{p['y']:04d}.json"), "w") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            if (i + 1) % 50 == 0:
                with open(cf, "w") as f:
                    json.dump(all_caps, f, indent=2, ensure_ascii=False)
                total = fast_count + slow_count
                pct = fast_count / total * 100 if total > 0 else 0
                tqdm.write(f"  Saved {len(all_caps)} | GPU:{get_gpu_temp()}°C | fast:{pct:.0f}%")

        except Exception as e:
            tqdm.write(f"  ERR {p['region']}/{p['year']} ({p['x']},{p['y']}): {e}")
            torch.cuda.empty_cache()

    with open(cf, "w") as f:
        json.dump(all_caps, f, indent=2, ensure_ascii=False)
    total = fast_count + slow_count
    print(f"\nDone! {len(all_caps)} patches | fast:{fast_count} slow:{slow_count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", default="akshaydudhane/EarthDial_4B_RGB")
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--rgb_prefix", default="s2_rgb_")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--target_temp", type=int, default=70)
    ap.add_argument("--max_temp", type=int, default=77)
    a = ap.parse_args()
    run(a.root_dir, a.output_dir, a.model_path, a.patch_size,
        a.rgb_prefix, not a.no_resume, a.target_temp, a.max_temp)