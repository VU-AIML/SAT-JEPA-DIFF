"""
Precompute Caption Embeddings
==============================
Encodes all captions (informative, geometric, semantic) into T5 hidden states
and CLIP pooled embeddings ONCE, saves as .pt files.

Training then loads these tensors directly — no text encoder needed at all.
This eliminates ~40% of training GPU load.

Output structure:
    output_dir/
        embeddings/
            {region}_{year}_x{X}_y{Y}.pt
                → dict with keys:
                    informative_hidden: (77, 4096)
                    informative_pooled: (2048,)
                    geometric_hidden: (77, 4096)
                    geometric_pooled: (2048,)
                    semantic_hidden: (77, 4096)
                    semantic_pooled: (2048,)

Usage:
    python precompute_caption_embeddings.py \
        --caption_dir /path/to/captions \
        --output_dir /path/to/captions \
        --hf_token hf_xxx
"""

import os
import json
import argparse
import time
import subprocess
import torch
from tqdm import tqdm
from transformers import (
    CLIPTextModelWithProjection, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
)


def get_gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def load_text_encoders(base_model, device, dtype, hf_token):
    """Load SD3.5 text encoders."""
    print("Loading CLIP tokenizer 1...")
    clip_tok = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", token=hf_token)
    print("Loading CLIP tokenizer 2...")
    clip_tok_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", token=hf_token)
    print("Loading T5 tokenizer...")
    t5_tok = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_3", token=hf_token)

    print("Loading CLIP encoder 1...")
    clip_enc = CLIPTextModelWithProjection.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
    ).to(device).eval()
    print("Loading CLIP encoder 2...")
    clip_enc_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model, subfolder="text_encoder_2", torch_dtype=dtype, token=hf_token
    ).to(device).eval()
    print("Loading T5 encoder...")
    t5_enc = T5EncoderModel.from_pretrained(
        base_model, subfolder="text_encoder_3", torch_dtype=dtype, token=hf_token
    ).to(device).eval()

    return {
        "clip_tokenizer": clip_tok,
        "clip_tokenizer_2": clip_tok_2,
        "t5_tokenizer": t5_tok,
        "clip_encoder": clip_enc,
        "clip_encoder_2": clip_enc_2,
        "t5_encoder": t5_enc,
    }


@torch.no_grad()
def encode_text(encoders, text, device, dtype, max_length=77):
    """Encode a single caption string → (hidden, pooled)."""
    if not text or not text.strip():
        return None, None

    clip_tok = encoders["clip_tokenizer"]
    clip_tok_2 = encoders["clip_tokenizer_2"]
    t5_tok = encoders["t5_tokenizer"]
    clip_enc = encoders["clip_encoder"]
    clip_enc_2 = encoders["clip_encoder_2"]
    t5_enc = encoders["t5_encoder"]

    # CLIP pooled
    c1 = clip_tok(text, padding="max_length", max_length=max_length,
                  truncation=True, return_tensors="pt").to(device)
    c2 = clip_tok_2(text, padding="max_length", max_length=max_length,
                    truncation=True, return_tensors="pt").to(device)
    out1 = clip_enc(c1.input_ids, output_hidden_states=True)
    out2 = clip_enc_2(c2.input_ids, output_hidden_states=True)
    pooled = torch.cat([out1.text_embeds, out2.text_embeds], dim=-1)  # (1, 2048)

    # T5 hidden
    t5_in = t5_tok(text, padding="max_length", max_length=max_length,
                   truncation=True, return_tensors="pt").to(device)
    hidden = t5_enc(t5_in.input_ids).last_hidden_state  # (1, 77, 4096)

    return hidden.squeeze(0).cpu().to(dtype), pooled.squeeze(0).cpu().to(dtype)


def main():
    parser = argparse.ArgumentParser(description="Precompute caption embeddings")
    parser.add_argument("--caption_dir", required=True, help="Dir with captions.json")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: caption_dir)")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--max_length", type=int, default=77)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.caption_dir

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load captions
    caption_file = os.path.join(args.caption_dir, "captions.json")
    if not os.path.exists(caption_file):
        print(f"No captions.json found in {args.caption_dir}")
        return

    with open(caption_file) as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    # Output dir
    emb_dir = os.path.join(args.output_dir, "caption_embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # Check what's already done (resume support)
    done = set()
    for f_name in os.listdir(emb_dir):
        if f_name.endswith(".pt"):
            done.add(f_name[:-3])  # Remove .pt

    remaining = []
    for entry in captions:
        key = f"{entry['region']}_{entry['year']}_x{entry['x']:04d}_y{entry['y']:04d}"
        if key not in done:
            remaining.append((key, entry))

    print(f"Already done: {len(done)}, Remaining: {len(remaining)}")
    if not remaining:
        print("All embeddings precomputed!")
        return

    # Load text encoders
    encoders = load_text_encoders(args.base_model, device, dtype, args.hf_token)
    print(f"Text encoders loaded on {device}")

    # Process
    for idx, (key, entry) in enumerate(tqdm(remaining, desc="Encoding captions")):
        # Thermal check every 10 patches
        if idx % 10 == 0:
            temp = get_gpu_temp()
            if temp > 78:
                tqdm.write(f"\n[THERMAL] {temp}°C — cooling to 70°C...")
                while get_gpu_temp() > 70:
                    time.sleep(5)
                tqdm.write(f"[THERMAL] Resumed at {get_gpu_temp()}°C")

        result = {}

        for cap_type in ["informative", "geometric", "semantic"]:
            text = entry.get(cap_type, "")
            hidden, pooled = encode_text(encoders, text, device, dtype, args.max_length)

            if hidden is not None:
                result[f"{cap_type}_hidden"] = hidden   # (77, 4096)
                result[f"{cap_type}_pooled"] = pooled   # (2048,)
            else:
                # Empty caption → zero tensors
                result[f"{cap_type}_hidden"] = torch.zeros(args.max_length, 4096, dtype=dtype)
                result[f"{cap_type}_pooled"] = torch.zeros(2048, dtype=dtype)

        # Save as .pt
        save_path = os.path.join(emb_dir, f"{key}.pt")
        torch.save(result, save_path)

    print(f"\nDone! Embeddings saved to {emb_dir}/")
    print(f"Total: {len(done) + len(remaining)} patch embeddings")

    # Cleanup GPU
    del encoders
    torch.cuda.empty_cache()
    print("Text encoders freed from GPU")


if __name__ == "__main__":
    main()