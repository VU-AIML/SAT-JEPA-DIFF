# sd_models.py
#
# Stable Diffusion 3.5 Medium + Multi-Caption + IJEPA Conditioning
#
# CAPTION-GUIDED ARCHITECTURE (No Coarse RGB):
# 1. Informative caption (t) - detailed pixel-level description
# 2. Geometric caption (t) - spatial/structural properties
# 3. Semantic caption (t+1) - forecasted high-level understanding
# 4. IJEPA tokens (t+1) - predicted visual embeddings
#
# All conditioning comes from text embeddings and IJEPA tokens.
# No pixel-level reference image is used.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    SD3Transformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Warning] PEFT not installed. Install with: pip install peft")


class TemporalAttentionBlock(nn.Module):
    """
    Temporal Attention for autoregressive consistency.
    During rollout, attends to embeddings from previous time steps.
    """
    def __init__(
        self,
        embed_dim: int = 4096,
        num_heads: int = 8,
        max_history: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_history = max_history

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.post_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(max_history, 1, embed_dim) * 0.02
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.temporal_gate[0].bias, -2.0)
        nn.init.xavier_uniform_(self.post_proj[0].weight, gain=0.1)
        nn.init.zeros_(self.post_proj[0].bias)

    def forward(self, current_tokens, history_tokens=None):
        if history_tokens is None or len(history_tokens) == 0:
            return current_tokens

        B, N, D = current_tokens.shape
        history = history_tokens[:self.max_history]

        history_with_pos = []
        for t_idx, h in enumerate(history):
            h = h.to(dtype=current_tokens.dtype, device=current_tokens.device)
            t_pos = self.temporal_pos_embed[t_idx:t_idx + 1]
            history_with_pos.append(h + t_pos)

        kv = torch.cat(history_with_pos, dim=1)
        q = self.norm_q(current_tokens)
        kv = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.post_proj(attn_out)

        gate_input = torch.cat([current_tokens, attn_out], dim=-1)
        gate = self.temporal_gate(gate_input)

        return current_tokens + gate * attn_out


class MultiCaptionConditioningAdapter(nn.Module):
    """
    Multi-Caption + IJEPA Conditioning Adapter for SD3.5.

    Fuses 4 conditioning signals into SD3.5 cross-attention format:
    1. Informative caption (t) - T5/CLIP encoded, detailed description
    2. Geometric caption (t) - T5/CLIP encoded, spatial properties
    3. Semantic caption (t+1) - T5/CLIP encoded, forecasted semantics
    4. IJEPA tokens (t+1) - predicted visual embeddings

    NO coarse RGB. All structure info comes from captions.

    Output:
        - encoder_hidden_states: (B, M, 4096) for cross-attention
        - pooled_projections: (B, 2048) for global conditioning
    """

    def __init__(
        self,
        ijepa_dim: int = 768,
        text_hidden_dim: int = 4096,
        text_pooled_dim: int = 2048,
        cross_attn_dim: int = 4096,
        pooled_dim: int = 2048,
        hidden_dim: int = 1024,
        temporal_num_heads: int = 8,
        temporal_max_history: int = 4,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()

        self.ijepa_dim = ijepa_dim
        self.cross_attn_dim = cross_attn_dim

        # =================================================================
        # IJEPA Token Projection
        # =================================================================
        self.ijepa_input_norm = nn.LayerNorm(ijepa_dim)
        self.ijepa_token_proj = nn.Sequential(
            nn.Linear(ijepa_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )

        self.max_ijepa_tokens = 1024
        self.ijepa_pos_embed = nn.Parameter(
            torch.randn(1, self.max_ijepa_tokens, ijepa_dim) * 0.02
        )

        self.ijepa_pool_proj = nn.Sequential(
            nn.Linear(ijepa_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pooled_dim),
        )

        # =================================================================
        # Caption Projections (3 separate streams)
        # =================================================================

        # Informative caption: detailed pixel-level -> structure signal
        self.informative_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_dim),
            nn.Linear(text_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )
        self.informative_pool_proj = nn.Linear(text_pooled_dim, pooled_dim)

        # Geometric caption: spatial/structural -> layout signal
        self.geometric_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_dim),
            nn.Linear(text_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )
        self.geometric_pool_proj = nn.Linear(text_pooled_dim, pooled_dim)

        # Semantic caption (forecasted): high-level -> generation guidance
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_dim),
            nn.Linear(text_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )
        self.semantic_pool_proj = nn.Linear(text_pooled_dim, pooled_dim)

        # =================================================================
        # Multi-stream Fusion
        # =================================================================
        self.stream_fusion = nn.MultiheadAttention(
            embed_dim=cross_attn_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.stream_fusion_norm = nn.LayerNorm(cross_attn_dim)

        # Pooled fusion: learned weights for 4 sources
        self.pooled_fusion_weights = nn.Parameter(torch.ones(4) / 4)

        # =================================================================
        # Temporal Attention
        # =================================================================
        self.temporal_attn = TemporalAttentionBlock(
            embed_dim=cross_attn_dim,
            num_heads=temporal_num_heads,
            max_history=temporal_max_history,
            dropout=temporal_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        ijepa_tokens: torch.Tensor,                     # (B, N, ijepa_dim)
        informative_hidden: torch.Tensor = None,         # (B, L, 4096)
        informative_pooled: torch.Tensor = None,         # (B, 2048)
        geometric_hidden: torch.Tensor = None,           # (B, L, 4096)
        geometric_pooled: torch.Tensor = None,           # (B, 2048)
        semantic_hidden: torch.Tensor = None,            # (B, L, 4096)
        semantic_pooled: torch.Tensor = None,            # (B, 2048)
        history_hidden: list = None,
    ):
        """
        Multi-stream conditioning fusion. Pure embeddings, no pixel input.

        Returns:
            encoder_hidden_states: (B, M, 4096)
            pooled_projections: (B, 2048)
        """
        B, N, C = ijepa_tokens.shape
        target_dtype = self.ijepa_token_proj[0].weight.dtype

        # =================================================================
        # 1. IJEPA Stream
        # =================================================================
        tokens = ijepa_tokens.to(dtype=target_dtype)
        tokens = self.ijepa_input_norm(tokens)

        if N <= self.max_ijepa_tokens:
            tokens = tokens + self.ijepa_pos_embed[:, :N, :].to(dtype=target_dtype)
        else:
            pos_interp = F.interpolate(
                self.ijepa_pos_embed.permute(0, 2, 1), size=N,
                mode='linear', align_corners=False
            ).permute(0, 2, 1)
            tokens = tokens + pos_interp.to(dtype=target_dtype)

        ijepa_hidden = self.ijepa_token_proj(tokens)        # (B, N, 4096)
        ijepa_pooled = self.ijepa_pool_proj(tokens.mean(dim=1))  # (B, 2048)

        # =================================================================
        # 2. Caption Streams
        # =================================================================
        all_hidden_streams = [ijepa_hidden]
        all_pooled = [ijepa_pooled]

        if informative_hidden is not None:
            info_h = self.informative_proj(informative_hidden.to(dtype=target_dtype))
            all_hidden_streams.append(info_h)
            if informative_pooled is not None:
                all_pooled.append(self.informative_pool_proj(
                    informative_pooled.to(dtype=target_dtype)
                ))

        if geometric_hidden is not None:
            geo_h = self.geometric_proj(geometric_hidden.to(dtype=target_dtype))
            all_hidden_streams.append(geo_h)
            if geometric_pooled is not None:
                all_pooled.append(self.geometric_pool_proj(
                    geometric_pooled.to(dtype=target_dtype)
                ))

        if semantic_hidden is not None:
            sem_h = self.semantic_proj(semantic_hidden.to(dtype=target_dtype))
            all_hidden_streams.append(sem_h)
            if semantic_pooled is not None:
                all_pooled.append(self.semantic_pool_proj(
                    semantic_pooled.to(dtype=target_dtype)
                ))

        # =================================================================
        # 3. Multi-stream Fusion
        # =================================================================
        if len(all_hidden_streams) > 1:
            kv_tokens = torch.cat(all_hidden_streams, dim=1)
            q_tokens = self.stream_fusion_norm(ijepa_hidden)
            kv_normed = self.stream_fusion_norm(kv_tokens)
            fused, _ = self.stream_fusion(q_tokens, kv_normed, kv_normed)
            encoder_hidden_states = ijepa_hidden + fused
        else:
            encoder_hidden_states = ijepa_hidden

        # =================================================================
        # 4. Pooled Projection Fusion
        # =================================================================
        if len(all_pooled) > 1:
            weights = F.softmax(self.pooled_fusion_weights[:len(all_pooled)], dim=0)
            pooled_stack = torch.stack(all_pooled, dim=0)
            pooled_projections = (weights.view(-1, 1, 1) * pooled_stack).sum(dim=0)
        else:
            pooled_projections = all_pooled[0]

        # =================================================================
        # 5. Temporal Attention
        # =================================================================
        encoder_hidden_states = self.temporal_attn(
            current_tokens=encoder_hidden_states,
            history_tokens=history_hidden,
        )

        return encoder_hidden_states, pooled_projections


# Backward compatibility aliases
IJEPAConditioningAdapter = MultiCaptionConditioningAdapter
IJEPACrossAttentionAdapter = MultiCaptionConditioningAdapter


def encode_prompt(
    prompt, clip_tokenizer, clip_tokenizer_2, t5_tokenizer,
    clip_encoder, clip_encoder_2, t5_encoder, device, dtype, max_length=77,
):
    """Encode text prompt for SD3."""
    clip_inputs = clip_tokenizer(
        prompt, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    clip_inputs_2 = clip_tokenizer_2(
        prompt, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        clip_out = clip_encoder(clip_inputs.input_ids, output_hidden_states=True)
        clip_out_2 = clip_encoder_2(clip_inputs_2.input_ids, output_hidden_states=True)
        pooled = torch.cat([clip_out.text_embeds, clip_out_2.text_embeds], dim=-1)

    t5_inputs = t5_tokenizer(
        prompt, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        t5_hidden = t5_encoder(t5_inputs.input_ids).last_hidden_state

    return pooled.to(dtype), t5_hidden.to(dtype)


def load_sd_model(
    device: torch.device = torch.device("cuda"),
    checkpoint_dir: str = "./sd_finetuned",
    base_model: str = "stabilityai/stable-diffusion-3.5-medium",
    dtype: torch.dtype = torch.float16,
    load_text_encoders: bool = True,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    hf_token: str = None,
    target_emb_dim: int = 768,
    crop_size: int = 128,
):
    """Load SD3.5 with Multi-Caption + IJEPA conditioning (no coarse RGB)."""

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)

    print(f"[SD] Loading {base_model}...")

    # Transformer
    unet = SD3Transformer2DModel.from_pretrained(
        base_model, subfolder="transformer", torch_dtype=dtype, token=hf_token,
    )

    # LoRA
    if use_lora and PEFT_AVAILABLE:
        print(f"[SD] LoRA rank={lora_rank}, alpha={lora_alpha}")
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        lora_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"[SD] LoRA params: {lora_params:,}")
    else:
        unet.requires_grad_(False)

    # VAE
    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=dtype, token=hf_token,
    )
    vae.requires_grad_(False)

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model, subfolder="scheduler", token=hf_token,
    )

    # Text embeddings (fallback prompt when captions not available)
    prompt_embeds = None
    pooled_prompt_embeds = None
    caption_encoder = None

    if load_text_encoders:
        print("[SD] Loading text encoders for caption conditioning...")
        try:
            clip_tok = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", token=hf_token)
            clip_tok_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", token=hf_token)
            clip_enc = CLIPTextModelWithProjection.from_pretrained(
                base_model, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
            ).to(device)
            clip_enc_2 = CLIPTextModelWithProjection.from_pretrained(
                base_model, subfolder="text_encoder_2", torch_dtype=dtype, token=hf_token
            ).to(device)
            t5_tok = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_3", token=hf_token)
            t5_enc = T5EncoderModel.from_pretrained(
                base_model, subfolder="text_encoder_3", torch_dtype=dtype, token=hf_token
            ).to(device)

            FALLBACK_PROMPT = (
                "High-resolution Sentinel-2 satellite image, "
                "multispectral earth observation, natural colors RGB composite"
            )
            pooled_prompt_embeds, prompt_embeds = encode_prompt(
                FALLBACK_PROMPT, clip_tok, clip_tok_2, t5_tok,
                clip_enc, clip_enc_2, t5_enc, device, dtype,
            )
            print(f"[SD] Fallback prompt encoded: {prompt_embeds.shape}")

            caption_encoder = {
                "clip_tokenizer": clip_tok,
                "clip_tokenizer_2": clip_tok_2,
                "t5_tokenizer": t5_tok,
                "clip_encoder": clip_enc,
                "clip_encoder_2": clip_enc_2,
                "t5_encoder": t5_enc,
            }
            print("[SD] Text encoders stored for caption conditioning")

        except Exception as e:
            print(f"[SD] Text encoder error: {e}")

    if prompt_embeds is None:
        pooled_prompt_embeds = torch.zeros(1, 2048, device=device, dtype=dtype)
        prompt_embeds = torch.zeros(1, 77, 4096, device=device, dtype=dtype)

    # Multi-Caption conditioning adapter (NO coarse RGB)
    cross_attn_dim = getattr(unet.config, "joint_attention_dim", 4096)
    pooled_dim = getattr(unet.config, "pooled_projection_dim", 2048)

    print(f"[SD] Initializing Multi-Caption + IJEPA Adapter (no coarse RGB):")
    print(f"     - IJEPA input dim: {target_emb_dim}")
    print(f"     - Caption streams: informative, geometric, semantic(forecasted)")

    cond_adapter = MultiCaptionConditioningAdapter(
        ijepa_dim=target_emb_dim,
        text_hidden_dim=4096,
        text_pooled_dim=2048,
        cross_attn_dim=cross_attn_dim,
        pooled_dim=pooled_dim,
    )
    cond_adapter.to(device)

    adapter_params = sum(p.numel() for p in cond_adapter.parameters())
    print(f"[SD] Multi-Caption adapter: {adapter_params:,} params")

    unet.to(device)
    vae.to(device)

    trainable_params = list(cond_adapter.parameters())
    if use_lora and PEFT_AVAILABLE:
        trainable_params.extend([p for p in unet.parameters() if p.requires_grad])

    print(f"[SD] Total trainable: {sum(p.numel() for p in trainable_params):,}")

    return {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": scheduler,
        "cond_adapter": cond_adapter,
        "trainable_params": trainable_params,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "caption_encoder": caption_encoder,
        "use_lora": use_lora and PEFT_AVAILABLE,
    }


def encode_caption_batch(caption_encoder, texts, device, dtype, max_length=77):
    """
    Encode a batch of caption texts using the stored text encoders.

    Returns:
        hidden: (B, L, 4096) - T5 hidden states
        pooled: (B, 2048) - CLIP pooled embeddings
    """
    if caption_encoder is None:
        return None, None

    # Skip if all captions are empty
    if all(t == "" for t in texts):
        return None, None

    clip_tok = caption_encoder["clip_tokenizer"]
    clip_tok_2 = caption_encoder["clip_tokenizer_2"]
    t5_tok = caption_encoder["t5_tokenizer"]
    clip_enc = caption_encoder["clip_encoder"]
    clip_enc_2 = caption_encoder["clip_encoder_2"]
    t5_enc = caption_encoder["t5_encoder"]

    clip_inputs = clip_tok(
        texts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    clip_inputs_2 = clip_tok_2(
        texts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        clip_out = clip_enc(clip_inputs.input_ids, output_hidden_states=True)
        clip_out_2 = clip_enc_2(clip_inputs_2.input_ids, output_hidden_states=True)
        pooled = torch.cat([clip_out.text_embeds, clip_out_2.text_embeds], dim=-1)

    t5_inputs = t5_tok(
        texts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        t5_hidden = t5_enc(t5_inputs.input_ids).last_hidden_state

    return t5_hidden.to(dtype), pooled.to(dtype)


def save_full_checkpoint(
    save_path, encoder, predictor, proj_head, target_encoder,
    sd_state, optimizer, scaler, epoch, best_val_loss, config,
    caption_forecaster=None,
):
    """Save checkpoint including caption forecaster."""
    ckpt = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "proj_head": proj_head.state_dict(),
        "cond_adapter": sd_state["cond_adapter"].state_dict(),
        "prompt_embeds": sd_state["prompt_embeds"].cpu(),
        "pooled_prompt_embeds": sd_state["pooled_prompt_embeds"].cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
        "use_lora": sd_state.get("use_lora", False),
    }

    if caption_forecaster is not None:
        ckpt["caption_forecaster"] = caption_forecaster.state_dict()

    if sd_state.get("use_lora"):
        lora_dict = {n: p.cpu() for n, p in sd_state["unet"].named_parameters()
                     if "lora" in n.lower() and p.requires_grad}
        ckpt["lora_state_dict"] = lora_dict

    torch.save(ckpt, save_path)
    print(f"[Checkpoint] Saved: {save_path}")


def load_full_checkpoint(
    load_path, encoder, predictor, proj_head, target_encoder,
    sd_state, optimizer=None, scaler=None, device=torch.device("cuda"),
    caption_forecaster=None,
):
    """Load checkpoint including caption forecaster."""
    print(f"[Checkpoint] Loading from: {load_path}")
    ckpt = torch.load(load_path, map_location=device)

    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    proj_head.load_state_dict(ckpt["proj_head"])

    unet_dtype = next(sd_state["unet"].parameters()).dtype

    try:
        sd_state["cond_adapter"].load_state_dict(ckpt["cond_adapter"])
    except RuntimeError as e:
        print(f"[Warning] Adapter size mismatch, re-initializing. Details: {e}")

    sd_state["cond_adapter"].to(dtype=unet_dtype)

    if caption_forecaster is not None and "caption_forecaster" in ckpt:
        caption_forecaster.load_state_dict(ckpt["caption_forecaster"])
        print("[Checkpoint] Caption forecaster restored.")

    if "prompt_embeds" in ckpt:
        sd_state["prompt_embeds"] = ckpt["prompt_embeds"].to(device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = ckpt["pooled_prompt_embeds"].to(device=device, dtype=unet_dtype)

    if ckpt.get("use_lora") and "lora_state_dict" in ckpt:
        sd_state["unet"].load_state_dict(ckpt["lora_state_dict"], strict=False)

    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[Warning] Could not restore optimizer: {e}")

    if scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except:
            pass

    epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_val_loss", float("inf"))

    print(f"[Checkpoint] Epoch {epoch} restored.")
    return epoch, best_loss