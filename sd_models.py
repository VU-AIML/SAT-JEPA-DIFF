# sd_models.py
#
# Stable Diffusion 3.5 Medium + IJEPA Cross-Attention Conditioning
# + Coarse RGB Conditioning (32x32 downsampled, NO VAE encoding)
#
# KEY CHANGES:
# 1. t image 32x32'ye downsample edilip doğrudan RGB olarak kullanılıyor
# 2. VAE encoding YOK - daha fazla bilgi korunuyor (3072 değer vs 256)
# 3. IJEPA tokens t+1 embedding'i temsil ediyor (fine detail signal)
# 4. SD coarse structure'ı RGB'den, fine detail'leri IJEPA'dan alıyor

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


SENTINEL2_PROMPT = (
    "High-resolution Sentinel-2 satellite image, "
    "multispectral earth observation, "
    "natural colors RGB composite, "
    "10m ground resolution, "
    "clear atmospheric conditions, "
    "detailed land surface features"
)


# =============================================================================
# Constants for Coarse RGB Conditioning
# =============================================================================
COARSE_SIZE = 32  # Downsample reference to 32x32 RGB (no VAE)


class TemporalAttentionBlock(nn.Module):
    """
    Temporal Attention for autoregressive consistency.
    
    During rollout, this module attends to embeddings from previous time steps,
    allowing the model to maintain temporal coherence and reduce error accumulation.
    
    Mechanism:
    - Current frame tokens are Queries
    - History tokens (from previous frames) are Keys/Values
    - Output is a residual blend: current + gate * temporal_context
    
    This prevents the "drift" problem where recursive generation degrades over time.
    """
    def __init__(
        self,
        embed_dim: int = 4096,
        num_heads: int = 8,
        max_history: int = 4,    # Max number of past frames to attend to
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_history = max_history
        
        # Multi-head cross-attention: current tokens attend to history tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Pre-norm for query and key/value
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        
        # Post-attention projection + norm
        self.post_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Learnable temporal position embeddings for history frames
        # Frame 0 = most recent history, frame max_history-1 = oldest
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(max_history, 1, embed_dim) * 0.02
        )
        
        # Adaptive gate: learns when to rely on temporal context vs current frame
        # Initialized near 0 so model starts by ignoring history (safe cold start)
        self.temporal_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize gate bias to -2 so sigmoid outputs ~0.12
        # Model starts mostly ignoring history, learns to use it
        nn.init.constant_(self.temporal_gate[0].bias, -2.0)
        nn.init.xavier_uniform_(self.post_proj[0].weight, gain=0.1)
        nn.init.zeros_(self.post_proj[0].bias)
    
    def forward(
        self, 
        current_tokens: torch.Tensor,       # (B, N, D) current frame embeddings
        history_tokens: list = None,         # List of (B, N, D) from previous frames
    ) -> torch.Tensor:
        """
        Args:
            current_tokens: (B, N, D) - Current frame's cross-attention embeddings
            history_tokens: List of (B, N, D) tensors from previous frames (most recent first)
                           If None or empty, returns current_tokens unchanged.
        Returns:
            (B, N, D) - Temporally-enhanced embeddings
        """
        if history_tokens is None or len(history_tokens) == 0:
            return current_tokens
        
        B, N, D = current_tokens.shape
        
        # Truncate history to max_history
        history = history_tokens[:self.max_history]
        num_history = len(history)
        
        # Add temporal position embeddings to each history frame
        history_with_pos = []
        for t_idx, h in enumerate(history):
            # Ensure same device/dtype
            h = h.to(dtype=current_tokens.dtype, device=current_tokens.device)
            # Add temporal position embedding (broadcast over B and N)
            t_pos = self.temporal_pos_embed[t_idx:t_idx+1]  # (1, 1, D)
            history_with_pos.append(h + t_pos)
        
        # Concatenate all history: (B, num_history * N, D)
        kv = torch.cat(history_with_pos, dim=1)
        
        # Normalize
        q = self.norm_q(current_tokens)
        kv = self.norm_kv(kv)
        
        # Cross-attention: current attends to history
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.post_proj(attn_out)
        
        # Gated residual: gate decides how much temporal context to inject
        gate_input = torch.cat([current_tokens, attn_out], dim=-1)
        gate = self.temporal_gate(gate_input)  # (B, N, D), values in [0, 1]
        
        # Blend: output = current + gate * temporal_context
        output = current_tokens + gate * attn_out
        
        return output


class IJEPAConditioningAdapter(nn.Module):
    """
    Projects IJEPA tokens + Coarse RGB to SD3 cross-attention format.
    
    COARSE RGB APPROACH (No VAE):
    - 32x32 RGB image = 1024 pixels × 3 channels = 3072 values
    - Much more information than 4x4 latent (256 values)
    - Converted to patch tokens for cross-attention
    
    TEMPORAL ATTENTION (NEW):
    - Attends to cross-attention embeddings from previous time steps
    - Reduces error accumulation during autoregressive rollout
    - Gate starts near 0 (safe for single-frame training), learns to use history
    
    Input: 
        - ijepa_tokens: (B, N, in_dim) - IJEPA predicted embeddings
        - ref_rgb: (B, 3, 32, 32) - Coarse RGB image (optional)
        - history_hidden: List of (B, M, 4096) from previous frames (optional)
    Output:
        - encoder_hidden_states: (B, N, 4096) for cross-attention
        - pooled_projections: (B, 2048) for global conditioning
    """
    def __init__(
        self, 
        in_dim: int = 768,  # IJEPA raw embedding dim
        cross_attn_dim: int = 4096,
        pooled_dim: int = 2048,
        hidden_dim: int = 1024,
        coarse_size: int = COARSE_SIZE,
        coarse_patch_size: int = 4,  # 32/4 = 8x8 = 64 tokens from coarse RGB
        # Temporal attention params
        temporal_num_heads: int = 8,
        temporal_max_history: int = 4,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.cross_attn_dim = cross_attn_dim
        self.coarse_size = coarse_size
        self.coarse_patch_size = coarse_patch_size
        
        # Number of coarse RGB tokens: (32/4)^2 = 64
        self.num_coarse_tokens = (coarse_size // coarse_patch_size) ** 2
        # Each patch has 4x4x3 = 48 values
        self.coarse_patch_dim = coarse_patch_size * coarse_patch_size * 3
        
        # Input Normalization for IJEPA
        self.input_norm = nn.LayerNorm(in_dim)
        
        # IJEPA Token projection: in_dim -> 4096
        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )
        
        # Pooled projection: mean(tokens) -> 2048
        self.pool_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pooled_dim),
        )
        
        # Learnable position embeddings for IJEPA tokens
        # Support up to 1024 tokens (for 256x256 image with patch_size=8)
        self.max_ijepa_tokens = 1024
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_ijepa_tokens, in_dim) * 0.02)
        
        # =======================================================================
        # Coarse RGB projection (no VAE!)
        # 32x32 RGB -> 8x8 patches (4x4 each) -> 64 tokens of dim 48 -> project to 4096
        # =======================================================================
        self.coarse_rgb_proj = nn.Sequential(
            nn.Linear(self.coarse_patch_dim, hidden_dim),  # 48 -> 1024
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim),  # 1024 -> 4096
            nn.LayerNorm(cross_attn_dim),
        )
        
        # Position embeddings for coarse RGB tokens
        self.coarse_pos_embed = nn.Parameter(
            torch.randn(1, self.num_coarse_tokens, self.coarse_patch_dim) * 0.02
        )
        
        # Fusion gate: learns to balance IJEPA vs coarse RGB signal
        self.fusion_gate = nn.Sequential(
            nn.Linear(cross_attn_dim * 2, cross_attn_dim),
            nn.Sigmoid(),
        )
        
        # =======================================================================
        # NEW: Temporal Attention - attends to history of previous frame embeddings
        # Reduces error accumulation during autoregressive rollout
        # =======================================================================
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
        
        # Bias init: start gate near 0.5 (balanced fusion)
        nn.init.constant_(self.fusion_gate[0].bias, 0.0)
    
    def rgb_to_patches(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert 32x32 RGB to patch tokens.
        
        Args:
            rgb: (B, 3, 32, 32) in [0, 1] range
        Returns:
            (B, 64, 48) patch tokens
        """
        B, C, H, W = rgb.shape
        pH = pW = self.coarse_patch_size  # 4
        nH, nW = H // pH, W // pH  # 8, 8
        
        # Reshape to patches: (B, 3, 8, 4, 8, 4) -> (B, 8, 8, 4, 4, 3) -> (B, 64, 48)
        patches = rgb.view(B, C, nH, pH, nW, pW)
        patches = patches.permute(0, 2, 4, 3, 5, 1)  # (B, nH, nW, pH, pW, C)
        patches = patches.reshape(B, nH * nW, pH * pW * C)  # (B, 64, 48)
        
        return patches
    
    def forward(
        self, 
        ijepa_tokens: torch.Tensor,
        ref_rgb: torch.Tensor = None,  # Changed from ref_latents to ref_rgb
        history_hidden: list = None,   # NEW: List of (B, M, 4096) from previous frames
    ):
        """
        Args:
            ijepa_tokens: (B, N, in_dim) - Predicted t+1 embeddings from IJEPA
            ref_rgb: (B, 3, 32, 32) - Coarse RGB image (already downsampled)
            history_hidden: List of (B, M, 4096) from previous time steps (most recent first).
                           Used by TemporalAttentionBlock for autoregressive consistency.
        Returns:
            encoder_hidden_states: (B, M, 4096) - M depends on fusion
            pooled_projections: (B, 2048)
        """
        B, N, C = ijepa_tokens.shape
        target_dtype = self.token_proj[0].weight.dtype
        tokens = ijepa_tokens.to(dtype=target_dtype)
        
        # Apply Input Normalization
        tokens = self.input_norm(tokens)
        
        # Add positional embeddings (with interpolation if needed)
        if N <= self.max_ijepa_tokens:
            tokens = tokens + self.pos_embed[:, :N, :].to(dtype=target_dtype)
        else:
            pos_embed_interp = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=N,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            tokens = tokens + pos_embed_interp.to(dtype=target_dtype)
        
        # IJEPA cross-attention embeddings (fine detail signal)
        ijepa_hidden = self.token_proj(tokens)  # (B, N, 4096)
        
        # Pooled projection
        pooled = tokens.mean(dim=1)
        pooled_projections = self.pool_proj(pooled)  # (B, 2048)
        
        # Process coarse RGB if provided
        if ref_rgb is not None:
            # Convert 32x32 RGB to patch tokens
            ref_rgb = ref_rgb.to(dtype=target_dtype)
            coarse_patches = self.rgb_to_patches(ref_rgb)  # (B, 64, 48)
            
            # Add positional embeddings
            coarse_patches = coarse_patches + self.coarse_pos_embed.to(dtype=target_dtype)
            
            # Project to cross-attention dimension
            coarse_hidden = self.coarse_rgb_proj(coarse_patches)  # (B, 64, 4096)
            
            N_coarse = coarse_hidden.shape[1]  # 64
            
            # Align token counts by interpolating IJEPA to match coarse
            if N != N_coarse:
                ijepa_interp = F.interpolate(
                    ijepa_hidden.permute(0, 2, 1),  # (B, 4096, N)
                    size=N_coarse,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)  # (B, 64, 4096)
            else:
                ijepa_interp = ijepa_hidden
            
            # Compute fusion gate
            combined = torch.cat([coarse_hidden, ijepa_interp], dim=-1)  # (B, 64, 8192)
            gate = self.fusion_gate(combined)  # (B, 64, 4096), values in [0, 1]
            
            # Blend: gate * ijepa + (1-gate) * coarse
            # High gate = trust IJEPA more (for fine details)
            # Low gate = trust coarse RGB more (for structure)
            encoder_hidden_states = gate * ijepa_interp + (1 - gate) * coarse_hidden
        else:
            encoder_hidden_states = ijepa_hidden
        
        # =======================================================================
        # NEW: Temporal Attention - attend to previous frames' embeddings
        # This is the key mechanism for autoregressive consistency.
        # During training, history_hidden can be from the same batch (simulated)
        # or None (standard single-step training).
        # During rollout, history_hidden accumulates real previous predictions.
        # =======================================================================
        encoder_hidden_states = self.temporal_attn(
            current_tokens=encoder_hidden_states,
            history_tokens=history_hidden,
        )
        
        return encoder_hidden_states, pooled_projections


# Keep old name for backward compatibility
IJEPACrossAttentionAdapter = IJEPAConditioningAdapter


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
    target_emb_dim: int = 768,  # IJEPA raw embedding dim (not projected)
    crop_size: int = 128,
    coarse_size: int = COARSE_SIZE,
):
    """Load SD3.5 with Coarse RGB + IJEPA conditioning."""
    
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        print("[Warning] No Hugging Face token found. Gated models may fail to load.")
    
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
    
    # Text embeddings
    prompt_embeds = None
    pooled_prompt_embeds = None
    
    if load_text_encoders:
        print("[SD] Loading text encoders...")
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
            
            pooled_prompt_embeds, prompt_embeds = encode_prompt(
                SENTINEL2_PROMPT, clip_tok, clip_tok_2, t5_tok,
                clip_enc, clip_enc_2, t5_enc, device, dtype,
            )
            print(f"[SD] Prompt encoded: {prompt_embeds.shape}, {pooled_prompt_embeds.shape}")
            
            del clip_enc, clip_enc_2, t5_enc, clip_tok, clip_tok_2, t5_tok
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[SD] Text encoder error: {e}")
    
    if prompt_embeds is None:
        pooled_prompt_embeds = torch.zeros(1, 2048, device=device, dtype=dtype)
        prompt_embeds = torch.zeros(1, 77, 4096, device=device, dtype=dtype)
    
    # Cross-attention adapter with Coarse RGB conditioning
    cross_attn_dim = getattr(unet.config, "joint_attention_dim", 4096)
    pooled_dim = getattr(unet.config, "pooled_projection_dim", 2048)
    
    print(f"[SD] Initializing IJEPA + Coarse RGB Adapter:")
    print(f"     - IJEPA input dim: {target_emb_dim}")
    print(f"     - Coarse RGB size: {coarse_size}x{coarse_size}")
    print(f"     - Coarse tokens: {(coarse_size // 4) ** 2}")
    
    cond_adapter = IJEPAConditioningAdapter(
        in_dim=target_emb_dim,
        cross_attn_dim=cross_attn_dim, 
        pooled_dim=pooled_dim,
        coarse_size=coarse_size,
    )
    cond_adapter.to(device)
    
    adapter_params = sum(p.numel() for p in cond_adapter.parameters())
    print(f"[SD] Conditioning adapter: {adapter_params:,} params")
    
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
        "use_lora": use_lora and PEFT_AVAILABLE,
    }


def save_full_checkpoint(
    save_path, encoder, predictor, proj_head, target_encoder,
    sd_state, optimizer, scaler, epoch, best_val_loss, config,
):
    """Save checkpoint."""
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
    
    if sd_state.get("use_lora"):
        lora_dict = {n: p.cpu() for n, p in sd_state["unet"].named_parameters() 
                     if "lora" in n.lower() and p.requires_grad}
        ckpt["lora_state_dict"] = lora_dict
    
    torch.save(ckpt, save_path)
    print(f"[Checkpoint] Saved: {save_path}")


def load_full_checkpoint(
    load_path, encoder, predictor, proj_head, target_encoder,
    sd_state, optimizer=None, scaler=None, device=torch.device("cuda"),
):
    """Load checkpoint."""
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
        print(f"[Warning] Adapter size mismatch. Re-initializing adapter weights.")
        print(f"Details: {e}")
    
    sd_state["cond_adapter"].to(dtype=unet_dtype)
    
    if "prompt_embeds" in ckpt:
        sd_state["prompt_embeds"] = ckpt["prompt_embeds"].to(device=device, dtype=unet_dtype)
        sd_state["pooled_prompt_embeds"] = ckpt["pooled_prompt_embeds"].to(device=device, dtype=unet_dtype)
    
    if ckpt.get("use_lora") and "lora_state_dict" in ckpt:
        print("[Checkpoint] Applying LoRA weights to UNet...")
        sd_state["unet"].load_state_dict(ckpt["lora_state_dict"], strict=False)
    
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[Checkpoint] Optimizer state restored.")
        except Exception as e:
            print(f"[Warning] Could not restore optimizer state: {e}")
            
    if scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except:
            pass
    
    epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_val_loss", float("inf"))
    
    print(f"[Checkpoint] Success: Epoch {epoch} restored.")
    return epoch, best_loss