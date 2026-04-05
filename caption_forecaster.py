"""
Caption Forecaster Module
=========================
Predicts the semantic caption embedding at t+1 given:
- IJEPA predicted tokens (visual representation of t+1)
- Semantic caption embedding at time t

Architecture:
- Encodes semantic caption text using a frozen CLIP/T5 text encoder
- Cross-attention between IJEPA tokens and caption embedding
- MLP to predict the semantic caption embedding for t+1

The forecasted caption is used to condition the diffusion model alongside
the informative and geometric captions from time t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionForecaster(nn.Module):
    """
    Forecasts semantic caption embedding at t+1.
    
    Input:
        - ijepa_tokens: (B, N, D_ijepa) - IJEPA predicted t+1 visual tokens
        - semantic_embed_t: (B, L, D_text) - Encoded semantic caption at time t
    Output:
        - semantic_embed_tp1: (B, L, D_text) - Predicted semantic caption at t+1
    
    This uses cross-attention: IJEPA tokens attend to the current semantic caption,
    then a residual MLP refines the prediction. The intuition is that visual change
    information from IJEPA guides what aspects of the semantic description will change.
    """
    
    def __init__(
        self,
        ijepa_dim: int = 768,       # IJEPA embedding dimension
        text_dim: int = 4096,        # Text encoder output dim (T5/CLIP)
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_text_len: int = 77,
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Project IJEPA tokens to shared hidden dim
        self.ijepa_proj = nn.Sequential(
            nn.LayerNorm(ijepa_dim),
            nn.Linear(ijepa_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Project text embeddings to shared hidden dim
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Cross-attention layers: text queries attend to IJEPA keys/values
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Temporal delta predictor: predicts the *change* in caption embedding
        # This is more stable than predicting the absolute embedding
        self.delta_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Project back to text embedding space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_dim),
        )
        
        # Learnable gate: controls how much of the delta to apply
        # Initialized small so model starts with identity (caption_t ≈ caption_t+1)
        self.delta_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Gate starts near 0.1 (mostly identity at start)
        nn.init.constant_(self.delta_gate[0].bias, -2.0)
    
    def forward(
        self,
        ijepa_tokens: torch.Tensor,     # (B, N, D_ijepa)
        semantic_embed_t: torch.Tensor,  # (B, L, D_text)
    ) -> torch.Tensor:
        """
        Predict semantic caption embedding at t+1.
        
        Returns:
            (B, L, D_text) - Predicted text embedding for t+1
        """
        # Project to hidden dim
        visual_hidden = self.ijepa_proj(ijepa_tokens)      # (B, N, hidden)
        text_hidden = self.text_proj(semantic_embed_t)      # (B, L, hidden)
        
        # Cross-attention: text attends to visual changes
        x = text_hidden
        for layer in self.cross_attn_layers:
            x = layer(x, visual_hidden)  # (B, L, hidden)
        
        # Predict delta (change from t to t+1)
        delta = self.delta_predictor(x)  # (B, L, hidden)
        
        # Gated residual: output = text_t + gate * delta
        gate_input = torch.cat([text_hidden, delta], dim=-1)
        gate = self.delta_gate(gate_input)
        
        refined = text_hidden + gate * delta  # (B, L, hidden)
        
        # Project back to text embedding space
        output = self.output_proj(refined)  # (B, L, D_text)
        
        return output


class CaptionEncoder(nn.Module):
    """
    Encodes text captions into embeddings using a frozen text encoder.
    
    This wraps CLIP + T5 text encoders (same as SD3.5) to encode
    our 3 caption types into conditioning embeddings.
    """
    
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-3.5-medium",
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        hf_token: str = None,
    ):
        super().__init__()
        
        from transformers import (
            CLIPTextModelWithProjection, CLIPTokenizer,
            T5EncoderModel, T5TokenizerFast,
        )
        
        self.device = device
        self.dtype = dtype
        
        # Load tokenizers
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer", token=hf_token
        )
        self.clip_tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer_2", token=hf_token
        )
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model, subfolder="tokenizer_3", token=hf_token
        )
        
        # Load text encoders (frozen)
        self.clip_encoder = CLIPTextModelWithProjection.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
        ).to(device).eval()
        self.clip_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_model, subfolder="text_encoder_2", torch_dtype=dtype, token=hf_token
        ).to(device).eval()
        self.t5_encoder = T5EncoderModel.from_pretrained(
            base_model, subfolder="text_encoder_3", torch_dtype=dtype, token=hf_token
        ).to(device).eval()
        
        # Freeze all
        for enc in [self.clip_encoder, self.clip_encoder_2, self.t5_encoder]:
            for p in enc.parameters():
                p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, text: str, max_length: int = 77):
        """
        Encode a single caption text.
        
        Returns:
            hidden_states: (1, L, 4096) - T5 hidden states
            pooled: (1, 2048) - CLIP pooled embeddings
        """
        # CLIP encoding
        clip_inputs = self.clip_tokenizer(
            text, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        clip_inputs_2 = self.clip_tokenizer_2(
            text, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        
        clip_out = self.clip_encoder(clip_inputs.input_ids, output_hidden_states=True)
        clip_out_2 = self.clip_encoder_2(clip_inputs_2.input_ids, output_hidden_states=True)
        pooled = torch.cat([clip_out.text_embeds, clip_out_2.text_embeds], dim=-1)
        
        # T5 encoding
        t5_inputs = self.t5_tokenizer(
            text, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        t5_hidden = self.t5_encoder(t5_inputs.input_ids).last_hidden_state
        
        return t5_hidden.to(self.dtype), pooled.to(self.dtype)
    
    @torch.no_grad()
    def encode_batch(self, texts: list, max_length: int = 77):
        """
        Encode a batch of caption texts.
        
        Returns:
            hidden_states: (B, L, 4096)
            pooled: (B, 2048)
        """
        # CLIP
        clip_inputs = self.clip_tokenizer(
            texts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        clip_inputs_2 = self.clip_tokenizer_2(
            texts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        
        clip_out = self.clip_encoder(clip_inputs.input_ids, output_hidden_states=True)
        clip_out_2 = self.clip_encoder_2(clip_inputs_2.input_ids, output_hidden_states=True)
        pooled = torch.cat([clip_out.text_embeds, clip_out_2.text_embeds], dim=-1)
        
        # T5
        t5_inputs = self.t5_tokenizer(
            texts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        t5_hidden = self.t5_encoder(t5_inputs.input_ids).last_hidden_state
        
        return t5_hidden.to(self.dtype), pooled.to(self.dtype)


class CaptionForecastLoss(nn.Module):
    """
    Loss function for training the caption forecaster.
    
    Components:
    1. MSE on T5 hidden states (main signal)
    2. Cosine similarity on pooled CLIP embeddings (global semantic match)
    3. Optional: contrastive loss to prevent mode collapse
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
    
    def forward(
        self,
        pred_hidden: torch.Tensor,    # (B, L, D) predicted
        target_hidden: torch.Tensor,  # (B, L, D) ground truth
        pred_pooled: torch.Tensor = None,   # (B, D_pool) optional
        target_pooled: torch.Tensor = None, # (B, D_pool) optional
    ):
        losses = {}
        
        # 1. MSE on hidden states
        mse = F.mse_loss(pred_hidden, target_hidden)
        losses["caption_mse"] = mse
        
        # 2. Cosine similarity (global direction match)
        pred_mean = pred_hidden.mean(dim=1)
        target_mean = target_hidden.mean(dim=1)
        cos_sim = F.cosine_similarity(pred_mean, target_mean, dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        losses["caption_cos"] = cos_loss
        losses["caption_cos_sim"] = cos_sim
        
        # 3. Contrastive (prevent collapse)
        B = pred_hidden.size(0)
        if B > 1:
            pred_norm = F.normalize(pred_mean, dim=-1)
            target_norm = F.normalize(target_mean, dim=-1)
            logits = torch.matmul(pred_norm, target_norm.T) / self.temperature
            labels = torch.arange(B, device=pred_hidden.device)
            contr = F.cross_entropy(logits, labels)
            losses["caption_contr"] = contr
        else:
            contr = torch.tensor(0.0, device=pred_hidden.device)
            losses["caption_contr"] = contr
        
        total = (
            self.mse_weight * mse
            + self.cosine_weight * cos_loss
            + self.contrastive_weight * contr
        )
        
        return total, losses