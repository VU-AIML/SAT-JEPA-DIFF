"""
IJEPA Loss Ablation Study
==========================
Runs 5 ablation configurations (IJEPA only, no SD) and produces:
  1. A summary table (CSV + LaTeX)
  2. Training curves (PNG) for paper appendix

Usage:
    python ablation.py --fname s2_future_vith16.yaml [--epochs 30] [--output_dir ./ablation_results]

Each configuration trains from scratch with identical seeds, data splits, 
and hyperparameters — only the loss weights differ.

Configurations:
    A) L1 only
    B) L1 + Cosine
    C) L1 + Cosine + Spatial
    D) L1 + Cosine + Contrastive
    E) Full (L1 + Cosine + Spatial + Contrastive + Feature Regression)
"""

import os
import sys
import copy
import math
import time
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── project imports ──────────────────────────────────────────────────────────
from data.data import S2FutureEmbeddingDataset
from masks.multiblock import MaskCollator as MBMaskCollator
from masks.utils import apply_masks
from utils.distributed import init_distributed
from utils.logging import AverageMeter
from utils.tensors import repeat_interleave_batch
from helper import init_model, init_opt

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ablation")

_GLOBAL_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations:  (name, l1, cosine, spatial, contrastive, feat_reg)
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    ("A: L1 only",                   20.0, 0.0, 0.0, 0.0, 0.0),
    ("B: L1+Cos",                    20.0, 2.0, 0.0, 0.0, 0.0),
    ("C: L1+Cos+Spatial",            20.0, 2.0, 2.0, 0.0, 0.0),
    ("D: L1+Cos+Contrastive",        20.0, 2.0, 0.0, 0.5, 0.0),
    ("E: Full (Ours)",               20.0, 2.0, 2.0, 0.5, 5.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# HybridIJEPALoss (identical to train.py — kept here for self-containment)
# ─────────────────────────────────────────────────────────────────────────────
class HybridIJEPALoss(torch.nn.Module):
    def __init__(
        self,
        l1_weight=20.0,
        cosine_weight=1.0,
        contrastive_weight=1.0,
        spatial_var_weight=10.0,
        feature_reg_weight=5.0,
        temperature=0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        self.contrastive_weight = contrastive_weight
        self.spatial_var_weight = spatial_var_weight
        self.feature_reg_weight = feature_reg_weight
        self.temperature = temperature

    def forward(self, pred, target, pred_raw=None, target_raw=None):
        B, N, C = pred.shape
        losses = {}

        # L1
        l1_loss = F.l1_loss(pred, target)
        losses["l1"] = l1_loss

        # Cosine
        pred_n = F.normalize(pred, dim=-1, eps=1e-6)
        tgt_n = F.normalize(target, dim=-1, eps=1e-6)
        cos_sim = (pred_n * tgt_n).sum(dim=-1).mean()
        cosine_loss = 1.0 - cos_sim
        losses["cosine"] = cosine_loss
        losses["cos_sim"] = cos_sim

        # Spatial variance
        pred_std = torch.sqrt(pred.var(dim=1) + 1e-6).mean()
        tgt_std = torch.sqrt(target.var(dim=1) + 1e-6).mean()
        spatial_loss = F.mse_loss(pred_std, tgt_std)
        losses["spatial"] = spatial_loss

        # Log raw variance values for collapse detection
        losses["pred_var"] = pred.var(dim=1).mean()
        losses["tgt_var"] = target.var(dim=1).mean()

        # Contrastive
        if B > 1:
            pg = F.normalize(pred.mean(dim=1), dim=-1)
            tg = F.normalize(target.mean(dim=1), dim=-1)
            logits = pg @ tg.T / self.temperature
            labels = torch.arange(B, device=pred.device)
            contr_loss = F.cross_entropy(logits, labels)
            losses["contr"] = contr_loss
        else:
            contr_loss = torch.tensor(0.0, device=pred.device)
            losses["contr"] = contr_loss

        # Feature regression
        if pred_raw is not None and target_raw is not None:
            p_rn = F.layer_norm(pred_raw, (pred_raw.size(-1),))
            t_rn = F.layer_norm(target_raw, (target_raw.size(-1),))
            feat_loss = F.l1_loss(p_rn, t_rn)
            losses["feat_reg"] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=pred.device)
            losses["feat_reg"] = feat_loss

        total = (
            self.l1_weight * l1_loss
            + self.cosine_weight * cosine_loss
            + self.spatial_var_weight * spatial_loss
            + self.contrastive_weight * contr_loss
            + self.feature_reg_weight * feat_loss
        )
        return total, losses


# ─────────────────────────────────────────────────────────────────────────────
# Helper: patches from spatial embeddings
# ─────────────────────────────────────────────────────────────────────────────
def embedding_to_patches(embs, patch_size):
    B, C, H, W = embs.shape
    Hp, Wp = H // patch_size, W // patch_size
    embs = embs.view(B, C, Hp, patch_size, Wp, patch_size)
    embs = embs.mean(dim=(3, 5))
    return embs.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)


# ─────────────────────────────────────────────────────────────────────────────
# Single ablation run
# ─────────────────────────────────────────────────────────────────────────────
def run_single_ablation(
    config_name, loss_weights, args, train_loader, val_loader,
    device, num_epochs, use_bfloat16,
):
    """
    Train IJEPA from scratch with given loss weights. Returns history dict.
    
    Args:
        config_name: str, human-readable name
        loss_weights: tuple (l1, cosine, spatial, contrastive, feat_reg)
        args: full yaml config dict
        train_loader / val_loader: DataLoaders
        device: torch device
        num_epochs: int
        use_bfloat16: bool
    Returns:
        history: dict with per-epoch metrics
    """
    l1_w, cos_w, spat_w, contr_w, feat_w = loss_weights

    logger.info(f"\n{'='*70}")
    logger.info(f"  ABLATION: {config_name}")
    logger.info(f"  Weights: L1={l1_w}, Cos={cos_w}, Spatial={spat_w}, "
                f"Contr={contr_w}, FeatReg={feat_w}")
    logger.info(f"{'='*70}")

    # ── Reproducibility ──
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.cuda.manual_seed_all(_GLOBAL_SEED)

    # ── Parse config ──
    patch_size   = args["mask"]["patch_size"]
    crop_size    = args["data"]["crop_size"]
    pred_depth   = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    target_emb_dim = args["meta"].get("target_emb_dim", 64)
    model_name   = args["meta"]["model_name"]

    wd        = float(args["optimization"]["weight_decay"])
    final_wd  = float(args["optimization"]["final_weight_decay"])
    start_lr  = args["optimization"]["start_lr"]
    lr        = args["optimization"]["lr"]
    final_lr  = args["optimization"]["final_lr"]
    warmup    = args["optimization"]["warmup"]
    ema_range = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]

    # ── Build models ──
    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    import copy
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    enc_dim = getattr(encoder, "embed_dim", 768)
    proj_head = torch.nn.Linear(enc_dim, target_emb_dim).to(device)

    # ── Loss ──
    loss_fn = HybridIJEPALoss(
        l1_weight=l1_w,
        cosine_weight=cos_w,
        spatial_var_weight=spat_w,
        contrastive_weight=contr_w,
        feature_reg_weight=feat_w,
    ).to(device)

    # ── Optimizer ──
    ipe = len(train_loader)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor,
        wd=wd, final_wd=final_wd,
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr,
        iterations_per_epoch=ipe, warmup=warmup,
        num_epochs=num_epochs, ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    optimizer.add_param_group({"params": proj_head.parameters()})

    momentum_scheduler = (
        ema_range[0] + i * (ema_range[1] - ema_range[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    # ── History ──
    history = defaultdict(list)
    num_patches = (crop_size // patch_size) ** 2

    # ── Training loop ──
    for epoch in range(num_epochs):
        encoder.train(); predictor.train(); proj_head.train()
        epoch_t0 = time.time()

        epoch_meters = {
            "loss": AverageMeter(), "l1": AverageMeter(),
            "cosine": AverageMeter(), "cos_sim": AverageMeter(),
            "spatial": AverageMeter(), "contr": AverageMeter(),
            "feat_reg": AverageMeter(), "pred_var": AverageMeter(),
            "tgt_var": AverageMeter(),
        }

        num_iters = len(train_loader)
        for itr, (batch_data, masks_enc, masks_pred) in enumerate(train_loader):
            imgs_t   = batch_data[0].to(device, non_blocking=True)
            embs_tp1 = batch_data[1].to(device, non_blocking=True)
            rgb_tp1  = batch_data[2].to(device, non_blocking=True)
            masks_enc  = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]

            dtype = torch.bfloat16 if use_bfloat16 else torch.float32

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                # Target
                with torch.no_grad():
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),))
                    B_local = h.size(0)

                    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(B_local, -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)

                    h_masked = apply_masks(h, masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(masks_enc))

                    h_raw_masked = apply_masks(h_raw_all, masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(masks_enc))

                # Prediction
                z_enc = encoder(imgs_t, masks_enc)
                z_raw = predictor(z_enc, masks_enc, masks_pred)
                z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                z_proj = proj_head(z_raw_norm)

                # Loss
                loss, stats = loss_fn(
                    pred=z_proj, target=h_masked,
                    pred_raw=z_raw_norm, target_raw=h_raw_masked,
                )

            # Backward
            scheduler.step()
            wd_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if use_bfloat16:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()) + list(proj_head.parameters()),
                    1.0,
                )
                optimizer.step()
            else:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()) + list(proj_head.parameters()),
                        1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()) + list(proj_head.parameters()),
                        1.0,
                    )
                    optimizer.step()

            # EMA
            with torch.no_grad():
                m = next(momentum_scheduler)
                for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)

            # Meters
            epoch_meters["loss"].update(loss.item())
            for k in ["l1", "cosine", "cos_sim", "spatial", "contr", "feat_reg", "pred_var", "tgt_var"]:
                v = stats.get(k, torch.tensor(0.0))
                epoch_meters[k].update(v.item() if torch.is_tensor(v) else v)

            # Per-iteration progress log
            if itr % 100 == 0:
                elapsed_ep = time.time() - epoch_t0
                eta_ep = elapsed_ep / (itr + 1) * (num_iters - itr - 1)
                logger.info(
                    f"    [{config_name}] Ep {epoch+1} iter {itr}/{num_iters}  "
                    f"loss={epoch_meters['loss'].avg:.4f}  "
                    f"cos_sim={epoch_meters['cos_sim'].avg:.4f}  "
                    f"ETA_epoch={eta_ep/60:.1f}min"
                )

        epoch_elapsed = (time.time() - epoch_t0) / 60
        logger.info(f"    [{config_name}] Ep {epoch+1} train done in {epoch_elapsed:.1f} min")

        # ── Epoch logging ──
        for k, meter in epoch_meters.items():
            history[f"train_{k}"].append(meter.avg)

        # ── Validation ──
        encoder.eval(); predictor.eval(); proj_head.eval()
        val_meters = {
            "loss": AverageMeter(), "l1": AverageMeter(),
            "cosine": AverageMeter(), "cos_sim": AverageMeter(),
            "spatial": AverageMeter(), "contr": AverageMeter(),
            "feat_reg": AverageMeter(), "pred_var": AverageMeter(),
            "tgt_var": AverageMeter(),
        }

        with torch.no_grad():
            for batch_data, masks_enc, masks_pred in val_loader:
                imgs_t   = batch_data[0].to(device, non_blocking=True)
                embs_tp1 = batch_data[1].to(device, non_blocking=True)
                rgb_tp1  = batch_data[2].to(device, non_blocking=True)
                masks_enc  = [m.to(device) for m in masks_enc]
                masks_pred = [m.to(device) for m in masks_pred]

                dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bfloat16):
                    h = embedding_to_patches(embs_tp1, patch_size)
                    h = F.layer_norm(h, (h.size(-1),))
                    B_local = h.size(0)

                    full_mask = [torch.arange(num_patches, device=device).unsqueeze(0).expand(B_local, -1)]
                    h_raw_all = target_encoder(rgb_tp1, full_mask)

                    h_masked = apply_masks(h, masks_pred)
                    h_masked = repeat_interleave_batch(h_masked, B_local, repeat=len(masks_enc))
                    h_raw_masked = apply_masks(h_raw_all, masks_pred)
                    h_raw_masked = repeat_interleave_batch(h_raw_masked, B_local, repeat=len(masks_enc))

                    z_enc = encoder(imgs_t, masks_enc)
                    z_raw = predictor(z_enc, masks_enc, masks_pred)
                    z_raw_norm = F.layer_norm(z_raw, (z_raw.size(-1),))
                    z_proj = proj_head(z_raw_norm)

                    loss, stats = loss_fn(
                        pred=z_proj, target=h_masked,
                        pred_raw=z_raw_norm, target_raw=h_raw_masked,
                    )

                val_meters["loss"].update(loss.item())
                for k in ["l1", "cosine", "cos_sim", "spatial", "contr", "feat_reg", "pred_var", "tgt_var"]:
                    v = stats.get(k, torch.tensor(0.0))
                    val_meters[k].update(v.item() if torch.is_tensor(v) else v)

        for k, meter in val_meters.items():
            history[f"val_{k}"].append(meter.avg)

        # Variance ratio (collapse indicator)
        pv = val_meters["pred_var"].avg
        tv = val_meters["tgt_var"].avg
        var_ratio = pv / (tv + 1e-8)
        history["val_var_ratio"].append(var_ratio)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  [{config_name}] Ep {epoch+1:3d}/{num_epochs}  "
                f"train_loss={epoch_meters['loss'].avg:.4f}  "
                f"val_loss={val_meters['loss'].avg:.4f}  "
                f"cos_sim={val_meters['cos_sim'].avg:.4f}  "
                f"var_ratio={var_ratio:.4f}"
            )

    # Cleanup
    del encoder, predictor, target_encoder, proj_head, optimizer
    torch.cuda.empty_cache()

    return dict(history)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

# Academic-friendly colour palette (colourblind-safe)
COLORS = ["#E24A33", "#348ABD", "#988ED5", "#FBC15E", "#8EBA42"]
MARKERS = ["o", "s", "^", "D", "v"]


def plot_ablation_curves(all_histories, output_dir):
    """Generate publication-quality training curves."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IJEPA Loss Ablation — Training Dynamics", fontsize=15, fontweight="bold", y=0.98)

    metrics_to_plot = [
        ("val_loss",      "Total Validation Loss",       "lower is better"),
        ("val_cos_sim",   "Cosine Similarity (val)",      "higher is better"),
        ("val_var_ratio", "Variance Ratio (pred/target)", "≈1.0 is ideal"),
        ("val_l1",        "L1 Loss (val)",                "lower is better"),
        ("val_spatial",   "Spatial Variance Loss (val)",  "lower is better"),
        ("val_pred_var",  "Predicted Embedding Var (val)","should match target"),
    ]

    for ax, (metric, title, note) in zip(axes.flat, metrics_to_plot):
        for idx, (name, hist) in enumerate(all_histories.items()):
            vals = hist.get(metric, [])
            if len(vals) == 0:
                continue
            epochs = list(range(1, len(vals) + 1))
            ax.plot(
                epochs, vals,
                color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                markevery=max(1, len(epochs) // 10),
                markersize=4, linewidth=1.5, alpha=0.85,
                label=name,
            )

        # Reference line for variance ratio
        if metric == "val_var_ratio":
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(note, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(all_histories),
               fontsize=9, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(output_dir, "ablation_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves → {path}")
    return path


def plot_collapse_figure(all_histories, output_dir):
    """
    Dedicated spatial-collapse figure: 
    predicted embedding variance across epochs for each config.
    This directly shows whether a config causes collapse.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Spatial Collapse Analysis", fontsize=13, fontweight="bold")

    for idx, (name, hist) in enumerate(all_histories.items()):
        epochs = list(range(1, len(hist.get("val_pred_var", [])) + 1))

        ax1.plot(epochs, hist["val_pred_var"],
                 color=COLORS[idx], marker=MARKERS[idx],
                 markevery=max(1, len(epochs) // 10),
                 markersize=4, linewidth=1.5, label=name)

        ax2.plot(epochs, hist["val_var_ratio"],
                 color=COLORS[idx], marker=MARKERS[idx],
                 markevery=max(1, len(epochs) // 10),
                 markersize=4, linewidth=1.5, label=name)

    # Target variance reference on left plot
    # Use last config's target var (all configs share the same target)
    last_hist = list(all_histories.values())[-1]
    if "val_tgt_var" in last_hist and len(last_hist["val_tgt_var"]) > 0:
        ax1.axhline(y=last_hist["val_tgt_var"][-1], color="black",
                     linestyle="--", linewidth=1, alpha=0.7, label="Target var")

    ax1.set_title("Predicted Embedding Variance", fontsize=11)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Variance")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("Variance Ratio (pred / target)", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Ratio")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_collapse.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved collapse figure → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
def make_summary_table(all_histories, output_dir):
    """
    Produce CSV and LaTeX table of final-epoch validation metrics.
    """
    rows = []
    header = [
        "Config", "Val Loss↓", "L1↓", "Cos Sim↑",
        "Spatial↓", "Var Ratio", "Contr↓",
    ]

    for name, hist in all_histories.items():
        row = [
            name,
            f"{hist['val_loss'][-1]:.4f}",
            f"{hist['val_l1'][-1]:.4f}",
            f"{hist['val_cos_sim'][-1]:.4f}",
            f"{hist['val_spatial'][-1]:.6f}",
            f"{hist['val_var_ratio'][-1]:.4f}",
            f"{hist['val_contr'][-1]:.4f}" if hist["val_contr"][-1] > 0 else "—",
        ]
        rows.append(row)

    # CSV
    csv_path = os.path.join(output_dir, "ablation_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    # LaTeX
    latex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        ncols = len(header)
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Ablation study of IJEPA loss components on validation set. "
                "Var Ratio $\\approx 1.0$ indicates no spatial collapse.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{" + "l" + "c" * (ncols - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table}\n")

    logger.info(f"Saved table → {csv_path}")
    logger.info(f"Saved LaTeX → {latex_path}")

    # Print table to console
    logger.info("\n" + "=" * 90)
    logger.info("  ABLATION RESULTS (Final Epoch)")
    logger.info("=" * 90)
    col_widths = [max(len(header[i]), max(len(r[i]) for r in rows)) + 2 for i in range(ncols)]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    logger.info(fmt.format(*header))
    logger.info("-" * sum(col_widths))
    for row in rows:
        logger.info(fmt.format(*row))
    logger.info("=" * 90)

    return csv_path, latex_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IJEPA Loss Ablation Study")
    parser.add_argument("--fname", type=str, required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: use config value)")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="Directory for outputs")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config indices (0-4) or 'all'")
    parser.add_argument("--subset_ratio", type=float, default=1.0,
                        help="Use fraction of training data (e.g. 0.3 = 30%%) for faster runs")
    cli = parser.parse_args()

    # Load config
    with open(cli.fname, "r") as f:
        args = yaml.full_load(f)

    num_epochs = cli.epochs if cli.epochs else args["optimization"]["epochs"]
    output_dir = cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Which configs to run
    if cli.configs == "all":
        configs_to_run = list(range(len(ABLATION_CONFIGS)))
    else:
        configs_to_run = [int(x.strip()) for x in cli.configs.split(",")]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    use_bfloat16 = args["meta"]["use_bfloat16"]

    # ── Data ──
    logger.info("Building data loaders (shared across all configs)...")

    patch_size  = args["mask"]["patch_size"]
    crop_size   = args["data"]["crop_size"]
    batch_size  = args["data"]["batch_size"]
    pin_mem     = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path   = args["data"]["root_path"]

    mask_collator = MBMaskCollator(
        input_size=crop_size, patch_size=patch_size,
        pred_mask_scale=args["mask"]["pred_mask_scale"],
        enc_mask_scale=args["mask"]["enc_mask_scale"],
        aspect_ratio=args["mask"]["aspect_ratio"],
        nenc=args["mask"]["num_enc_masks"],
        npred=args["mask"]["num_pred_masks"],
        allow_overlap=args["mask"]["allow_overlap"],
        min_keep=args["mask"]["min_keep"],
    )

    full_dataset = S2FutureEmbeddingDataset(
        root_dir=root_path, patch_size=crop_size, transform=None,
    )

    np.random.seed(_GLOBAL_SEED)
    num_samples = len(full_dataset)
    indices = np.random.permutation(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    # Optionally subsample for faster ablation runs
    if cli.subset_ratio < 1.0:
        n_train = max(batch_size * 2, int(len(train_indices) * cli.subset_ratio))
        n_val = max(batch_size, int(len(val_indices) * cli.subset_ratio))
        train_indices = train_indices[:n_train]
        val_indices = val_indices[:n_val]
        logger.info(f"Subsampled to {cli.subset_ratio*100:.0f}%: {n_train} train, {n_val} val")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, collate_fn=mask_collator, shuffle=True,
        batch_size=batch_size, drop_last=True,
        pin_memory=pin_mem, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=mask_collator, shuffle=False,
        batch_size=batch_size, drop_last=False,
        pin_memory=pin_mem, num_workers=num_workers,
    )

    logger.info(f"Dataset: {num_samples} total, {len(train_indices)} train, "
                f"{len(val_indices)} val, {len(train_loader)} iters/epoch")
    logger.info(f"Running {len(configs_to_run)} ablation configs × {num_epochs} epochs")

    # ── Distributed init (needed by init_model even for single GPU) ──
    try:
        mp.set_start_method("spawn")
    except:
        pass
    init_distributed()

    # ── Run ablations ──
    all_histories = {}
    t_start = time.time()

    for cfg_idx in configs_to_run:
        name, l1_w, cos_w, spat_w, contr_w, feat_w = ABLATION_CONFIGS[cfg_idx]
        t0 = time.time()

        history = run_single_ablation(
            config_name=name,
            loss_weights=(l1_w, cos_w, spat_w, contr_w, feat_w),
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )

        elapsed = (time.time() - t0) / 60
        logger.info(f"  ✓ {name} completed in {elapsed:.1f} min")
        all_histories[name] = history

    total_time = (time.time() - t_start) / 3600
    logger.info(f"\nAll ablations done in {total_time:.2f} hours")

    # ── Save raw histories ──
    json_path = os.path.join(output_dir, "ablation_histories.json")
    with open(json_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Saved raw histories → {json_path}")

    # ── Generate outputs ──
    curves_path = plot_ablation_curves(all_histories, output_dir)
    collapse_path = plot_collapse_figure(all_histories, output_dir)
    csv_path, latex_path = make_summary_table(all_histories, output_dir)

    logger.info(f"\n{'='*70}")
    logger.info("  ABLATION STUDY COMPLETE")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Files:")
    logger.info(f"    - {curves_path}")
    logger.info(f"    - {collapse_path}")
    logger.info(f"    - {csv_path}")
    logger.info(f"    - {latex_path}")
    logger.info(f"    - {json_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()