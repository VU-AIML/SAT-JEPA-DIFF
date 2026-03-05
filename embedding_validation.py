"""
Embedding Consistency Validation Module

Purpose: Validate that generated t+1 images produce similar embeddings
to ground truth t+1 images.

This proves that the predicted embedding carries meaningful information
through the diffusion process.
"""

import os
import csv
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingConsistencyValidator:
    """
    Validates embedding consistency between generated and ground truth images.
    
    Core question: Does gen_rgb(t+1) produce the same embedding as gt_rgb(t+1)?
    """
    
    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        target_encoder: torch.nn.Module,
        sd_state: Dict,
        device: torch.device,
        output_dir: str = "./embedding_validation",
        diffusion_steps: int = 20,
        img2img_strength: float = 0.75,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.sd_state = sd_state
        self.device = device
        self.output_dir = output_dir
        self.diffusion_steps = diffusion_steps
        self.img2img_strength = img2img_strength
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(output_dir, "embedding_consistency.csv")
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch',
                    'cos_gen_gt_mean', 'cos_gen_gt_std',
                    'cos_random_gt_mean', 'cos_random_gt_std',
                    'l2_gen_gt_mean', 'l2_gen_gt_std',
                    'num_samples',
                ])
    
    def _log_to_csv(self, epoch: int, results: Dict):
        """Append results to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                results['cos_gen_gt_mean'],
                results['cos_gen_gt_std'],
                results['cos_random_gt_mean'],
                results['cos_random_gt_std'],
                results['l2_gen_gt_mean'],
                results['l2_gen_gt_std'],
                results['num_samples'],
            ])
    
    @torch.no_grad()
    def get_predicted_embedding(
        self,
        rgb_t: torch.Tensor,
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predict t+1 embedding from t image."""
        z_enc = self.encoder(rgb_t, masks_enc)
        z_pred = self.predictor(z_enc, masks_enc, masks_pred)
        
        B = rgb_t.shape[0]
        num_masks = z_pred.shape[0] // B
        if num_masks > 1:
            z_pred = z_pred[::num_masks]
        
        return z_pred
    
    @torch.no_grad()
    def get_image_embedding(self, rgb: torch.Tensor) -> torch.Tensor:
        """Get embedding of an RGB image using target encoder."""
        return self.target_encoder(rgb)
    
    @torch.no_grad()
    def generate_image(
        self,
        ijepa_tokens: torch.Tensor,
        rgb_source: torch.Tensor,
    ) -> torch.Tensor:
        """Generate t+1 image using diffusion."""
        from sd_joint_loss import diffusion_sample
        
        H, W = rgb_source.shape[-2:]
        
        gen_rgb = diffusion_sample(
            unet=self.sd_state["unet"],
            vae=self.sd_state["vae"],
            scheduler=self.sd_state["noise_scheduler"],
            cond_adapter=self.sd_state["cond_adapter"],
            ijepa_tokens=ijepa_tokens,
            text_embeds=self.sd_state.get("prompt_embeds"),
            pooled_text_embeds=self.sd_state.get("pooled_prompt_embeds"),
            num_steps=self.diffusion_steps,
            image_size=(H, W),
            device=self.device,
            ref_rgb=rgb_source,
            img2img_strength=self.img2img_strength,
        )
        
        return gen_rgb
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        emb1_flat = emb1.view(emb1.shape[0], -1)
        emb2_flat = emb2.view(emb2.shape[0], -1)
        emb1_norm = F.normalize(emb1_flat, dim=-1)
        emb2_norm = F.normalize(emb2_flat, dim=-1)
        return (emb1_norm * emb2_norm).sum(dim=-1)
    
    def l2_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance between embeddings."""
        emb1_flat = emb1.view(emb1.shape[0], -1)
        emb2_flat = emb2.view(emb2.shape[0], -1)
        return torch.norm(emb1_flat - emb2_flat, dim=-1)
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
        max_batches: int = 30,
    ) -> Dict[str, float]:
        """
        Main validation: Compare embeddings of generated vs ground truth t+1 images.
        
        For each sample:
        1. Predict t+1 embedding from t image
        2. Generate t+1 image using predicted embedding
        3. Get embedding of generated image
        4. Get embedding of ground truth t+1 image
        5. Compare: cos_sim(gen_emb, gt_emb)
        """
        print("\n" + "="*60)
        print("EMBEDDING CONSISTENCY VALIDATION")
        print("="*60)
        
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()
        
        cos_gen_gt_list = []
        cos_random_gt_list = []
        l2_gen_gt_list = []
        
        # For visualization
        all_gen_emb = []
        all_gt_emb = []
        
        for batch_idx, (data, masks_enc, masks_pred) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{max_batches}")
            
            rgb_t = data[0].to(self.device)
            rgb_tp1 = data[2].to(self.device)
            masks_enc = [m.to(self.device) for m in masks_enc]
            masks_pred = [m.to(self.device) for m in masks_pred]
            
            # 1. Predict t+1 embedding
            pred_emb = self.get_predicted_embedding(rgb_t, masks_enc, masks_pred)
            
            # 2. Generate t+1 image
            gen_rgb = self.generate_image(pred_emb, rgb_t)
            
            # 3. Get embedding of generated image
            gen_emb = self.get_image_embedding(gen_rgb)
            
            # 4. Get embedding of ground truth t+1
            gt_emb = self.get_image_embedding(rgb_tp1)
            
            # 5. Random baseline
            random_emb = torch.randn_like(gt_emb)
            
            # 6. Compute metrics
            cos_gen_gt = self.cosine_similarity(gen_emb, gt_emb)
            cos_random_gt = self.cosine_similarity(random_emb, gt_emb)
            l2_gen_gt = self.l2_distance(gen_emb, gt_emb)
            
            cos_gen_gt_list.extend(cos_gen_gt.cpu().tolist())
            cos_random_gt_list.extend(cos_random_gt.cpu().tolist())
            l2_gen_gt_list.extend(l2_gen_gt.cpu().tolist())
            
            # Store for visualization
            gen_emb_pooled = gen_emb.mean(dim=1).cpu().numpy()
            gt_emb_pooled = gt_emb.mean(dim=1).cpu().numpy()
            all_gen_emb.append(gen_emb_pooled)
            all_gt_emb.append(gt_emb_pooled)
            
            # Save visual comparison for first batch
            if batch_idx == 0:
                self._save_comparison_grid(rgb_t, rgb_tp1, gen_rgb, epoch)
        
        # Aggregate results
        results = {
            'cos_gen_gt_mean': float(np.mean(cos_gen_gt_list)),
            'cos_gen_gt_std': float(np.std(cos_gen_gt_list)),
            'cos_random_gt_mean': float(np.mean(cos_random_gt_list)),
            'cos_random_gt_std': float(np.std(cos_random_gt_list)),
            'l2_gen_gt_mean': float(np.mean(l2_gen_gt_list)),
            'l2_gen_gt_std': float(np.std(l2_gen_gt_list)),
            'num_samples': len(cos_gen_gt_list),
        }
        
        # Print results
        print("\n" + "-"*50)
        print("Results:")
        print(f"  gen_emb <-> gt_emb cosine:  {results['cos_gen_gt_mean']:.4f} +/- {results['cos_gen_gt_std']:.4f}")
        print(f"  random <-> gt_emb cosine:   {results['cos_random_gt_mean']:.4f} +/- {results['cos_random_gt_std']:.4f}")
        print(f"  gen_emb <-> gt_emb L2:      {results['l2_gen_gt_mean']:.4f} +/- {results['l2_gen_gt_std']:.4f}")
        print(f"  Samples: {results['num_samples']}")
        print("-"*50)
        
        # Interpretation
        improvement = results['cos_gen_gt_mean'] - results['cos_random_gt_mean']
        print(f"\n  Improvement over random: {improvement:.4f}")
        if improvement > 0.3:
            print("  [GOOD] Generated images preserve semantic information well")
        elif improvement > 0.1:
            print("  [OK] Generated images preserve some semantic information")
        else:
            print("  [WARN] Generated images may not preserve semantic information")
        
        # Log to CSV
        self._log_to_csv(epoch, results)
        
        # Create t-SNE visualization
        all_gen_emb = np.vstack(all_gen_emb)
        all_gt_emb = np.vstack(all_gt_emb)
        self._create_tsne_plot(all_gen_emb, all_gt_emb, epoch)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return results
    
    def _save_comparison_grid(
        self,
        rgb_t: torch.Tensor,
        rgb_tp1: torch.Tensor,
        gen_rgb: torch.Tensor,
        epoch: int,
    ):
        """Save visual comparison: [Source t | GT t+1 | Generated t+1]"""
        n_samples = min(4, rgb_t.shape[0])
        
        rows = []
        for i in range(n_samples):
            row = torch.cat([
                rgb_t[i].clamp(0, 1),
                rgb_tp1[i].clamp(0, 1),
                gen_rgb[i].clamp(0, 1),
            ], dim=2)
            rows.append(row)
        
        grid = torch.cat(rows, dim=1)
        save_path = os.path.join(self.output_dir, f"comparison_ep{epoch}.png")
        save_image(grid, save_path)
        print(f"  Saved: {save_path}")
    
    def _create_tsne_plot(
        self,
        gen_emb: np.ndarray,
        gt_emb: np.ndarray,
        epoch: int,
    ):
        """Create t-SNE visualization of gen vs gt embeddings."""
        if not TSNE_AVAILABLE:
            print("  [SKIP] t-SNE not available (sklearn not installed)")
            return
        
        n_samples = len(gen_emb)
        if n_samples < 10:
            print("  [SKIP] Too few samples for t-SNE")
            return
        
        print("  Creating t-SNE visualization...")
        
        # Combine embeddings
        all_emb = np.vstack([gen_emb, gt_emb])
        labels = np.array([0]*n_samples + [1]*n_samples)
        
        # Run t-SNE
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        emb_2d = tsne.fit_transform(all_emb)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generated embeddings
        gen_mask = labels == 0
        ax.scatter(
            emb_2d[gen_mask, 0], emb_2d[gen_mask, 1],
            c='#e41a1c', label='Generated t+1', alpha=0.6, s=50, marker='o'
        )
        
        # GT embeddings
        gt_mask = labels == 1
        ax.scatter(
            emb_2d[gt_mask, 0], emb_2d[gt_mask, 1],
            c='#377eb8', label='Ground Truth t+1', alpha=0.6, s=50, marker='^'
        )
        
        # Draw lines connecting same-sample pairs
        for i in range(min(50, n_samples)):
            gen_idx = i
            gt_idx = n_samples + i
            ax.plot(
                [emb_2d[gen_idx, 0], emb_2d[gt_idx, 0]],
                [emb_2d[gen_idx, 1], emb_2d[gt_idx, 1]],
                'k-', alpha=0.1, linewidth=0.5
            )
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f'Embedding Space: Generated vs Ground Truth (Epoch {epoch})')
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"tsne_ep{epoch}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")