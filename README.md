# Sat-JEPA-Diff

**Bridging Self-Supervised Learning and Generative Diffusion for Satellite Image Forecasting**

*Accepted at ICLR 2026 — Machine Learning for Remote Sensing (ML4RS) Workshop*

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](https://openreview.net)
[![Dataset & Model](https://img.shields.io/badge/Zenodo-Dataset%20%26%20Model-orange)](https://doi.org/10.5281/zenodo.18868643)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="docs/ijepa_diagram.pdf" width="100%" alt="Sat-JEPA-Diff Architecture"/>
</p>

Sat-JEPA-Diff predicts future satellite images by first forecasting **what** will be there (semantic embeddings via IJEPA), then generating **how** it looks (RGB imagery via Stable Diffusion 3.5). This two-stage approach produces sharp, realistic predictions that preserve roads, buildings, and vegetation boundaries — details that traditional methods blur away.

---

## Key Results

| Model | L1 ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | GSSIM ↑ | LPIPS ↓ | FID ↓ |
|-------|------|-------|--------|--------|---------|---------|-------|
| *Deterministic Baselines* | | | | | | | |
| Default | 0.0131 | 0.0008 | 37.52 | 0.9361 | 0.7858 | **0.0708** | 0.6959 |
| PredRNN | **0.0117** | 0.0005 | **38.38** | **0.9476** | 0.7836 | 0.0726 | 9.9720 |
| SimVP v2 | 0.0131 | 0.0006 | 37.63 | 0.9391 | 0.7719 | 0.0928 | 18.7208 |
| *Generative Models* | | | | | | | |
| Stable Diff. 3.5 | 0.0175 | 0.0005 | 32.98 | 0.8398 | 0.8711 | 0.4528 | 0.1533 |
| MCVD | 0.0314 | 0.0031 | 31.28 | 0.8637 | 0.7665 | 0.1890 | 0.1956 |
| **Ours** | 0.0158 | **0.0004** | 33.81 | 0.8672 | **0.8984** | 0.4449 | **0.1475** |

Our model achieves **+11% GSSIM** over the best baseline, confirming superior preservation of geospatial boundaries and structural gradients.

---

## How It Works

```
Input: Satellite image at time t
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
 IJEPA Encoder         Downsample
 + Predictor            to 32×32
    │                       │
    ▼                       │
 Predicted t+1              │
 embeddings                 │
    │                       │
    └───────┬───────────────┘
            ▼
    Conditioning Adapter
    (fusion gate α)
            │
            ▼
    Frozen SD 3.5 + LoRA
            │
            ▼
Output: Predicted RGB at time t+1
```

1. **IJEPA** encodes the current image and predicts future semantic embeddings
2. The current image is **downsampled to 32×32** for coarse spatial structure
3. A **conditioning adapter** fuses both signals via a learned gate
4. **Stable Diffusion 3.5** (frozen backbone + LoRA) generates the final image

---

## Installation

```bash
git clone https://github.com/VU-AIML/SAT-JEPA-DIFF.git
cd SAT-JEPA-DIFF

conda create -n satjepa python=3.12
conda activate satjepa

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers peft accelerate
pip install rasterio matplotlib pyyaml lpips
```

You'll need a [Hugging Face token](https://huggingface.co/settings/tokens) with access to [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). Set it in the config file or as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

---

## Dataset

We use Sentinel-2 RGB imagery (10m GSD) paired with Google Earth Engine Foundation Model embeddings across 100 global Regions of Interest (2017–2024).

**Download:** The full dataset and pretrained model weights are available on [Zenodo (DOI: 10.5281/zenodo.18868643)](https://doi.org/10.5281/zenodo.18868643).

**Expected directory structure:**

```
downloads/
├── Region_Name_1/
│   ├── s2_rgb_Region_Name_1_2017_10km.tif
│   ├── s2_rgb_Region_Name_1_2018_10km.tif
│   ├── ...
│   ├── gee_embeddings_Region_Name_1_2017_10km.tif   # 64-dim per pixel
│   └── gee_embeddings_Region_Name_1_2018_10km.tif
├── Region_Name_2/
│   └── ...
```

---

## Training

**Configure** your paths and hyperparameters in `src/config/s2_future_vith16.yaml`, then:

```bash
python src/main.py --fname path/to/config/s2_future_vith16.yaml
```

Key config options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_ijepa` | `true` | Train IJEPA module (set `false` for SD-only) |
| `use_lora` | `true` | LoRA fine-tuning on SD 3.5 |
| `lora_rank` | `8` | LoRA rank |
| `sd_loss_weight` | `1.0` | Weight for diffusion loss |
| `crop_size` | `128` | Input image resolution |
| `batch_size` | `8` | Per-GPU batch size |

Training takes approximately **5 days** on a single NVIDIA RTX 5090 (24GB) with our dataset.

---

## Inference

**Single image prediction:**

```bash
python src/inference.py --checkpoint path/to/best.pth.tar --input image_t.tif --output prediction_t+1.tif
```

**Autoregressive rollout** (multi-year forecasting):

```bash
python src/autoregressive_rollout.py --checkpoint path/to/best.pth.tar --steps 30
```

---

## Evaluation

Run the full evaluation suite on the validation set:

```bash
python src/evaluate.py --checkpoint path/to/best.pth.tar
```

This computes L1, MSE, PSNR, SSIM, GSSIM, LPIPS, and FID metrics.

---

## Project Structure

```
src/
├── config/
│   └── s2_future_vith16.yaml    # Training configuration
├── data/
│   └── data.py                  # Sentinel-2 + GEE embedding dataset
├── masks/
│   └── multiblock.py            # IJEPA multi-block masking
├── models/
│   └── vision_transformer.py    # ViT encoder & predictor
├── utils/
│   ├── distributed.py           # Multi-GPU utilities
│   ├── logging.py               # CSV logger & metrics
│   ├── schedulers.py            # LR & weight decay schedules
│   └── tensors.py               # Tensor operations
├── main.py                      # Entry point
├── train.py                     # Training loop (IJEPA + SD joint training)
├── sd_models.py                 # SD 3.5 loading, LoRA, conditioning adapter
├── sd_joint_loss.py             # Flow matching loss + diffusion sampling
├── metrics.py                   # PSNR, SSIM, GSSIM, LPIPS, FID
├── inference.py                 # Single-step prediction
├── evaluate.py                  # Full evaluation pipeline
├── autoregressive_rollout.py    # Multi-year recursive prediction
├── embedding_validation.py      # Embedding consistency checks
└── helper.py                    # Model & optimizer initialization
```

---

## Citation

```bibtex
@inproceedings{komurcu2026satjepadiff,
  title={Sat-JEPA-Diff: Bridging Self-Supervised Learning and Generative Diffusion for Remote Sensing},
  author={Kömürcü Kürşat and Petkevicius, Linas},
  booktitle={ICLR 2026 Machine Learning for Remote Sensing (ML4RS) Workshop},
  year={2026}
}
```

---

## Acknowledgments

This project was funded by the European Union (project No S-MIP-23-45) under the agreement with the Research Council of Lithuania (LMTLT).

The IJEPA implementation is based on [Meta's I-JEPA](https://github.com/facebookresearch/ijepa). The diffusion backbone uses [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) by Stability AI.
