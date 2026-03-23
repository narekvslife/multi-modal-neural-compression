# Multi-Modal Multi-Task Neural Compression

Source code accompanying the paper **[Multi-Modal Multi-Task Dataset Compression](Multi-Modal%20Multi-Task%20Dataset%20Compression.pdf)** (EPFL VILAB).

## Overview

This project explores neural image compression when a single bitstream must serve multiple visual tasks simultaneously (RGB, depth, surface normals, semantic segmentation). Rather than compressing each modality independently, the models learn a shared latent representation optimized for rate-distortion across all tasks at once.

## Models

Four compression architectures are implemented, all built on top of a [CompressAI](https://github.com/InterDigitalInc/CompressAI) `ScaleHyperprior` backbone:

| # | Class | Description |
|---|-------|-------------|
| 1 | `SingleTaskCompressor` | Baseline — one model per task |
| 2 | `MultiTaskMixedLatentCompressor` | Task inputs are embedded separately then mixed into a single shared latent |
| 3 | `MultiTaskDisjointLatentCompressor` | Latent space is partitioned per task so each task can be decoded from its own slice |
| 4 | `MultiTaskSharedLatentCompressor` | Each task code = shared latent + task-specific latent |

## Source Layout

```
src/
├── train.py                  # Training entry point (PyTorch Lightning + W&B)
├── compress.py               # Compress / decompress utilities
├── models/
│   ├── multi_task_compressor.py   # Abstract base class
│   ├── mixed_latent.py
│   ├── disjoint_latent.py
│   ├── shared_latent.py
│   └── single_task_compressor.py
├── datasets/
│   ├── clevr.py              # CLEVR multi-task dataset loader
│   ├── task_configs.py       # Per-task channel counts and loss functions
│   └── transforms.py
├── loss_balancing.py         # Uncertainty-weighting for multi-task losses
├── callbacks.py              # W&B prediction logging callback
└── utils.py
```

## Supported Tasks & Datasets

**Tasks:** `rgb`, `depth_euclidean`, `normal`, `semantic`, `mono`

**Datasets:** CLEVR (primary), MNIST, FashionMNIST

## Training

```bash
python src/train.py \
  -d clevr \
  -t rgb depth_euclidean normal \
  -m 3 \
  -l 128 -c 100 \
  --lmbda 1e-2 \
  -w my-run-name \
  -e 500
```

Key arguments:

| Flag | Description |
|------|-------------|
| `-m` | Model type (1–4, see table above) |
| `-l` | Latent channels (bottleneck width) |
| `-c` | Conv channels throughout the network |
| `--lmbda` | Rate-distortion trade-off weight |

Training logs and checkpoints are tracked with [Weights & Biases](https://wandb.ai).

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `torch 2.0`, `pytorch-lightning 2.0`, `compressai 1.2.4`, `wandb`.
