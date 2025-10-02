# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DEIMv2 is a real-time object detection framework that combines DEIM (DETR with Improved Matching) with DINOv3 features. The codebase supports multiple model variants from ultra-light (Atto) to large (X) models, achieving state-of-the-art performance on COCO benchmark.

## Key Commands

### Environment Setup
```bash
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### Training
```bash
# ViT-based variants (S, M, L, X)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --use-amp --seed=0

# HGNetv2-based variants (Atto, Femto, Pico, N)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

Replace `${model}` with: `s`, `m`, `l`, `x` for DINOv3 variants, or `atto`, `femto`, `pico`, `n` for HGNetv2 variants.

### Testing
```bash
# Test with checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --test-only -r model.pth
```

### Tuning (Resume Training)
```bash
# Continue training from checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml --use-amp --seed=0 -t model.pth
```

### Deployment
```bash
# Export to ONNX
python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth

# Convert to TensorRT
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

### Inference
```bash
# ONNX runtime
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg

# TensorRT
python tools/inference/trt_inf.py --trt model.engine --input image.jpg

# PyTorch
python tools/inference/torch_inf.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```

### Utilities
```bash
# Model FLOPs, MACs, and Parameters
python tools/benchmark/get_info.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml

# TensorRT Latency Benchmark
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine

# Fiftyone Visualization
python tools/visualization/fiftyone_vis.py -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth
```

## Architecture

### Core Components

1. **DEIM Model (`engine/deim/deim.py`)**: Main detection model composed of:
   - **Backbone**: Feature extraction (DINOv3 adapters or HGNetv2)
   - **Encoder**: Feature encoding (HybridEncoder or LiteEncoder)
   - **Decoder**: Detection head (DEIMTransformer or RTDETRTransformer)

2. **Backbones (`engine/backbone/`)**:
   - `dinov3_adapter.py`: Adapters for DINOv3 variants (ViT-Tiny, Small, Base, Large)
   - `vit_tiny.py`: Custom distilled ViT-Tiny implementation
   - `hgnetv2.py`: HGNetv2 for lightweight variants

3. **Encoders (`engine/deim/`)**:
   - `hybrid_encoder.py`: Hybrid encoder for multi-scale features
   - `lite_encoder.py`: Lightweight encoder for efficient models

4. **Decoders (`engine/deim/`)**:
   - `deim_decoder.py`: DEIM decoder with improved matching
   - `rtdetrv2_decoder.py`: RT-DETR v2 decoder variant
   - `dfine_decoder.py`: D-FINE decoder variant

5. **Training Infrastructure**:
   - `engine/solver/det_solver.py`: Detection solver managing training loop
   - `engine/solver/det_engine.py`: Training and evaluation engine
   - `engine/deim/deim_criterion.py`: Loss computation with improved matcher
   - `engine/deim/matcher.py`: Bipartite matching for DETR-based detection

### Configuration System

The codebase uses YAML-based hierarchical configuration:
- **Base configs** (`configs/base/`): Shared components (optimizer, dataloader, model architectures)
- **Dataset configs** (`configs/dataset/`): Dataset-specific settings (COCO, custom datasets)
- **Model configs** (`configs/deimv2/`): Final model configurations that include base configs

Model configs use `__include__` to compose multiple base configs. Example structure:
```yaml
__include__: [
  '../dataset/coco_detection.yml',
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/deimv2.yml',
]
```

### Data Pipeline

1. **Transforms** (`engine/data/transforms/`): Data augmentation including Mosaic, RandomIoUCrop, photometric distortions
2. **Collate Functions** (`engine/data/dataloader.py`): Batch collation with Mixup and CopyBlend augmentation
3. **Datasets** (`engine/data/dataset/`): COCO format dataset loaders

### Training Features

- **Two-Stage Training**: Initial stage with strong augmentations, second stage with EMA refinement (controlled by `stop_epoch`)
- **Dynamic Matcher**: Switches from IoU-based to classification-aware matching at `matcher_change_epoch`
- **Custom LR Scheduler**: `FlatCosineLRScheduler` with warmup, flat, and cosine phases
- **EMA**: Exponential Moving Average with restart mechanism between stages

## Dataset Setup

### COCO2017
1. Download from [COCO](https://cocodataset.org/#download)
2. Update paths in `configs/dataset/coco_detection.yml`:
```yaml
train_dataloader:
  img_folder: /data/COCO2017/train2017/
  ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
  img_folder: /data/COCO2017/val2017/
  ann_file: /data/COCO2017/annotations/instances_val2017.json
```

### Custom Dataset
1. Organize in COCO format
2. Set `remap_mscoco_category: False` in config
3. Update `num_classes` and paths in a custom config (see `configs/dataset/custom_detection.yml`)

### Backbone Checkpoints

Download pretrained backbones and place in `./ckpts/`:
- DINOv3: From [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) → `dinov3_vits16.pth`
- Distilled ViT-Tiny: [Download link](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) → `vitt_distill.pt`
- Distilled ViT-Tiny+: [Download link](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) → `vittplus_distill.pt`

## Customization Guidelines

### Batch Size
When modifying `total_batch_size`, scale learning rates linearly and adjust EMA decay:
```yaml
optimizer:
  lr: 0.0005  # scale with batch size
ema:
  decay: 0.9998  # adjust by: 1 - (1 - decay) * scale_factor
  warmups: 500  # scale inversely
```

### Input Size
Update `eval_spatial_size`, Mosaic `output_size` (typically input_size / 2), Resize `size`, and collate `base_size`.

### Training Epochs
Key parameters to adjust together:
- `epoches`: Total epochs
- `flat_epoch`: Epoch when LR starts cosine decay (typically 4 + epoches // 2)
- `no_aug_epoch`: Epochs without augmentation at end (typically 4n where n is model size factor)
- `stop_epoch`: When stage 1 ends (typically epoches - no_aug_epoch)
- `matcher_change_epoch`: When matcher switches (~90% of stop_epoch)

## DINOv3 Integration

The `dinov3/` subdirectory contains the DINOv3 framework from Meta. It's integrated via:
- `engine/backbone/dinov3_adapter.py`: Adapts DINOv3 backbones for detection
- `engine/backbone/vit_tiny.py`: Custom distilled variants

When using DINOv3 backbones, specify in config:
```yaml
DEIM:
  backbone: DINOv3STAs  # or DINOv3S, DINOv3M, DINOv3L, DINOv3X

DINOv3STAs:
  name: vit_tiny
  weights_path: ./ckpts/vitt_distill.pt
  interaction_indexes: [5,8,11]  # layers for multi-scale features
```
