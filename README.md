# FuncGrasp: Functional Grasp Generation with Flow Matching

A training pipeline for learning functional grasps using vision-language models and flow matching.

## Repository Structure

```
FuncGrasp/
├── models/               # Model implementations
│   ├── functional_grasp_model.py  # Main model integrating all components
│   ├── semantics_qwen.py          # Qwen2.5-VL wrapper for semantic encoding
│   ├── pointnet2_encoder.py       # PointNet++ for geometric features
│   ├── fusion_transformer.py      # Transformer for multimodal fusion
│   ├── contact_head.py            # 7-way contact prediction head
│   └── flow_matching.py           # Conditional flow matching for poses
├── data/                 # Data loaders and utilities
│   ├── oakink_loader.py           # OakInk dataset loader
│   └── prepare_data.py            # Object rendering utilities
├── docs/                 # Documentation
│   ├── pipeline.md                # Architecture, training details, and usage
│   └── dataset.md                 # Dataset documentation
├── config.py             # Training configuration
├── train.py              # Main training script
└── test_pipeline.py      # Testing script
```

## Quick Start

### Installation

```bash
conda create -n grasp python=3.10
conda activate grasp

# Core
pip install torch torchvision transformers qwen-vl-utils trimesh

# Geometry backbone (required)
pip install torch-geometric  # Use the official install command for your Torch/CUDA
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Data Preparation (render object views)

```bash
python dataset/prepare_data.py --oakink_root /path/to/OakInk --sample --num_samples 3
```

### Training

Train with OakInk dataset (7-way contacts: thumb/index/middle/ring/little/palm/no_contact):
```bash
python train.py --data_path /path/to/OakInk
```

## Architecture

The model combines:
- **Qwen2.5-VL** (3B): Vision-language understanding for semantic features
- **PointNet++**: Point cloud encoding for geometric features
- **Fusion Transformer**: Multimodal feature fusion
- **Contact Prediction**: 7-way finger/palm contact classification with pooling via `1 − p(no_contact)`
- **Flow Matching**: Conditional flow model for grasp pose generation

📊 **Visual Reference**: Original architecture sketches available in `assets/`

See `docs/pipeline.md` for detailed architecture documentation.
