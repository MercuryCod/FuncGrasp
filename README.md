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
│   ├── contact_head.py            # Contact prediction head
│   └── flow_matching.py           # Conditional flow matching for poses
├── data/                 # Data loaders and utilities
│   ├── oakink_loader.py           # OakInk dataset loader
│   └── prepare_oakink.py          # Data preparation script
├── docs/                 # Documentation
│   ├── pipeline.md                # Architecture and training details
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
pip install torch torchvision transformers
```

### Testing

Test the pipeline with dummy data:
```bash
python test_pipeline.py
```

### Training

Train with OakInk dataset:
```bash
python train.py --data_path /path/to/OakInk
```

## Architecture

The model combines:
- **Qwen2.5-VL** (3B): Vision-language understanding for semantic features
- **PointNet++**: Point cloud encoding for geometric features
- **Fusion Transformer**: Multimodal feature fusion
- **Contact Prediction**: Binary contact point prediction
- **Flow Matching**: Conditional flow model for grasp pose generation

📊 **Visual Reference**: Original architecture sketches available in `assets/`

See `docs/pipeline.md` for detailed architecture documentation.