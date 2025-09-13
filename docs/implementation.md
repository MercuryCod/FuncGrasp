# Implementation Guide & Repository Documentation

## Repository Status

### Current Structure
```
FuncGrasp/
├── models/               # Model implementations
│   ├── functional_grasp_model.py  # Main model integrating all components
│   ├── semantics_qwen.py          # Qwen2.5-VL wrapper
│   ├── pointnet2_encoder.py       # PointNet++ encoder
│   ├── fusion_transformer.py      # Multimodal fusion
│   ├── contact_head.py            # Contact prediction
│   └── flow_matching.py           # Conditional flow matching
├── data/                 # Data loaders
│   ├── oakink_loader.py           # Dataset implementation
│   └── prepare_oakink.py          # Data validation
├── docs/                 # Documentation
│   ├── pipeline.md                # Architecture design
│   ├── dataset.md                 # Dataset details
│   └── implementation.md          # This file
├── config.py             # Training configuration
├── train.py              # Training script
├── test_pipeline.py      # Testing script
├── README.md             # Project overview
└── CLAUDE.md             # AI assistant context
```

### Implementation Status ✓
- **Core Models**: All implemented and tested
- **Data Pipeline**: OakInk loader with contact approximation
- **Training Loop**: Complete with dual losses
- **Testing**: Comprehensive test suite passes
- **Documentation**: Architecture, dataset, and usage documented

## Quick Start

### Environment Setup
```bash
conda create -n grasp python=3.10
conda activate grasp

# Core
pip install torch torchvision transformers

# Geometry backbone (required)
pip install torch-geometric  # Use the official install command for your Torch/CUDA
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Extras
pip install tensorboard trimesh qwen-vl-utils einops wandb
```

### Testing Pipeline
```bash
python test_pipeline.py  # Tests with mock Qwen (no GPU needed)
```

### Training
```bash
python train.py --data_path ./OakInk --epochs 100
```

## Critical Implementation Details

### 1. Qwen2.5-VL Integration
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (3.75B params)
- **Freezing Options**:
  - `freeze_qwen=True` (default): Only projection layer trainable (15.4M params)
  - `freeze_qwen=False`: Full fine-tuning (3.77B params trainable)
- **Processing**: Joint image-text through chat template
- **Output**: Global semantic feature via masked pooling

**Fine-tuning Configuration** (when `freeze_qwen=False`):
```python
MODEL = {
    'freeze_qwen': False,  # Enable gradient flow through backbone
}
TRAINING = {
    'backbone_lr': 1e-5,   # Lower LR for pretrained weights
    'learning_rate': 1e-4, # Higher LR for new layers
    'batch_size': 1,       # Reduce for memory
    'gradient_accumulation': 8,  # Effective batch size
}
```
**Memory Requirements**:
- Frozen: ~0.2GB for gradients
- Unfrozen: ~42GB for gradients (requires 24GB+ GPU)

### 2. Baseline Pooling (Contact‑weighted)
**Current implementation uses contact‑weighted pooling:**
```python
logits_c = self.cm(f_fuse)              # [B, N, 1]
p = torch.sigmoid(logits_c)             # [B, N, 1]
w = p / (p.sum(dim=1, keepdim=True) + 1e-6)
z = (w * f_fuse).sum(dim=1)             # [B, CFUSE]
c = z                                   # Conditioning is pooled fused
```

**Optional variants:**
- **M.1 Plain mean pooling**: `z = f_fuse.mean(dim=1)`
- **M.2 Global geometry skip**: Concatenate `[z, s, g]` before conditioning
- **M.3 Input projection bottleneck**: Add an input projection in fusion transformer (not baseline)

### 3. Rectified Flow Matching
**Training**: Sample time uniformly, interpolate linearly
```python
x0 = torch.randn_like(target_pose)  # Start: noise
t = torch.rand(B)  # Time ~ U[0,1]
x_t = (1 - t) * x0 + t * target_pose  # Linear interpolation
v_pred = model.flow_step(x_t, t, conditioning)
v_target = target_pose - x0  # Constant velocity
loss = MSE(v_pred, v_target)
```

**Inference**: Integrate ODE with Euler method
```python
x = torch.randn(B, 28)  # Start from noise
for k in range(num_steps):
    t = (k + 0.5) / num_steps
    v = model.flow_step(x, t, c)
    x = x + (1/num_steps) * v
```

### 4. OakInk Adaptations
- **Language**: Generated from semantic attributes (e.g., "grasp the bottle to pour")
- **Contacts**: Approximated via 1cm hand-object proximity
- **Pose**: 28D = 3D wrist + 25 flattened relative joints

### 5. Geometry Backbone (PointNet++)
- Uses `torch_geometric.nn.models.PointNet2` (PyTorch Geometric) as the only PointNet++ implementation.
- This dependency is required. Install PyG matching your torch/CUDA.
- Returns `(F_geo [B,N,Cgeo], g [B,Cgeo])` as conditioning for fusion.

## Configuration Reference

### Key Dimensions (`config.py`)
```python
CSEM = 256      # Semantic features (from Qwen)
CGEO = 256      # Geometric features (from PointNet++)
CFUSE = CSEM + CGEO = 512  # Fused features (concatenated dimensions)
DPOSE = 28      # Pose: 3 wrist + 25 joints
N = 1024        # Points per object
```

### Training Parameters
```python
batch_size = 4 (GPU) / 2 (CPU)
learning_rate = 1e-4
epochs = 100
lambda_contact = 1.0  # Contact loss weight
lambda_flow = 1.0     # Flow loss weight
gradient_clip = 1.0
```

### CPU vs GPU
- **CPU Mode**: Auto-detected, reduces batch size and point count
- **GPU Mode**: Mixed precision (fp16), larger batches
- **Memory**: ~4GB for full model (3.5GB Qwen + 0.5GB others)

## Training Pipeline

### Data Flow
1. **Load**: Images, point clouds, hand joints from OakInk
2. **Process**: Generate prompts, compute contact labels
3. **Encode**: 
   - Images+Text → Qwen → Semantic features
   - Points → PointNet++ → Geometric features
4. **Fuse**: Transformer combines semantic and geometric
5. **Predict**: Contact points and grasp poses
6. **Loss**: Binary CE (contacts) + MSE (flow matching)

### Monitoring
```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Logged metrics:
# - Loss/total, Loss/contact, Loss/flow
# - Metrics/contact_accuracy
# - Validation/* (every 5 epochs)
```

### Checkpointing
- Saved every 500 batches to `./checkpoints/`
- Contains model, optimizer, step count
- Resume: `--checkpoint path/to/checkpoint.pt`

## API Usage

### Training
```python
from models import FunctionalGraspModel
from config import Config
from data import create_oakink_loaders

# Setup
cfg = Config.get_config(mode='train')
model = FunctionalGraspModel(**cfg['model'])
train_loader, val_loader = create_oakink_loaders(...)

# Train step
out = model.forward_train(images, texts, pts)
loss_contact = BCE(out['logits_c'], contact_labels)
loss_flow = compute_flow_matching_loss(model, out['cond'], poses)
```

### Inference
```python
# Load trained model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate grasp
with torch.no_grad():
    pose = model.sample(
        images=[PIL_image],
        texts=["grasp the bottle to pour"],
        pts=point_cloud,
        num_steps=20
    )
```

## Common Issues & Solutions

### Out of Memory
- Reduce `batch_size` or `n_points`
- Enable gradient accumulation
- Use CPU mode for development

### NaN Losses
- Check gradient clipping
- Reduce learning rate
- Verify data preprocessing

### Slow Training
- Ensure GPU usage: `nvidia-smi`
- Increase batch size if memory allows
- Reduce validation frequency

### Poor Convergence
- Adjust loss weights (`lambda_contact`, `lambda_flow`)
- Try learning rates from 1e-3 to 1e-5
- Verify Qwen loads correctly

## Development Guidelines

### Code Style
- **No comments** unless requested
- Follow existing patterns
- Use absolute imports
- Check dependencies before use

### Adding Features
1. Create module in appropriate directory
2. Update `__init__.py` 
3. Integrate in main model
4. Add tests in `test_pipeline.py`

### Testing Changes
```bash
# Always test after modifications
python test_pipeline.py

# Run specific component tests
python -c "from models import PointNet2Encoder; print('Import OK')"
```

## Important Reminders

1. **Qwen is frozen**: Only projection layer trains (15.6M params total)
2. **Contact labels are approximate**: 1cm threshold is heuristic
3. **Semantic features used twice**: In fusion AND conditioning
4. **Flow is rectified**: Linear interpolation, constant velocity
5. **Pose is 28D**: Wrist (3) + relative joints (25)

## Future Improvements

- **Multi-task**: Add affordance prediction
- **Temporal**: Sequence modeling for manipulation
- **Real robot**: Sim2real adaptation
- **Efficiency**: Model distillation
- **Data**: Active learning for sample selection

## Commands Reference

```bash
# Training variations
python train.py --data_path ./OakInk --batch_size 8 --device cuda:0
python train.py --checkpoint checkpoints/latest.pt  # Resume
python train.py --epochs 200 --learning_rate 5e-5  # Hyperparameters

# Data preparation
python data/prepare_oakink.py --root ./OakInk --validate

# Testing
python test_pipeline.py  # Quick test
python evaluate.py --checkpoint best_model.pt --split test  # Full eval
```

---

*For architecture details → `docs/pipeline.md`*  
*For dataset details → `docs/dataset.md`*  
*For AI context → `CLAUDE.md`*
