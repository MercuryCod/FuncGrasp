# FuncGrasp: Functional Grasping Dataset Loader

Unified PyTorch dataset loader for hand-object interaction with functional grasp annotations.

## What This Provides

Load complete grasp data in one line:
- 📝 Text instructions (category + intent)
- ✋ MANO hand poses (pose + shape + translation)
- 🔷 Object point clouds (sampled & centered)
- 🎯 Contact labels (7-class finger-specific)
- 🖼️ Multi-view images (6 viewpoints, pre-rendered)
- 📊 Metadata (category, intent, IDs)

**Current Dataset**: OakInk (51,177 grasps, 85 objects, 33 categories)

## Setup

### 1. Install Dependencies

```bash
pip install torch numpy trimesh matplotlib pillow tqdm
```

### 2. Pre-render Object Images (Optional)

For training models that use object images, pre-render all 68 unique objects once:

```bash
cd /workspace/FuncGrasp
bash prepare/prepare_renders.sh
```

This will:
- Render all objects from 6 viewpoints (front, back, left, right, top, bottom)
- Save to `/workspace/data/OakInk/rendered_objects/`
- Take ~2 minutes
- Use ~20MB disk space

**Custom rendering:**
```bash
python prepare/prerender_objects.py \
    --oakink_root /path/to/OakInk \
    --image_size 512 \
    --zoom 1.6 \
    --force  # Re-render existing objects
```

## Usage

### Basic Usage

```python
from dataset.dataset import OakInkDataset

# Load dataset
dataset = OakInkDataset(
    data_root='/workspace/data/OakInk',
    split='train',  # 'train', 'val', or 'test'
    num_points=1024,
    compute_contacts=True,  # Pre-compute contact labels
)

# Get a sample
sample = dataset[0]
print(sample.keys())
# dict_keys(['instruction', 'mano_pose', 'mano_shape', 'mano_trans', 
#            'obj_points', 'category', 'shape_id', 'subject_id', 
#            'action_id', 'intent', 'contact_labels', 'contact_mask', 
#            'contact_distances'])
```

### With Pre-rendered Images

```python
# Load dataset with pre-rendered object images
dataset = OakInkDataset(
    data_root='/workspace/data/OakInk',
    split='train',
    load_object_images=True,  # Load pre-rendered images
)

sample = dataset[0]
print(sample['object_images'].keys())
# dict_keys(['front', 'back', 'left', 'right', 'top', 'bottom'])

# Each image is a (H, W, 3) tensor in range [0, 1]
front_view = sample['object_images']['front']  # (512, 512, 3)
```

### Computing Hand Geometry

```python
from utils.mano_utils import mano_forward

# Compute hand joints and vertices from MANO parameters
joints, vertices, faces = mano_forward(
    sample['mano_pose'],
    sample['mano_shape'],
    sample['mano_trans']
)
# joints: (21, 3) - 21 hand joint positions
# vertices: (778, 3) - hand mesh vertices
# faces: (1538, 3) - hand mesh faces
```

### DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in dataloader:
    instructions = batch['instruction']  # List of strings
    hand_poses = batch['mano_pose']      # (B, 48)
    obj_points = batch['obj_points']     # (B, N, 3)
    contact_labels = batch['contact_labels']  # (B, N)
    
    if 'object_images' in batch:
        front_views = batch['object_images']['front']  # (B, 512, 512, 3)
```

## Quick Stats

- **51,177 grasps** across train/val/test splits
- **85 unique objects** | 33 categories | 4 intents
- **Splits**: ~76% train / 9% val / 15% test (by object MD5 hash)
- **Contact labels**: 7 classes (no contact + palm + 5 fingers)
- **Images**: 512×512 RGB, 6 viewpoints per object, pre-rendered

---

## Training

### Quick Start
```bash
conda activate grasp
cd /workspace/FuncGrasp
python train.py  # Starts training with default config
```

### Configuration
```bash
EXP_NAME=my_exp python train.py          # Custom experiment name
QWEN_TUNING=frozen python train.py       # Freeze Qwen (fastest)
DEBUG=true python train.py               # Debug mode (frequent logging)
```

### Testing
Debug mode for quick validation:
```bash
DEBUG=true bash scripts/train.sh debug_test
# Runs with: 1-step logging, 5-step checkpoints, 50 validation samples
```

See **PIPELINE_DESIGN.md** for complete architecture and implementation details.

---

## Recent Updates

### 2024-10-16 Critical Bugs Fixed & Training Verified
- ✅ **Full pipeline implemented and training successfully**
- ✅ **CRITICAL FIX**: Visualization now shows actual predicted poses (not GT)
- ✅ **CRITICAL FIX**: Validation computes both contact AND flow losses
- ✅ Per-class metrics aligned with multi-label BCE training objective
- ✅ Qualitative samples from val set (not test) to avoid data leakage
- ✅ Global seeding for reproducibility (seed=42)
- ✅ Coordinate frame verified: both poses in object frame (object-centered)
- ✅ Distance-based soft contact targets with Gaussian kernel (τ=10mm)
- ✅ Multi-label BCE loss with contact-strength weighting
- ✅ Rectified flow for pose generation (20-step Euler integration)
- ✅ Step-based validation with per-class accuracy metrics
- ✅ Debug mode with faster validation (50 samples vs 4714)
- ✅ Interval-based logging (clean log files, no tqdm)
- ✅ All 85 objects pre-rendered (train+val+test splits)

### Training Verified Working
- Model: Qwen2.5-VL-3B (LoRA) + PointNet++ + Flow Matching
- Parameters: 3.8B total, 90M trainable (2.35%)
- Speed: ~5s/batch on A100 80GB (validation: ~2s/batch)
- Logs: `exp/{EXP_NAME}/logs/run.log`
- Checkpoints: Auto-saved every N steps + best F1
- Visualization: HTML files with predicted vs GT poses

---

## Documentation Map

📚 **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete navigation with quick links

Core documents:
- **This file (README.md)**: Quick start and navigation
- **PIPELINE_DESIGN.md**: Complete architecture, soft targets, flow matching, and training loop
- **OAKINK_DATASET_DOCUMENTATION.md**: Dataset structure and implementation status
- **MANO_HAND_MODEL.md**: Hand model, FK, contact computation, and rendering
- **QWEN25VL_DOCUMENTATION.md**: Qwen2.5-VL model architecture, API, and integration

---

## Documentation Guidelines

### Core Principles
1. **Minimal documentation**: Create new docs only when necessary; prefer editing existing docs
2. **Always up-to-date**: Update docs immediately when code changes to maintain accuracy
3. **Separation of concerns**: Each doc focuses on a single area comprehensively
4. **README as map**: This file serves as a navigation guide, not a comprehensive reference

### Document Structure
- **README.md** (this file): Quick start, training, and navigation
- **PIPELINE_DESIGN.md**: Architecture, soft targets, losses, training loop, and debugging
- **OAKINK_DATASET_DOCUMENTATION.md**: Dataset structure and conventions
- **MANO_HAND_MODEL.md**: MANO model, FK, and rendering
- **QWEN25VL_DOCUMENTATION.md**: Qwen2.5-VL model architecture and API
- **DOCUMENTATION_INDEX.md**: Navigation hub with quick links

### When to Create New Docs
- ✅ New dataset integration (create `<DATASET>_DOCUMENTATION.md`)
- ✅ New model architecture (create `<MODEL>_ARCHITECTURE.md`)
- ✅ Complex subsystem requiring detailed explanation
- ❌ Implementation details (belong in code comments)
- ❌ Temporary notes (use TODO.md instead)
- ❌ Step-by-step guides (belong in README examples)
