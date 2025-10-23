# FuncGrasp: Functional Grasping Dataset Loader

**Version**: 2.0
**Status**: ✅ Production Ready - Training Verified
**Last Updated**: October 2025

Unified PyTorch dataset loader for hand-object interaction with functional grasp annotations.

## Version 2.0 Status

**All critical fixes applied and verified**:
- ✅ Soft target normalization fixed (RBF kernel without normalization)
- ✅ Gradient contamination eliminated (detached pooling)
- ✅ Numerical stability improved (bfloat16 for Qwen)
- ✅ Input validation added (fast debugging)
- ✅ MANO normalization disabled (prioritizes cross-dataset generalization)

**Ready for training** - See [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) for architecture details.

## What This Provides

Load complete grasp data in one line:
- 📝 Text instructions (category + intent)
- ✋ MANO hand poses (pose + shape + translation)
- 🔷 Object point clouds (sampled & centered)
- 🎯 Contact labels (7-class finger-specific)
- 🖼️ Multi-view images (6 viewpoints, pre-rendered)
- 📊 Metadata (category, intent, IDs)
- 🚀 Lazy filtering for zero-contact samples (automatic during data loading)

**Current Dataset**: OakInk (51,177 grasps, 85 objects, 33 categories)

## Setup

### 1. Install Dependencies

```bash
pip install torch numpy trimesh matplotlib pillow tqdm chumpy
```

### 2. Setup MANO Model

**Critical**: Install manotorch in your training environment to avoid NaN losses:

```bash
# Install from the project's manotorch directory
cd /workspace/FuncGrasp/manotorch
pip install -e .

# Download MANO models from https://mano.is.tue.mpg.de/ (registration required)
# Place MANO_RIGHT.pkl and MANO_LEFT.pkl in manotorch/assets/mano_v1_2/models/
```

**Important**: If manotorch is not installed, contact computation will fail silently, causing:
- All `per_finger_distances` to be `inf`
- Invalid soft targets
- **NaN losses during training**

### 3. Pre-render Pose-Aligned Multi-View Images

**Required for training** - Generate grasp-centric multi-view images (~100K grasp instances):

```bash
cd /workspace/FuncGrasp
bash prepare/prepare_pose_aligned_renders.sh
```

This will:
- Render **pose-aligned views** where "front" = hand's approach direction
- Process ~100,000 grasp instances (train + val + test)
- Save to `/workspace/data/OakInk/rendered_objects_pose_aligned/`
- Take **14-28 hours** with 8 workers (7-14 hours with 16 workers)
- Use **~300GB** disk space

**Parallelization**: Adjust workers based on CPU cores:
```bash
NUM_WORKERS=16 bash prepare/prepare_pose_aligned_renders.sh
```

**Monitoring**: Track progress in another terminal:
```bash
watch -n 10 'cat /workspace/data/OakInk/rendered_objects_pose_aligned/render_summary.json'
```

**For details**, see [OakInk Dataset Documentation - Pose-Aligned Rendering](OAKINK_DATASET_DOCUMENTATION.md#12-pose-aligned-multi-view-rendering)




See **PIPELINE_DESIGN.md** for complete architecture and implementation details.

---

## Data Filtering

### Lazy Zero-Contact Filtering

Approach poses (no contact with object) are automatically skipped during data loading:

**Enable in config.py** (default: `True`):
```python
DATA = {
    'filter_zero_contact': True,  # Filters ~4% of samples with zero contact
}
```

**Implementation**:
- Dataset initializes instantly (~1 second, no pre-filtering overhead)
- When accessing `dataset[idx]`, contact is computed on-demand
- If sample has zero contact, automatically skips to next valid sample
- Max 10 attempts per access to find valid sample

**Performance**:
- Dataset init: ~1 second (vs 55+ minutes if pre-filtering)
- Per-sample overhead: <0.1ms
- Result: ~96% of original dataset (removes ~4% zero-contact samples)
- Benefit: Cleaner training signal without approach poses

## Model Architecture

**Functional Grasp Prediction**: Multi-modal grasp synthesis with contact-aware flow matching
- **Vision-Language Model**: Qwen3-VL-4B with **bfloat16** for numerical stability (LoRA fine-tuning)
- **Geometric Encoder**: PointNet++ for object point clouds
- **Grasp Generation**: Rectified flow for MANO pose synthesis
- **Contact Prediction**: 7-class finger-specific labels with soft targets

**Training verified on A100 80GB** - See [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) for full architecture details.

### Numerical Stability Note
The Qwen model uses `torch.bfloat16` instead of `float16` for better numerical stability:
- `bfloat16` has wider exponent range (8 bits vs 5 bits in float16)
- Prevents overflow/underflow in large language models
- Critical for avoiding NaN losses during training

---

## Documentation Map

📚 **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete navigation with quick links

Core documents:
- **This file (README.md)**: Quick start, setup, and v2.0 status
- **[PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)**: Complete architecture, v2.0 changes, soft targets, flow matching, and training loop
- **[OAKINK_DATASET_DOCUMENTATION.md](OAKINK_DATASET_DOCUMENTATION.md)**: Dataset structure, pose-aligned rendering, and implementation
- **[MANO_HAND_MODEL.md](MANO_HAND_MODEL.md)**: Hand model, FK, contact computation, and rendering
- **[QWEN3VL_DOCUMENTATION.md](QWEN3VL_DOCUMENTATION.md)**: Qwen3-VL model architecture, API, and integration

---

## Documentation Guideline

### Core Principles
1. **Minimal documentation**: Create new docs only when necessary; prefer editing existing docs
2. **Always up-to-date**: Update docs immediately when code changes to maintain accuracy
3. **Separation of concerns**: Each doc focuses on a single area comprehensively
4. **README as map**: This file serves as a navigation guide, not a comprehensive reference

### Document Structure
- **README.md** (this file): Quick start, training, and navigation
- **PIPELINE_DESIGN.md**: Architecture, soft targets, losses, training loop, and debugging
- **OAKINK_DATASET_DOCUMENTATION.md**: Dataset structure, pose-aligned rendering, and conventions
- **MANO_HAND_MODEL.md**: MANO model, FK, and rendering
- **QWEN3VL_DOCUMENTATION.md**: Qwen3-VL model architecture and API
- **DOCUMENTATION_INDEX.md**: Navigation hub with quick links

### When to Create New Docs
- ✅ New dataset integration (create `<DATASET>_DOCUMENTATION.md`)
- ✅ New model architecture (create `<MODEL>_ARCHITECTURE.md`)
- ✅ Complex subsystem requiring detailed explanation
- ❌ Implementation details (belong in code comments)
- ❌ Temporary notes (use TODO.md instead)
- ❌ Step-by-step guides (belong in README examples)
