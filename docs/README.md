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
pip install torch numpy trimesh matplotlib pillow tqdm chumpy
```

### 2. Setup MANO Model

```bash
# Clone and install manotorch
git clone https://github.com/lixiny/manotorch.git
cd manotorch
pip install .

# Download MANO models from https://mano.is.tue.mpg.de/ (registration required)
# Place MANO_RIGHT.pkl and MANO_LEFT.pkl in manotorch/assets/mano_v1_2/models/
```

### 3. Pre-render Object Images (Optional)

For training models that use object images, pre-render all 85 unique objects once:

```bash
cd /workspace/FuncGrasp
bash prepare/prepare_renders.sh
```

This will:
- Render all objects from 6 viewpoints (front, back, left, right, top, bottom)
- Save to `/workspace/data/OakInk/rendered_objects/`
- Take ~2 minutes
- Use ~20MB disk space




See **PIPELINE_DESIGN.md** for complete architecture and implementation details.

---

## Model Architecture

**Functional Grasp Prediction**: Multi-modal grasp synthesis with contact-aware flow matching
- **Vision-Language Model**: Qwen2.5-VL-3B (LoRA fine-tuning)
- **Geometric Encoder**: PointNet++ for object point clouds
- **Grasp Generation**: Rectified flow for MANO pose synthesis
- **Contact Prediction**: 7-class finger-specific labels with soft targets

**Training verified on A100 80GB** - See [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) for full architecture details.

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

## Docuemntation Guideline

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
