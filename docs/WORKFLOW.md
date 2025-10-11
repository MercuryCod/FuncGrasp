# FuncGrasp Training Workflow

## Overview
This document explains the complete workflow from data preparation to training with the contact accuracy improvements.

## Workflow Steps

### 1. Data Preparation (ONE-TIME)
**Purpose**: Render object meshes to multi-view images

```bash
bash scripts/prepare_data.sh
```

**What it does**:
- Loads object meshes from OakInk dataset
- Renders 6 views per object (front, back, left, right, top, bottom)
- Saves to `rendered_objects/` directory
- This is a one-time setup

**Output**: `rendered_objects/` with PNG images for each object

---

### 2. Training Setup (ONE-TIME before first training)
**Purpose**: Compute class imbalance statistics for pos_weight

```bash
bash scripts/setup_training.sh
```

**What it does**:
- Runs `compute_class_frequencies.py` (root level)
- Loads the training dataset (using already-prepared rendered images)
- Iterates through all training samples
- Counts contact labels per class (thumb, index, middle, ring, little, palm, no_contact)
- Computes `pos_weight = 1/sqrt(frequency)` to reweight BCE loss
- Saves to `/mnt/data/OakInk/cache/class_frequencies_train.json`

**Output**: 
- `cache/class_frequencies_train.json` with class statistics
- Console output showing class distribution

**When to re-run**:
- If contact_threshold changes
- If you modify the training split
- Otherwise, only run once

---

### 3. Training
**Purpose**: Train the model with contact regression

```bash
bash scripts/train.sh
```

**What it does**:
- Automatically loads `pos_weight` from cache (if available)
- Trains with class-based regression (BCE) by default
- Computes and caches soft contact targets per frame
- Logs enhanced metrics (per-class accuracy, macro-F1, confusion matrix)
- Generates qualitative visualizations every validation
- Saves checkpoints periodically

**Customization**:
```bash
# Change batch size
BATCH_SIZE=32 bash scripts/train.sh

# Change epochs
EPOCHS=20 bash scripts/train.sh

# Use classification mode instead of regression
CONTACT_REGRESSION=false bash scripts/train.sh

# Adjust tau parameter
TAU_MM=12.0 bash scripts/train.sh
```

**Output**:
- `checkpoints/*.pt` - Model checkpoints
- `logs/run.log` - Training logs with enhanced metrics
- `outputs/qualitative/*.png` - Pred/GT visualization pairs
- TensorBoard logs in `logs/`

---

### 4. Visualization
**Purpose**: Plot training curves and metrics

```bash
python visualize_training.py
```

**What it does**:
- Parses `logs/run.log`
- Generates 6-subplot figure with:
  - Total loss, contact loss, flow loss
  - Contact accuracy with macro-F1 overlay
  - Macro-F1 dedicated plot
  - Per-class accuracy trends (color-coded by finger/palm)

**Output**: `outputs/training_metrics.png`

---

## Script Purposes Summary

| Script | Purpose | When to Run | Output |
|--------|---------|-------------|--------|
| `prepare_data.sh` | Render object images | Once (data prep) | `rendered_objects/` |
| `setup_training.sh` | Compute class frequencies | Once before training | `cache/class_frequencies_train.json` |
| `train.sh` | Train model | Every training run | Checkpoints, logs, visualizations |
| `visualize_training.py` | Plot metrics | After/during training | `training_metrics.png` |

## Key Differences: prepare_data.sh vs setup_training.sh

### `prepare_data.sh`:
- **Stage**: Data preparation (before any training)
- **Input**: Raw OakInk meshes
- **Output**: Rendered object images
- **Frequency**: Run once per dataset
- **Dependencies**: None (just needs meshes)

### `setup_training.sh`:
- **Stage**: Training preparation (after data prep, before training)
- **Input**: Prepared dataset with rendered images
- **Output**: Class frequency statistics
- **Frequency**: Run once, or when contact threshold changes
- **Dependencies**: Requires `prepare_data.sh` to have run first

### `compute_class_frequencies.py` (root level):
- **Purpose**: Analyze training data distribution
- **NOT data preparation** - it reads already-prepared data
- **Loads dataset** to count contact labels
- **Computes pos_weight** for BCE loss
- **Part of training setup**, not data preparation
- **Called by**: `scripts/setup_training.sh`

## Complete Workflow Example

```bash
# First time setup (do once):
bash scripts/prepare_data.sh          # Render objects (slow, ~hours)
bash scripts/setup_training.sh        # Compute class frequencies (fast, ~minutes)

# Training:
bash scripts/train.sh                 # Train model (hours/days)

# Monitoring:
python visualize_training.py          # Generate plots anytime
tensorboard --logdir logs/            # Real-time monitoring
```

## Troubleshooting

### "No rendered images found"
→ Run `bash scripts/prepare_data.sh` first

### "pos_weight is None in training logs"
→ Run `bash scripts/setup_training.sh` first

### "Slow first epoch, then faster"
→ Normal! Soft targets are cached after first computation

### "Want to compare CE vs BCE"
```bash
# BCE (default)
CONTACT_REGRESSION=true bash scripts/train.sh

# CE (classification)
CONTACT_REGRESSION=false bash scripts/train.sh
```

