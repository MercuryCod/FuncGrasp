# FuncGrasp Quick Start Guide

## Complete Workflow (3 Steps)

### Step 1: Data Preparation (one-time)
Render object meshes to multi-view images:
```bash
bash scripts/prepare_data.sh
```
**Output**: `rendered_objects/` directory with PNG images

---

### Step 2: Training Setup (one-time)
Compute class frequencies for pos_weight:
```bash
bash scripts/setup_training.sh
```
**Output**: `/mnt/data/changma/OakInk/cache/class_frequencies_train.json`

**What you'll see**:
```
Class           Count           Frequency       pos_weight     
thumb           24343           0.029827        0.8791         
index           15701           0.019238        1.0946         
middle          8215            0.010066        1.5133         
ring            13402           0.016421        1.1848         
little          23107           0.028313        0.9023         
palm            11773           0.014425        1.2641         
no_contact      719587          0.881709        0.1617         
```

---

### Step 3: Training
Train the model with contact regression:
```bash
bash scripts/train.sh
```

**Default settings**:
- Contact regression (BCE) mode
- LoRA fine-tuning of Qwen2.5-VL
- Batch size: 64
- Epochs: 10
- Auto-loads pos_weight from cache

**Customize**:
```bash
# Change batch size
BATCH_SIZE=32 bash scripts/train.sh

# More epochs
EPOCHS=20 bash scripts/train.sh

# Use classification mode (CE) instead
CONTACT_REGRESSION=false bash scripts/train.sh

# Adjust Gaussian tau parameter
TAU_MM=12.0 bash scripts/train.sh
```

---

## Monitoring Training

### View logs in real-time:
```bash
tail -f logs/run.log
```

### TensorBoard:
```bash
tensorboard --logdir logs/
```

### Generate plots:
```bash
python visualize_training.py
```
**Output**: `outputs/training_metrics.png` with 6 subplots:
1. Total loss (train + val)
2. Contact loss (train + val)
3. Flow loss (train + val)
4. Contact accuracy (train + val) with macro-F1 overlay
5. Macro-F1 trend (validation)
6. Per-class accuracy (7 color-coded curves)

---

## What's Happening During Training

### Enhanced Metrics Logged:
- **Overall accuracy**: Traditional point-wise accuracy
- **Macro-F1**: Class-balanced F1 score (prevents no_contact dominance)
- **Per-class accuracy**: Separate accuracy for each finger/palm/no_contact
- **Confusion matrix**: 7×7 matrix showing misclassification patterns

### Qualitative Visualizations:
Every validation, 3 samples are visualized:
- `outputs/qualitative/val_step_*_sample_*_pred.png` - Predicted contact map + pose
- `outputs/qualitative/val_step_*_sample_*_gt.png` - Ground truth contact map + pose

### Soft Target Caching:
- First epoch: Slow (computes soft targets)
- Later epochs: Fast (loads from cache)
- Cache location: `/mnt/data/changma/OakInk/cache/<split>/soft_targets/`

---

## Troubleshooting

### "No rendered images found"
→ Run `bash scripts/prepare_data.sh` first

### "pos_weight is None in logs"
→ Run `bash scripts/setup_training.sh` first

### "Slow first epoch"
→ Normal! Soft targets are being computed and cached

### Compare BCE vs CE performance
```bash
# BCE (regression) - better for minority classes
CONTACT_REGRESSION=true bash scripts/train.sh

# CE (classification) - simpler baseline
CONTACT_REGRESSION=false bash scripts/train.sh
```

---

## File Structure

```
FuncGrasp/
├── compute_class_frequencies.py    # Compute class imbalance stats
├── scripts/
│   ├── prepare_data.sh              # Step 1: Render objects
│   ├── setup_training.sh            # Step 2: Compute frequencies
│   └── train.sh                     # Step 3: Train model
├── visualize_training.py            # Generate plots
├── docs/
│   ├── CONTACT_ACCURACY_IMPROVEMENT.md  # Full implementation details
│   ├── WORKFLOW.md                      # Detailed workflow guide
│   └── ...
└── outputs/
    ├── training_metrics.png         # Training plots
    └── qualitative/                 # Visualization samples
```

---

## Expected Results

### Class Frequency Analysis:
- ~88% no_contact (expected - most surface points don't touch hand)
- ~1-3% per finger/palm (minority classes)
- pos_weight rebalances: minority classes get higher weight (0.9-1.5), no_contact gets lower (0.16)

### Training Progress:
- Contact loss should decrease steadily
- Overall accuracy: expect 85-90%+
- Macro-F1: target 0.4-0.6+ (better than CE baseline)
- Per-class accuracy: fingers/palm should improve with BCE

### Qualitative Samples:
- Pred visualizations show model's contact predictions and generated pose
- GT visualizations show ground truth for comparison
- Color-coded by contact class (red=thumb, orange=index, etc.)

---

For full implementation details, see `docs/CONTACT_ACCURACY_IMPROVEMENT.md`.

