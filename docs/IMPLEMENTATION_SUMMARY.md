# Contact Accuracy Improvement - Implementation Summary

## Overview
Successfully implemented class-based regression for contact prediction to address the 85-88% validation accuracy plateau. The implementation switches from hard 7-way classification (CE) to soft, distance-shaped targets trained with BCE-with-logits, while maintaining backward compatibility with the original CE approach.

## What Was Implemented

### 1. Configuration System (`config.py`)
- **New model flags**: `contact_regression` (bool), `inference_threshold` (float)
- **New training flags**: `contact_loss_type` ('bce'|'ce'), `pos_weight` (list)
- **Regression hyperparameters**: `label_type`, `tau_mm`, `t_mm`, `s_mm`, `clamp_radius_factor`
- **Evaluation flags**: `num_qualitative`, `save_gt_qualitative`
- **Data flags**: `contact_aware_sampling`, `contact_aware_ratio`, `contact_aware_radius_factor`
- **CLI support**: All flags can be overridden via command-line arguments

### 2. Dataset Soft Targets (`dataset/oakink_loader.py`)
- **`_compute_soft_contact_targets()` method**:
  - Vectorized distance computation using `trimesh.proximity.closest_point()`
  - Gaussian mapping: `s_j = exp(-(d_j/tau)^2)`
  - Logistic mapping: `s_j = sigmoid((t-d_j)/s)`
  - No-contact score: `s_nc = 1 - max(part_scores)`
  - Distance clamping beyond configurable radius
- **Batch structure updates**:
  - Returns both `contact_labels` [B,N] (for CE) and `contact_targets` [B,N,7] (for BCE)
  - Added `hand_vertices` to `batch['meta']` for visualization
- **Collate function**: Updated to handle optional `contact_targets`

### 3. Model Pooling (`models/functional_grasp_model.py`)
- **Constructor changes**: Added `contact_regression` and `inference_threshold` parameters
- **Branching pooling logic** in `forward_backbone()`:
  ```python
  if self.contact_regression:
      # Regression: use max part probability
      part_probs = torch.sigmoid(logits_c[..., :6])
      w = torch.max(part_probs, dim=-1)[0]
  else:
      # Classification: use 1 - p(no_contact)
      probs = logits_c.softmax(dim=-1)
      w = 1.0 - probs[..., 6]
  ```
- **Checkpoint compatibility**: Flags stored as instance attributes

### 4. Training & Metrics (`train.py`)
- **Helper functions**:
  - `compute_contact_predictions()`: Applies threshold logic for regression mode
  - `compute_contact_metrics()`: Computes per-class accuracy, macro-F1, confusion matrix
  
- **Loss computation** in `train_one_epoch()`:
  ```python
  if cfg['training']['contact_loss_type'] == 'bce':
      loss_contact = F.binary_cross_entropy_with_logits(
          logits_c, contact_targets, pos_weight=pos_weight
      )
  else:
      loss_contact = F.cross_entropy(
          logits_c.view(B*N, 7), contact_labels.view(B*N)
      )
  ```

- **Enhanced validation**:
  - Computes and logs per-class accuracy for all 7 classes
  - Computes macro-F1 score
  - Generates 7×7 confusion matrix
  - Logs to both `run.log` and TensorBoard

## Key Design Decisions

### 1. Backward Compatibility
- Original CE path fully preserved behind `contact_loss_type='ce'` flag
- Model can switch between modes via config
- Existing checkpoints remain loadable

### 2. Soft Target Computation
- Computed on-the-fly (caching deferred for initial testing)
- Vectorized per-part distance queries for efficiency
- Configurable distance-to-score mapping (Gaussian or logistic)

### 3. Inference Threshold
- Configurable θ for no_contact prediction in regression mode
- Default: 0.4 (can be tuned on validation)
- Prediction rule: `max_part_prob ≤ θ → no_contact, else argmax(parts)`

### 4. Pooling Weight Change
- Regression mode uses `max(sigmoid(parts))` as contactness
- More direct measure of contact confidence than `1 - p(no_contact)`
- Affects conditioning vector for flow matching

## What's Deferred (Pending Testing)

### 1. Class Frequency Computation
- Need data access to compute train-set class frequencies
- Will compute `pos_weight = 1/sqrt(freq)` after testing
- Currently `pos_weight=None` (uniform weighting)

### 2. Soft Target Caching
- Currently computed on-the-fly each epoch
- Will add disk caching keyed by `(frame_id, label_type, tau_mm, threshold)`
- Expected speedup: ~2-3x after first epoch

### 3. Qualitative Visualization
- Framework ready (`num_qualitative`, `save_gt_qualitative` flags)
- Will generate pred/GT pairs showing:
  - Object points colored by contact class
  - Predicted hand pose from `model.sample()`
  - Ground-truth hand pose and vertices
- Saves to `outputs/qualitative/`

### 4. Contact-Aware Sampling (Optional)
- Config flags added but sampling not implemented
- Would mix near-hand points (within 2× threshold) with uniform points
- Expected benefit: more informative gradients for contact classes

### 5. Visualization Script Updates
- `visualize_training.py` needs updates to plot:
  - Macro-F1 trends
  - Per-class accuracy over time
  - Confusion matrix evolution

## Testing Plan
See `docs/TESTING_CHECKLIST.md` for detailed testing tasks once data access is restored.

### Critical Tests:
1. **Config loading**: Verify all flags load and override correctly
2. **Dataset**: Check `contact_targets` shape and values
3. **Model forward**: Test both regression and classification modes
4. **Training loop**: Verify BCE loss computes without errors
5. **Validation**: Check enhanced metrics in logs
6. **End-to-end**: Train for 5 epochs without crashes

## Expected Benefits

### 1. Reduced Boundary Noise
- Soft targets smooth label flips near the 8mm threshold
- Gradients more informative for borderline points

### 2. Better Calibration
- Per-channel BCE provides independent probability estimates
- No forced competition through softmax

### 3. Class Imbalance Mitigation
- `pos_weight` directly addresses minority class under-fitting
- Per-channel weighting more flexible than class-weighted CE

### 4. Improved Metrics Visibility
- Per-class accuracy reveals finger/palm performance
- Macro-F1 prevents no_contact dominance
- Confusion matrix shows specific misclassification patterns

## Usage Examples

### Train with regression (default):
```bash
python train.py --epochs 10 --batch_size 4
```

### Train with classification (fallback):
```bash
python train.py --no_contact_regression --contact_loss_type ce --epochs 10
```

### Override regression parameters:
```bash
python train.py --tau_mm 10.0 --inference_threshold 0.35
```

### Test both modes:
```bash
# Regression
python train.py --contact_regression --contact_loss_type bce --epochs 1

# Classification
python train.py --no_contact_regression --contact_loss_type ce --epochs 1
```

## Files Modified

### Core Implementation:
- `config.py`: Added 15+ new configuration parameters
- `dataset/oakink_loader.py`: Added soft target computation (~90 lines)
- `models/functional_grasp_model.py`: Updated pooling logic (~20 lines)
- `train.py`: Added helpers, BCE loss path, enhanced metrics (~150 lines)

### Documentation:
- `docs/CONTACT_ACCURACY_IMPROVEMENT.md`: Implementation plan
- `docs/TESTING_CHECKLIST.md`: Testing tasks
- `docs/IMPLEMENTATION_SUMMARY.md`: This file

## Next Steps
1. **Restore data access** and run testing checklist
2. **Compute class frequencies** from training data
3. **Set pos_weight** in config based on frequencies
4. **Run initial training** for 5-10 epochs
5. **Compare CE vs BCE** performance
6. **Implement qualitative visualization**
7. **Tune inference threshold** θ on validation
8. **(Optional) Add contact-aware sampling**

## Notes
- Implementation is complete and ready for testing
- All changes are backward-compatible
- No breaking changes to existing pipeline
- Data access issues prevent immediate testing
