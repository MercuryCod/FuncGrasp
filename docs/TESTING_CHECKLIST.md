# Testing Checklist for Contact Accuracy Improvements

## Status
**Implementation Complete** - Awaiting data access restoration for testing

## Implemented Components

### Phase 1: Config + Switches ✅
- Added `MODEL.contact_regression` and `MODEL.inference_threshold`
- Added `TRAINING.contact_loss_type` and `TRAINING.pos_weight`
- Added `REGRESSION_HPARAMS` dict with label_type, tau_mm, etc.
- Added `EVAL.num_qualitative` and `EVAL.save_gt_qualitative`
- Added `DATA.contact_aware_sampling` and related params
- Added CLI arguments for all new flags in train.py

### Phase 2: Dataset Soft Targets ✅
- Implemented `_compute_soft_contact_targets()` with vectorized distance computation
- Added Gaussian and logistic target mapping
- Updated `__getitem__()` to return both `contact_labels` and `contact_targets`
- Added `hand_vertices` to `batch['meta']` for visualization
- Updated collate function to handle `contact_targets`
- Dataset accepts `use_soft_targets` and `regression_hparams`

### Phase 3: Model Pooling ✅
- Added `contact_regression` and `inference_threshold` parameters to model
- Implemented branching pooling logic in `forward_backbone()`:
  - Regression: `w = max(sigmoid(parts))`
  - Classification: `w = 1 - p(no_contact)`
- Stored config as instance attributes for checkpoint compatibility

### Phase 4: Training Loss + Metrics ✅
- Added `compute_contact_predictions()` helper with threshold logic
- Added `compute_contact_metrics()` for per-class acc, macro-F1, confusion matrix
- Implemented BCE loss path in `train_one_epoch()` with pos_weight support
- Updated validation to compute and log enhanced metrics
- Logging includes per-class accuracy and confusion matrix

### Phase 5: Contact-Aware Sampling ⏸️
- **Not yet implemented** - deferred as optional enhancement

## Testing Tasks (When Data Access Restored)

### 1. Config Loading
```bash
python -c "from config import Config; cfg = Config.get_config(); print(cfg['model']['contact_regression'])"
```
- [ ] Verify all new config keys load correctly
- [ ] Test CLI overrides: `--contact_regression`, `--tau_mm`, etc.

### 2. Dataset Soft Targets
```bash
python -c "from dataset.oakink_loader import OakInkDataset; ds = OakInkDataset(...); batch = ds[0]; print(batch.keys())"
```
- [ ] Verify `contact_targets` shape is `[1, N, 7]`
- [ ] Verify soft targets sum approximately to 1 per point
- [ ] Check `hand_vertices` in `batch['meta']`
- [ ] Test caching: run twice, second should be faster

### 3. Model Forward Pass
```python
model = FunctionalGraspModel(contact_regression=True, inference_threshold=0.4)
# Test with dummy batch
out = model.forward_train(images_list, texts_list, pts)
print(out['logits_c'].shape)  # Should be [B, N, 7]
```
- [ ] Verify model initializes with new parameters
- [ ] Test both regression and classification pooling modes
- [ ] Check conditioning vector shape

### 4. Training Loop
```bash
python train.py --epochs 1 --batch_size 2 --contact_regression --contact_loss_type bce
```
- [ ] Verify BCE loss computes without errors
- [ ] Check training logs show correct predictions
- [ ] Verify no NaN/Inf in losses
- [ ] Test CE fallback: `--no_contact_regression --contact_loss_type ce`

### 5. Validation Metrics
- [ ] Run validation and check run.log for:
  - Per-class accuracy (7 values)
  - Macro-F1 score
  - 7×7 confusion matrix
- [ ] Verify TensorBoard logs all metrics
- [ ] Check metrics make sense (no negative values, etc.)

### 6. End-to-End Training
```bash
python train.py --epochs 5 --batch_size 4 --log_interval 10
```
- [ ] Train for 5 epochs without crashes
- [ ] Monitor for memory leaks
- [ ] Verify checkpoints save correctly
- [ ] Check validation runs every 5 epochs

### 7. Backward Compatibility
```bash
python train.py --no_contact_regression --contact_loss_type ce --epochs 1
```
- [ ] Verify CE mode still works
- [ ] Load old checkpoints (if any exist)
- [ ] Ensure no breaking changes to existing pipeline

## Known Issues / Notes
- Data access currently unavailable (system issue)
- Class frequency computation not yet implemented (pos_weight will be None initially)
- Qualitative visualization not yet implemented
- Contact-aware sampling deferred to Phase 5

## Next Steps After Testing
1. Compute and cache class frequencies from training data
2. Set `pos_weight` in config based on frequencies
3. Implement qualitative visualization in validation
4. (Optional) Implement contact-aware sampling
5. Run ablation studies comparing CE vs BCE
