## Contact accuracy plateau: problem, approach, and implementation plan

> **Status**: ✅ Implementation complete. Ready for testing with data at `/mnt/data/OakInk`.  
> **Quick start**: Run `bash scripts/setup_training.sh` then `bash scripts/train.sh`.  
> **Documentation**: This is the single source of truth for the contact accuracy improvement pipeline.

### Problem
- Validation contact accuracy saturates around 85–88% while training accuracy keeps rising. Total/flow losses continue to improve, so the bottleneck is specific to the contact head/data rather than overall optimization.

### Likely root causes in the pipeline
- **Class imbalance**: The majority of surface points are labeled as `no_contact`, so overall accuracy improves quickly while minority finger/palm classes stall.
- **Boundary label noise**: A hard 1 cm proximity threshold on MANO meshes flips labels for borderline points; small geometry/registration errors exacerbate this.
- **Uniform/FPS point sampling**: Uniform surface sampling over‑represents far‑from‑hand regions, making `no_contact` easy and limiting useful signal for contact parts.
- **Limited diversity**: Training currently keeps only the last frame per sequence and a single rendered view, reducing variation in contact patches and conditioning.
- **Loss balance**: With \(\lambda_{contact}=\lambda_{flow}\), the model may allocate capacity to flow after `no_contact` becomes easy, under‑optimizing the contact head.
- **Metric masking**: Point‑wise accuracy is dominated by the `no_contact` class; per‑class performance (especially fingers/palm) can stagnate without visibility.

## Proposed approach

### Metrics and reporting
- **Add per‑class accuracy, macro‑F1, and a 7×7 confusion matrix** to validation output and run.log.
- **Keep run.log as the single source of truth** and provide a log visualization script (`visualize_training.py`) to plot train/val trends.

### Loss shaping
- **Switch to class‑based regression (primary path)**: train the 7 logits with **BCE‑with‑logits** against soft, distance‑shaped targets.
- **Keep CE as a fallback**: retain 7‑class cross‑entropy behind a config flag for A/B comparisons.
- Optional: **Focal loss** for contact if minority classes remain under‑fit.
- **Schedule \(\lambda_{contact}/\lambda_{flow}\)**: temporarily increase \(\lambda_{contact}\) (e.g., 2.0) or reduce \(\lambda_{flow}\) (e.g., 0.5) until minority classes improve; then restore.

### Class‑based regression (primary): distance‑shaped per‑part scores
- **Idea**: Replace hard 7‑way classification with 7 **logits trained by BCE‑with‑logits** against soft, distance‑shaped targets in \[0,1\]. This reduces boundary noise and improves calibration.
- **Targets from geometry**:
  - For each point and hand part j (thumb/index/middle/ring/little/palm), compute nearest‑surface distance \(d_j\) to that part mesh.
  - Map distance to a score \(s_j\in[0,1]\) with a monotone decay, e.g. Gaussian \(s_j = \exp(-(d_j/\tau)^2)\) or logistic around threshold t: \(s_j = \sigma((t - d_j)/s)\).
  - Set no‑contact target as \(s_{nc} = 1 - \max_j s_j\).
- **Loss**: sum of per‑channel BCE‑with‑logits; use class‑wise `pos_weight` from train frequencies to counter imbalance.
- **Inference**: predict `no_contact` if \(\max_j \sigma(z_j) \le \theta\), else predict \(\arg\max_j \sigma(z_j)\) over the six parts. Use configurable \(\theta\) (default 0.4).
- **Conditioning pool weight**: use predicted contactness (e.g., \(\max_j \sigma(z_j)\)) instead of \(1 - p(\text{no\_contact})\).
- **Notes**: Works well with contact‑aware sampling; cache distances when possible to minimize overhead.

### Sampling and labeling
- **Contact‑aware point sampling**: mix near‑hand points (e.g., within 2× the contact threshold to any MANO vertex) with uniform surface points; recommended 50/50.
- **Label policy**:
  - We standardize the contact threshold at **10 mm**.
  - Optionally use **soft labels** within a boundary band (e.g., 5–15 mm) for analysis; however training will use the distance‑shaped soft targets (regression mode) and 10 mm hard threshold (classification fallback).
  - For regression, compute and (optionally) cache per‑part nearest distances for soft targets; clamp to zero beyond a radius (e.g., 3× threshold) to bound influence.

<!-- Removed former points 4 and 5 per review: data/conditioning diversity and eval/checkpoint cadence -->

## Implementation plan (status-aware, single source of truth)

This section consolidates implementation status, testing steps, and the remaining work. Data is now available at `/mnt/data`.

### A. Current status at a glance

- Core BCE/CE dual‑mode pipeline is integrated and config‑gated.
- Dataset returns both hard labels `[B,N]` and soft targets `[B,N,7]`.
- Model pooling supports regression (`max(sigmoid(parts))`) and classification (`1 − p(no_contact)`).
- Training supports BCE with optional `pos_weight`, and CE fallback.
- Validation logs per‑class accuracy, macro‑F1, and confusion matrix.
- Documentation cross‑links updated; this file is the single source.

Defaults we will use now:
- Contact threshold: **10 mm** (0.01 m)
- Regression label mapping: Gaussian by default; logistic available
- Inference threshold θ: default 0.4 (tunable)

### B. What is READY now

- `config.py` flags: `MODEL.contact_regression`, `MODEL.inference_threshold`, `TRAINING.contact_loss_type`, `TRAINING.pos_weight` (config key only), `REGRESSION_HPARAMS`, `EVAL.num_qualitative`, `EVAL.save_gt_qualitative`.
- Dataset (`dataset/oakink_loader.py`):
  - `_compute_soft_contact_targets()` (vectorized per‑part distances; Gaussian/Logistic; clamp radius)
  - Returns `contact_labels` and `contact_targets`; adds `hand_vertices` to `batch['meta']`.
- Model (`models/functional_grasp_model.py`):
  - `contact_regression` + `inference_threshold` constructor args; branching pooling.
- Training (`train.py`):
  - BCE path with optional `pos_weight`; CE fallback
  - `compute_contact_predictions()` and `compute_contact_metrics()`
  - Validation logs overall acc, per‑class acc, macro‑F1, confusion matrix

### C. Implementation status

**✅ COMPLETED:**
1) ✅ **Contact threshold standardized to 10mm**
   - Updated `config.py`: `'contact_threshold': 0.01`
   - Updated `REGRESSION_HPARAMS.tau_mm: 10.0`
   - Updated documentation

2) ✅ **Class frequency computation script**
   - Created `compute_class_frequencies.py` (root level)
   - Computes `pos_weight = 1 / sqrt(freq)` per class
   - Saves to `/mnt/data/OakInk/cache/class_frequencies_train.json`
   - Successfully tested and ready to use

3) ✅ **Soft‑target caching**
   - Added cache logic in `dataset/oakink_loader.py __getitem__()`
   - Cache key: `f"{frame_id}_soft_{label_type}_{tau_mm}_{threshold_mm}_{clamp_factor}.pkl"`
   - Cache location: `self.cache_dir / 'soft_targets' / cache_key`
   - Loads from cache if exists, else computes and saves

4) ✅ **Qualitative visualization**
   - Implemented `generate_qualitative_visualizations()` in `train.py`
   - Called from `validate()` after metrics
   - Generates pred/GT figure pairs for first N samples
   - Uses `utils/visualization_3d.create_hand_object_figure()`
   - Saves to `cfg['paths']['qual_dir']`

5) ✅ **Updated `visualize_training.py`**
   - Parses enhanced metrics: macro-F1, per-class accuracy
   - Plots 6 subplots (3×2) when enhanced metrics available
   - Shows per-class accuracy trends with color-coded curves
   - Shows macro-F1 dedicated plot and overlaid on accuracy

**⏸️ OPTIONAL (deferred):**
6) ⏸️ **Contact‑aware sampling**
   - Config flags added but not implemented
   - Can add later if needed for further improvements

### D. Next steps (with data at /mnt/data)

**Step 1: Compute class frequencies** (required for pos_weight)

⚠️ **Important**: This is NOT part of data preparation. Run this ONCE before training.

```bash
# Option A: Run setup script (recommended)
bash scripts/setup_training.sh

# Option B: Run directly
python compute_class_frequencies.py --data_path /mnt/data/OakInk --render_dir /mnt/data/OakInk/rendered_objects
```

What it does:
- Loads the already-prepared training dataset
- Iterates through all samples to count contact labels
- Computes `pos_weight = 1/sqrt(freq)` per class (normalized)
- Saves to `/mnt/data/OakInk/cache/class_frequencies_train.json`
- Prints pos_weight values for inspection

**Step 2: Train** (pos_weight auto-loads from cache)

```bash
# Quick test (1 epoch)
python train.py --data_path /mnt/data/OakInk --epochs 1 --batch_size 2 --log_interval 5

# Full training with train.sh
bash scripts/train.sh

# Or customize directly
python train.py --data_path /mnt/data/OakInk --epochs 10 --batch_size 4
```

During training, verify:
- Dataset loads correctly
- Soft targets computed/cached (faster on subsequent epochs)
- BCE loss decreases
- Validation shows enhanced metrics (per-class acc, macro-F1, confusion matrix)
- Qualitative visualizations saved to `outputs/qualitative/`

**Step 3: Visualize results**
```bash
python visualize_training.py
# Generates: outputs/training_metrics.png with 6 subplots including macro-F1 and per-class accuracy
```

### E. Expected outputs

After running the steps above, you should have:

1. **Class frequencies JSON**:
   - `/mnt/data/OakInk/cache/class_frequencies_train.json`
   - Contains counts, frequencies, and computed pos_weight per class

2. **Soft target caches**:
   - `/mnt/data/OakInk/cache/<split_mode>/<split>/soft_targets/*.pkl`
   - One file per frame with soft regression targets
   - Significantly speeds up training after first epoch

3. **Training logs**:
   - `logs/run.log` with enhanced metrics (per-class acc, macro-F1, confusion matrix)
   - TensorBoard logs under `logs/`

4. **Qualitative visualizations**:
   - `outputs/qualitative/val_step_*_sample_*_pred.png` (predicted contact + pose)
   - `outputs/qualitative/val_step_*_sample_*_gt.png` (ground truth contact + pose)

5. **Training plots**:
   - `outputs/training_metrics.png` with 6 subplots showing all metrics

### F. Comparison: BCE vs CE modes

To compare regression (BCE) vs classification (CE):

```bash
# Regression mode (default)
python train.py --data_path /mnt/data/OakInk --epochs 5 --batch_size 4 --contact_regression

# Classification mode (fallback)
python train.py --data_path /mnt/data/OakInk --epochs 5 --batch_size 4 --no_contact_regression --contact_loss_type ce
```

Compare:
- Overall accuracy (should be similar or CE slightly higher)
- Macro-F1 (expect BCE >> CE due to better minority class handling)
- Per-class accuracy (expect BCE better for fingers/palm, CE better for no_contact)
- Confusion matrix (check finger misclassifications)

---

## Implementation details (for reference)

### Key configuration defaults

**Contact labeling:**
- Threshold: 10mm (0.01m) standardized
- Gaussian soft targets: τ = 10mm
- Logistic soft targets: t = 10mm, s = 3mm
- Clamp radius: 3× threshold (30mm)

**Training:**
- Loss: BCE with pos_weight (sqrt-inverse frequency)
- λ_contact = 1.5, λ_flow = 1.0
- Inference threshold θ = 0.4

**Evaluation:**
- Enhanced metrics: per-class accuracy, macro-F1, confusion matrix
- Qualitative samples: 3 per validation
- Save both pred and GT visualizations

**Sampling:**
- Default: uniform FPS (contact_aware_sampling=False)
- Optional: 50/50 near/far mix when enabled


