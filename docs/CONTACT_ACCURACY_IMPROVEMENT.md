## Contact accuracy plateau: problem, approach, and implementation plan

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
- **Label refinement**:
  - Reduce the threshold from 10 mm to 5–8 mm to lessen boundary ambiguity.
  - Optionally use **soft labels** within a boundary band (e.g., 5–15 mm) to reduce flips near the threshold.
  - For regression, compute and (optionally) cache per‑part nearest distances for soft targets; clamp to zero beyond a radius (e.g., 3× threshold) to bound influence.

<!-- Removed former points 4 and 5 per review: data/conditioning diversity and eval/checkpoint cadence -->

## Implementation plan (phased checklist)

### Phase 0 – Documentation scaffolding
- [ ] Add this file and cross‑link it from `docs/pipeline.md` Monitoring/Evaluation sections.
- [ ] In `docs/oakink.md`, expand contact approximation details (threshold/soft labels; near‑hand sampling rationale).

### Phase 1 – Config + switches
- [ ] Add flags to `config.py`:
  - `MODEL`: `contact_regression: bool` (default: True), `inference_threshold: float` (default: 0.4)
  - `TRAINING`: `contact_loss_type: {'bce','ce'}` (default: 'bce'), `pos_weight: Optional[List[float]]` (computed from data)
  - `REGRESSION_HPARAMS`: `{label_type: 'gaussian'|'logistic', tau_mm: 8.0, t_mm: 8.0, s_mm: 3.0, clamp_radius_factor: 3.0}`
  - `EVAL`: `num_qualitative: int` (default: 3), `save_gt_qualitative: bool` (default: True)
  - `PATHS`: `qual_dir: str` (default: `./outputs/qualitative`)
  - `DATA`: `contact_aware_sampling: bool` (default: False), `contact_aware_ratio: float` (default: 0.5)
- [ ] Wire flags through CLI overrides in `train.py` (add args for contact_regression, loss_type, tau_mm, inference_threshold).

### Phase 2 – Dataset: soft targets + caching
- [ ] Add `_compute_soft_contact_targets()` method using `trimesh.proximity.closest_point(part_mesh, points)` for vectorized distance computation per part.
- [ ] Map distances → soft targets `[N,7]`:
  - Parts 0‑5: Gaussian `s_j = exp(-(d_j/tau)^2)` or logistic `s_j = sigmoid((t-d_j)/s)`
  - Clamp to 0 beyond `clamp_radius_factor * contact_threshold` to bound influence
  - No‑contact: `s_nc = 1 - max(s_j for j in 0..5)`
- [ ] Update `__getitem__()` to return both:
  - `contact_labels: LongTensor [1,N]` (existing, for CE)
  - `contact_targets: FloatTensor [1,N,7]` (new, for BCE) when config enables regression
- [ ] Cache soft targets with key: `f"{frame_id}_soft_{label_type}_{tau_mm}_{threshold}.pkl"` in existing cache_dir.
- [ ] Add `hand_vertices` to `batch['meta']` dict for GT visualization (currently computed but not returned).
- [ ] On first epoch, compute and cache train set class frequencies to `cache_dir/class_frequencies_{split}.json`.

### Phase 3 – Model: pooling rule
- [ ] Add `contact_regression` parameter to `FunctionalGraspModel.__init__()` and store as instance attribute for checkpoint compatibility.
- [ ] In `forward_backbone()`, branch pooling logic based on `self.contact_regression`:
  - If True: `w = torch.max(torch.sigmoid(logits_c[..., :6]), dim=-1)[0]` (max part probability)
  - If False: `w = 1 - softmax(logits_c)[..., 6]` (existing: 1 - p(no_contact))
- [ ] Store `inference_threshold` as model attribute from config, used later in validation.

### Phase 4 – Training loss + metrics
- [ ] Load cached class frequencies from Phase 2, compute `pos_weight = 1/sqrt(freq)` per class.
- [ ] In `train_one_epoch()`, branch loss computation:
  - BCE: `F.binary_cross_entropy_with_logits(logits_c, contact_targets, pos_weight=pos_weight.unsqueeze(0).unsqueeze(0))`
  - CE: `F.cross_entropy(logits_c.view(B*N, 7), contact_labels.view(B*N))` (existing)
- [ ] Add `compute_contact_predictions()` helper:
  - Regression: `probs = sigmoid(logits[..., :6])`, predict class 6 if `max(probs) ≤ θ`, else `argmax(probs)`
  - CE: `argmax(logits)` (existing)
- [ ] Add `compute_contact_metrics()` returning:
  - Per‑class accuracy: `[acc_0, ..., acc_6]`
  - Macro‑F1: average of per‑class F1 scores
  - Confusion matrix: 7×7 numpy array
- [ ] In `validate()`, compute and log these metrics to both `run.log` and TensorBoard.
- [ ] Qualitative visualization in `validate()`:
  - Select first `cfg['eval']['num_qualitative']` samples from batch
  - For predictions: use `compute_contact_predictions()` + `model.sample()` for pose
  - For GT: use `batch['contact_labels']`, `batch['pose'].view(-1,21,3)`, `batch['meta'][i]['hand_vertices']`
  - Save figures to `cfg['paths']['qual_dir']` with descriptive names
  - Log file paths in logger with format: `Saved qualitative: {path}`
- [ ] Update `visualize_training.py` to parse and plot macro‑F1 and per‑class accuracy from `run.log`.

### Phase 5 – Contact‑aware sampling
- [ ] In `OakInkDataset._load_object_points()`, add contact‑aware sampling when `cfg['data']['contact_aware_sampling']` is True:
  - Compute distances from all mesh points to nearest MANO vertex using vectorized `scipy.spatial.distance.cdist`
  - Select points within `2 * contact_threshold` as "near" pool
  - Sample `n_points * contact_aware_ratio` from near pool, remainder from far pool
  - If near pool is too small, sample with replacement or fall back to uniform
- [ ] Update config to expose sampling parameters:
  - `DATA.contact_aware_sampling: bool` (default: False for initial testing)
  - `DATA.contact_aware_ratio: float` (default: 0.5)
  - `DATA.contact_aware_radius_factor: float` (default: 2.0)

<!-- Removed legacy duplicate phases from earlier draft -->

### Phase 6 – Sanity examples
- [ ] Include sample run.log excerpts before/after changes and a small confusion matrix snippet in `pipeline.md`.
- [ ] Provide a one‑liner to regenerate plots: `python visualize_training.py` → `outputs/training_metrics.png`.

<!-- Ablation and tuning removed: document focuses on implementation steps only -->

## Recommended defaults (initial)
- **Metrics**: enable per‑class/macro metrics in validation.
- **Loss**: BCE‑with‑logits with class‑wise `pos_weight` (sqrt‑inverse frequency); \(\lambda_{contact}=1.5\), \(\lambda_{flow}=1.0\) for first ~1–2 epochs.
- **Pooling**: use `max(sigmoid(parts))` as contactness when `contact_regression=True`.
- **Sampling**: Initially keep uniform (contact_aware_sampling=False); enable 50/50 mix after verifying soft targets work.
- **Labels/targets**: threshold 8 mm; Gaussian soft targets with \(\tau=8\,\text{mm}\); clamp beyond 3× threshold.
- **Qualitative**: `eval.num_qualitative=3`; save to `./outputs/qualitative` with both `_pred.png` and `_gt.png` per example.

## Regression defaults
- **Targets**: Gaussian \(\tau = 8\,\text{mm}\) or logistic \(t = 8\,\text{mm}, s = 3\,\text{mm}\).
- **Inference**: threshold \(\theta\) configurable (default 0.4).
- **Loss**: BCE‑with‑logits with class‑wise `pos_weight` (sqrt‑inverse frequency).
- **Pooling**: use max part probability as contactness for conditioning.


