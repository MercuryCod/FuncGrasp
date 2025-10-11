# Documentation Overview

This directory contains the technical documentation for the FuncGrasp project.

## Documentation Structure

### Core Documents

1. **[pipeline.md](pipeline.md)** - Architecture & Design Specification (single source of truth)
   - Model architecture details
   - Component specifications (Qwen, PointNet++, Fusion, etc.)
   - Training methodology (includes Quick Start and API usage examples)
   - Optional design variants (M.0-M.3)
   - Code examples for each component

2. **[oakink.md](oakink.md)** - Dataset Documentation
   - OakInk dataset structure
   - Data loading pipeline
   - Contact approximation method (hard labels and soft targets)
   - Semantic attribute mapping

3. **[CONTACT_ACCURACY_IMPROVEMENT.md](CONTACT_ACCURACY_IMPROVEMENT.md)** - Contact Accuracy Enhancement Plan
   - Problem analysis and root causes
   - Class-based regression approach (BCE vs CE)
   - Implementation status and next steps
   - Configuration and hyperparameters

4. **[WORKFLOW.md](WORKFLOW.md)** - Training Workflow Guide
   - Complete workflow from data prep to training
   - Script purposes and when to run them
   - Troubleshooting guide
   - Examples and customization

### Supporting Documents

4. **[../CLAUDE.md](../CLAUDE.md)** - AI Assistant Context
   - Project overview for AI assistants
   - Critical implementation details
   - Development guidelines
   - Documentation update practices

5. **[../README.md](../README.md)** - Project README
   - Quick project overview
   - Installation instructions
   - Basic usage

6. **[pointnet2.md](pointnet2.md)** - PointNet++ Geometry Encoder

7. **[oakink.md](oakink.md)** - OakInk Dataset Documentation

## Recent Updates

### Contact Accuracy Improvements (Latest)
- **Class-based regression**: Switched from hard 7-way classification to soft, distance-shaped targets trained with BCE-with-logits
- **Enhanced metrics**: Added per-class accuracy, macro-F1, and confusion matrix logging
- **Configurable modes**: Can switch between regression (BCE) and classification (CE) via config
- **Improved pooling**: Regression mode uses `max(sigmoid(parts))` as contactness
- See `CONTACT_ACCURACY_IMPROVEMENT.md` for full details

### Previous Updates
- Qwen2.5‑VL (3B) is always trainable; the semantics encoder returns hidden states `[B, L_max, 2048]` for single text with multiple images. Pooling/projection to `CSEM` happens inside `FunctionalGraspModel`.
- Custom PointNet++ encoders (SSG/MSG) implemented with PyG primitives. See `pointnet2.md`.
   - Architecture and parameterization
   - I/O shapes and feature flow
   - Tuning guide and best practices

## Documentation Maintenance

### When to Update Each Document

| Change Type | Update These Docs |
|------------|------------------|
| Architecture changes | pipeline.md, CLAUDE.md |
| New features | pipeline.md, README.md |
| Config changes | pipeline.md |
| Training updates | pipeline.md |
| Dataset changes | dataset.md |
| Bug fixes | pipeline.md (if affects usage) |

### Update Checklist

- [ ] Code changes tested with `python test_pipeline.py`
- [ ] Relevant documentation updated
- [ ] Examples in docs match actual code
- [ ] Version/date updated in modified docs
- [ ] Changes committed together with code

## Key Information by Document

### pipeline.md
- **Purpose**: Design specification and architecture reference
- **Audience**: Developers implementing or extending the system
- **Key Sections**: Model components, training loop, optional variants

<!-- implementation.md removed: pipeline.md now contains usage and training guidance -->

### dataset.md
- **Purpose**: Data pipeline documentation
- **Audience**: Users preparing or understanding training data
- **Key Sections**: OakInk structure, data loading, preprocessing

## Implementation Status

### Contact Prediction
- **Current**: Class-based regression with soft targets (BCE-with-logits)
- **Fallback**: 7-way classification (CE) available via config
- **Pooling**: Regression uses `max(sigmoid(parts))`, classification uses `1 − p(no_contact)`
- **Metrics**: Per-class accuracy, macro-F1, confusion matrix

### Model Components
- **Semantics**: Qwen2.5-VL returns `[B, L_max, 2048]` hidden states; pooled/projected to CSEM in model
- **Geometry**: PyTorch Geometric PointNet2 (SSG/MSG variants)
- **Fusion**: Transformer across points with tiled semantic features
- **Flow Matching**: Rectified flow for pose generation
