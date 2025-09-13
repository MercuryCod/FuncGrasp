# Documentation Overview

This directory contains the technical documentation for the FuncGrasp project.

## Documentation Structure

### Core Documents

1. **[pipeline.md](pipeline.md)** - Architecture & Design Specification
   - Model architecture details
   - Component specifications (Qwen, PointNet++, Fusion, etc.)
   - Training methodology
   - Optional design variants (M.0-M.3)
   - Code examples for each component

2. **[implementation.md](implementation.md)** - Current Implementation Status
   - Repository structure
   - Quick start guide
   - Configuration reference
   - Training instructions
   - API usage examples
   - Troubleshooting guide

3. **[dataset.md](dataset.md)** - Dataset Documentation
   - OakInk dataset structure
   - Data loading pipeline
   - Contact approximation method
   - Semantic attribute mapping

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

## Documentation Maintenance

### When to Update Each Document

| Change Type | Update These Docs |
|------------|------------------|
| Architecture changes | pipeline.md, CLAUDE.md |
| New features | implementation.md, README.md |
| Config changes | implementation.md, pipeline.md |
| Training updates | implementation.md |
| Dataset changes | dataset.md |
| Bug fixes | implementation.md (if affects usage) |

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

### implementation.md
- **Purpose**: Current state and practical usage guide
- **Audience**: Users training or deploying the model
- **Key Sections**: Configuration, training guide, API reference

### dataset.md
- **Purpose**: Data pipeline documentation
- **Audience**: Users preparing or understanding training data
- **Key Sections**: OakInk structure, data loading, preprocessing

## Recent Updates

- **Fine-tuning Support**: Added `freeze_qwen` parameter to enable backbone fine-tuning (3.77B params)
- **Baseline Implementation**: Uses contact‑weighted pooling and a Transformer across points with no input bottleneck
- **PointNet++ Backbone**: Uses PyTorch Geometric PointNet2 (required dependency)
- **Repository Restructure**: Flattened directory structure; models live under top-level `models/`
