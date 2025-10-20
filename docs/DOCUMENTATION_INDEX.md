# Documentation Index

**Version**: 2.0
**Last Updated**: October 2025

Quick navigation for FuncGrasp documentation.

## Core Documents

### 📋 [README.md](README.md)
**Purpose**: Quick start guide and navigation map
**Contains**:
- v2.0 status and changes
- Setup instructions
- Basic usage examples
- Quick statistics
- Documentation guidelines

### 📊 [OAKINK_DATASET_DOCUMENTATION.md](OAKINK_DATASET_DOCUMENTATION.md)
**Purpose**: Complete OakInk dataset structure and conventions  
**Contains**:
- Full dataset structure (OakBase, image, shape directories)
- File formats and naming conventions
- Metadata specifications
- Data splits and statistics
- Implementation status
- Recent updates (pre-rendering, contacts)

### ✋ [MANO_HAND_MODEL.md](MANO_HAND_MODEL.md)
**Purpose**: MANO hand model details and usage  
**Contains**:
- Model parameters (pose, shape, translation)
- Joint topology (21 joints)
- Forward kinematics usage
- Contact computation details
- Visualization methods
- Rendering pipeline
- Troubleshooting

### 🚀 [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)
**Purpose**: Complete architecture and training pipeline design
**Contains**:
- **v2.0 Changes**: Critical fixes and improvements
- Full architecture diagram (geometric + semantic encoders, fusion, contact, flow)
- Distance-based soft contact targets (RBF kernel, independent probabilities)
- Contact loss design (multi-label BCE with contact-strength weighting)
- Flow matching implementation (rectified flow, Euler integration)
- Training loop details (forward pass, loss computation, backward pass)
- Inference pipeline (grasp sampling from noise)
- Configuration and hyperparameters
- Debug features and common issues

### 🤖 [QWEN25VL_DOCUMENTATION.md](QWEN25VL_DOCUMENTATION.md)
**Purpose**: Comprehensive Qwen2.5-VL model reference  
**Contains**:
- Model overview and architecture (3B parameters, 2048 hidden size)
- Key features and capabilities
- Installation and setup instructions
- Usage guide with code examples
- API reference and best practices
- Performance benchmarks and comparison
- Integration with our functional grasp project
- Troubleshooting guide

### 📝 [CHANGELOG.md](../CHANGELOG.md)
**Purpose**: Version history and migration guide
**Contains**:
- v2.0 critical fixes (soft targets, gradient flow, numerical stability)
- Technical details of formula changes
- Migration guide for existing code
- Rollback commands if needed

---

## Document Relationships

```
README.md (Map)
├── v2.0 status and changes
├── Quick start and setup
├── Training quick start
├── Points to → PIPELINE_DESIGN.md (for architecture and training details)
└── Points to → Other docs for specific topics

PIPELINE_DESIGN.md (Architecture & Training)
├── v2.0 changes and fixes
├── Full system architecture
├── Contact prediction with soft targets (RBF kernel)
├── Flow matching for pose generation
├── Training loop and losses
├── Debug features
└── Points to → All component docs

CHANGELOG.md (Version History)
├── v2.0 critical fixes
├── Technical details
├── Migration guide
└── Rollback commands

OAKINK_DATASET_DOCUMENTATION.md (Dataset)
├── Dataset structure and conventions
├── Implementation status
└── Points to → MANO_HAND_MODEL.md (for hand details)

MANO_HAND_MODEL.md (Hand Model)
├── Model specification and parameters
├── Forward kinematics
├── Contact computation
└── Visualization and rendering

QWEN25VL_DOCUMENTATION.md (VL Model)
├── Model architecture and features
├── Usage guide and API reference
├── Integration with project
└── Best practices and troubleshooting
```

---

## Quick Links by Topic

### Getting Started
- [Setup](README.md#setup)
- [Basic Usage](README.md#usage)
- [Pre-rendering Objects](README.md#2-pre-render-object-images-optional)

### Dataset
- [OakInk Structure](OAKINK_DATASET_DOCUMENTATION.md#dataset-structure)
- [File Formats](OAKINK_DATASET_DOCUMENTATION.md#data-format)
- [Data Splits](OAKINK_DATASET_DOCUMENTATION.md#official-data-split-convention)

### Hand Model
- [MANO Parameters](MANO_HAND_MODEL.md#input-parameters)
- [Joint Topology](MANO_HAND_MODEL.md#joint-topology)
- [Forward Kinematics](MANO_HAND_MODEL.md#forward-kinematics)
- [Contact Labels](MANO_HAND_MODEL.md#contact-labels-7-classes)

### Implementation
- [Dataset Loader](OAKINK_DATASET_DOCUMENTATION.md#dataset-loader)
- [Pre-rendering Pipeline](OAKINK_DATASET_DOCUMENTATION.md#pre-rendering-pipeline)
- [Visualization Tools](OAKINK_DATASET_DOCUMENTATION.md#visualization-tools)

### Model Architecture
- [Qwen2.5-VL Overview](QWEN25VL_DOCUMENTATION.md#overview)
- [Key Features](QWEN25VL_DOCUMENTATION.md#key-features)
- [Model Architecture](QWEN25VL_DOCUMENTATION.md#model-architecture)
- [Usage Guide](QWEN25VL_DOCUMENTATION.md#usage-guide)
- [API Reference](QWEN25VL_DOCUMENTATION.md#api-reference)
- [Best Practices](QWEN25VL_DOCUMENTATION.md#best-practices)
- [Integration with Project](QWEN25VL_DOCUMENTATION.md#integration-with-our-project)

### Architecture & Training
- [System Architecture](PIPELINE_DESIGN.md#architecture)
- [Component Details](PIPELINE_DESIGN.md#component-details)
- [Soft Contact Targets](PIPELINE_DESIGN.md#contact-prediction-distance-based-soft-targets)
- [Loss Functions](PIPELINE_DESIGN.md#loss-functions)
- [Training Loop](PIPELINE_DESIGN.md#training-loop)
- [Inference Pipeline](PIPELINE_DESIGN.md#inference)
- [Configuration](PIPELINE_DESIGN.md#training-configuration)
- [Debug Features](PIPELINE_DESIGN.md#debug-features)
- [Common Issues](PIPELINE_DESIGN.md#common-issues--solutions)

### Troubleshooting
- [MANO Issues](MANO_HAND_MODEL.md#troubleshooting)
- [Contact Computation](MANO_HAND_MODEL.md#contact-computation)
- [Common Pipeline Issues](PIPELINE_DESIGN.md#common-issues--solutions)

### Version History
- [v2.0 Changes](PIPELINE_DESIGN.md#version-20-changes)
- [CHANGELOG](../CHANGELOG.md)
- [Migration Guide](../CHANGELOG.md#migration-guide)

---

## Documentation Guidelines

See [README.md - Documentation Guidelines](README.md#documentation-guidelines) for:
- Core principles (minimal, up-to-date, separation of concerns)
- Document structure
- When to create new docs

---

**Last Updated**: October 2025

