# Joint-Based Grasp Representation

## Overview

FuncGrasp now uses a direct 21-joint representation for hand poses instead of MANO parameters. This simplifies the pipeline and removes external dependencies while maintaining accuracy.

## Representation Details

### Pose Format
- **Dimension**: 63D vector (21 joints × 3 coordinates)
- **Ordering**: Flattened row-major format: `[x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]`
- **Coordinate Frame**: Object-centered (joints transformed to object frame for pose invariance)

### Joint Indices
The 21 joints follow the standard hand skeleton:
- 0: Wrist
- 1-4: Thumb (CMC, MCP, IP, TIP)
- 5-8: Index finger (MCP, PIP, DIP, TIP)
- 9-12: Middle finger (MCP, PIP, DIP, TIP)
- 13-16: Ring finger (MCP, PIP, DIP, TIP)
- 17-20: Little finger (MCP, PIP, DIP, TIP)

## Data Loading

The `OakInkDataset` now:
1. Loads ground truth joints from `hand_j/{frame_id}.pkl`
2. Transforms joints to object frame using inverse of `obj_transf` matrix
3. Flattens the 21×3 array to 63D vector
4. Computes contact labels using expanded hand surface from joints

Key parameters:
- `transform_to_object_frame=True`: Enable object-frame transformation
- Output pose shape: `[B, 63]` where B is batch size

## Model Architecture

### Configuration Changes
- `DPOSE = 63` (was 51 for MANO)
- Flow matching predicts 63D joint positions directly
- All other components remain unchanged

### Optional Regularization
The training includes optional bone length regularization:
```python
bone_length_reg=0.01  # Weight for bone length consistency
```

This helps maintain anatomically plausible hand poses without MANO constraints.

## Training

No changes required to training command:
```bash
python train.py --config config.py
```

The flow matching loss automatically adapts to the 63D representation.

## Visualization

### Interactive 3D Viewer
The consolidated `visualize_grasp.py` script provides all visualization functionality:

```bash
# Default hand pose
python visualize_grasp.py

# Specific frame from OakInk
python visualize_grasp.py --frame_id "A01001_0001_0000__20210820_163917__00220__60"

# Show as mesh instead of skeleton
python visualize_grasp.py --show_mesh --frame_id "A01001_0001_0000__20210820_163917__00220__60"

# For headless environments (saves PNG instead of interactive view)
python visualize_grasp.py --matplotlib
```

Options:
- `--show_mesh`: Display hand as mesh instead of skeleton
- `--object_frame`: Show in object-centered coordinates (default: True)
- `--frame_id`: Load specific frame from dataset
- `--interactive`: Enable browsing mode (simplified implementation)
- `--matplotlib`: Force matplotlib visualization for headless environments

### Hand Mesh Generation
The `create_hand_mesh_from_joints()` function creates a simplified mesh from 21 joints without MANO dependencies.

## Benefits

1. **Simplicity**: Direct joint prediction without complex MANO parameters
2. **No Dependencies**: Removed chumpy, opendr, manotorch requirements
3. **Flexibility**: Easy to add custom joint constraints or regularizers
4. **Interpretability**: Joint positions are directly interpretable

## Migration Notes

- Existing checkpoints with MANO representation (51D) are incompatible
- New checkpoints will be tagged with pose format in metadata
- Contact label computation uses joint-based hand surface approximation
