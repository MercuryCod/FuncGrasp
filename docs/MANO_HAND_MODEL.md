# MANO Hand Model

Documentation for the MANO (hand Model with Articulated and Non-rigid deformations) parametric hand model used in this project.

## Overview

MANO is a statistical hand model that represents hand shape and pose using a compact parametric representation. We use it to:
- Represent hand poses in the dataset
- Compute hand joint positions and mesh vertices
- Render hand meshes for visualization
- Compute hand-object contacts

**Implementation**: We use `manotorch`, a PyTorch implementation of MANO.

---

## Model Parameters

### Input Parameters

MANO takes three inputs to generate a hand mesh:

#### 1. **Pose Parameters** (`mano_pose`) - **48 dimensions**
- **Global rotation** (3D): Root joint orientation (axis-angle)
- **Joint angles** (45D): 15 hand joints × 3 DOF each (axis-angle)
- Total: 16 joints × 3 = 48 parameters
- Format: Axis-angle representation (rotation vector)

#### 2. **Shape Parameters** (`mano_shape`) - **10 dimensions**
- PCA coefficients controlling hand shape variations
- Represents individual hand differences (bone lengths, proportions)
- Typically set to zeros for average hand shape

#### 3. **Translation** (`mano_trans`) - **3 dimensions**
- Global 3D translation of the hand (meters)
- Represents wrist position in world coordinates

### Output

From these 61 parameters, MANO outputs:

- **Joints**: (21, 3) - 3D positions of 21 hand joints
- **Vertices**: (778, 3) - 3D positions of hand mesh vertices
- **Faces**: (1538, 3) - Triangle face connectivity

---

## Joint Topology

MANO defines 21 joints organized hierarchically:

```
Joint Index | Name                    | Parent
----------- | ----------------------- | ------
0           | Wrist                   | -
1           | Thumb CMC               | 0
2           | Thumb MCP               | 1
3           | Thumb IP                | 2
4           | Thumb Tip               | 3
5           | Index MCP               | 0
6           | Index PIP               | 5
7           | Index DIP               | 6
8           | Index Tip               | 7
9           | Middle MCP              | 0 (HAND CENTER)
10          | Middle PIP              | 9
11          | Middle DIP              | 10
12          | Middle Tip              | 11
13          | Ring MCP                | 0
14          | Ring PIP                | 13
15          | Ring DIP                | 14
16          | Ring Tip                | 15
17          | Pinky MCP               | 0
18          | Pinky PIP               | 17
19          | Pinky DIP               | 18
20          | Pinky Tip               | 19
```

**Important**: Joint 9 (Middle MCP) is used as the **hand center** in our convention.

---

## Coordinate System

### OakInk Convention
- **Units**: Meters
- **Coordinate frame**: World coordinates (object-centered after alignment)
- **Hand center**: Middle finger MCP joint (index 9)
- **Handedness**: Right hand by default

### Transformations
1. MANO parameters → Local hand mesh
2. Apply `mano_trans` → World coordinates
3. Objects centered at their bounding box center

---

## Forward Kinematics

### Usage

```python
from utils.mano_utils import mano_forward

# Compute hand geometry from parameters
joints, vertices, faces = mano_forward(
    mano_pose,   # (48,) tensor
    mano_shape,  # (10,) tensor
    mano_trans   # (3,) tensor
)
# joints: (21, 3) - joint positions in meters
# vertices: (778, 3) - mesh vertices in meters
# faces: (1538, 3) - triangle indices
```

### Implementation Details

**Location**: `utils/mano_utils.py`

**Key Functions**:
- `get_mano_layer()`: Initialize MANO layer with correct assets
- `mano_forward()`: Compute joints and vertices from parameters

**MANO Assets**: Located in `manotorch/assets/mano_v1_2/`
- `MANO_RIGHT.pkl`: Right hand model parameters
- Contains shape basis, pose basis, template mesh

**Center Index**: We use `center_idx=0` (wrist) in MANO layer, then apply custom centering at joint 9 in post-processing if needed.

---

## Contact Computation

### Approach

We compute contacts between hand mesh and object point cloud:

1. **Distance computation**: For each object point, compute distance to nearest hand vertex
2. **Contact threshold**: Points within 10mm (0.01m) are considered "in contact"
3. **Finger assignment**: Assign contact points to specific fingers based on nearest vertex's finger region

### Contact Labels (7 classes)

```python
0: No contact      # Distance > threshold
1: Palm contact    # Contact with palm region
2: Thumb contact   # Contact with thumb (joints 1-4)
3: Index contact   # Contact with index finger (joints 5-8)
4: Middle contact  # Contact with middle finger (joints 9-12)
5: Ring contact    # Contact with ring finger (joints 13-16)
6: Pinky contact   # Contact with pinky finger (joints 17-20)
```

### Implementation

**Location**: `utils/contact_utils.py`

**Key Function**:
```python
from utils.contact_utils import compute_contact_points

contact_info = compute_contact_points(
    mano_pose,
    mano_shape,
    mano_trans,
    obj_points,
    contact_threshold=0.01,  # 10mm
    use_vertices=True
)
# Returns: {
#   'finger_labels': (N,) array with values 0-6
#   'contact_mask': (N,) boolean array
#   'contact_distances': (N,) float array (meters)
# }
```

---

## Visualization

### 3D Interactive Visualization

**Location**: `utils/visualize.py`

```python
from utils.visualize import visualize_grasp

visualize_grasp(
    mano_pose, 
    mano_shape, 
    mano_trans, 
    obj_points,
    output_path='output.html',
    contact_info={'finger_labels': labels, ...},  # Optional
    show_joints=True  # Show joint spheres
)
```

**Output**: Interactive HTML with k3d renderer
- Hand mesh (gray/colored by contacts)
- Object point cloud (colored by finger contacts)
- Joint positions (optional spheres)
- Rotate, zoom, pan controls

### Contact Coloring

When `contact_info` is provided:
- **Gray**: No contact (0)
- **Blue**: Palm (1)
- **Red**: Thumb (2)
- **Green**: Index (3)
- **Yellow**: Middle (4)
- **Cyan**: Ring (5)
- **Magenta**: Pinky (6)

---

## Object Rendering

### Multi-view Rendering

**Location**: `utils/render_utils.py`

We render object meshes (not hands) from 6 canonical viewpoints using matplotlib:

```python
from utils.render_utils import render_mesh_multiview

saved_paths = render_mesh_multiview(
    mesh_path='path/to/object.obj',
    output_folder='output/',
    object_id='object_name',
    image_size=512,
    zoom=1.6
)
# Returns: {'front': path, 'back': path, ...}
```

### Viewpoints

| View | Azimuth | Elevation | Description |
|------|---------|-----------|-------------|
| front | 0° | 0° | Looking at front |
| back | 180° | 0° | Looking at back |
| left | -90° | 0° | Looking from left |
| right | 90° | 0° | Looking from right |
| top | 0° | 90° | Looking from above |
| bottom | 0° | -90° | Looking from below |

### Rendering Pipeline

1. **Mesh loading**: Load .obj file with trimesh
2. **Normalization**: Center mesh and scale to unit sphere
3. **Projection**: Orthographic projection (no perspective)
4. **Zoom**: Configurable zoom factor (1.6 default fills ~62.5% of frame)
5. **Export**: 512×512 RGB PNG images

**Settings**:
- Background: White
- Surface color: Gray (0.7, 0.7, 0.7)
- Shading: Enabled with antialiasing
- Axes: Hidden for clean appearance

---

## Dataset Integration

### Pre-computed Contacts

Contacts are pre-computed during dataset loading if `compute_contacts=True`:

```python
dataset = OakInkDataset(
    data_root='/workspace/data/OakInk',
    compute_contacts=True,
    contact_threshold=0.01  # 10mm
)

sample = dataset[0]
# sample['contact_labels']: (N,) tensor with values 0-6
# sample['contact_mask']: (N,) boolean tensor
# sample['contact_distances']: (N,) float tensor (meters)
```

### Pre-rendered Images

Object images are pre-rendered and loaded if `load_object_images=True`:

```python
dataset = OakInkDataset(
    data_root='/workspace/data/OakInk',
    load_object_images=True
)

sample = dataset[0]
# sample['object_images']: dict with 6 views
#   Each view: (512, 512, 3) tensor in range [0, 1]
```

---

## Dependencies

### Required Packages

```bash
# Core MANO implementation
pip install /workspace/Grasp/manotorch

# Dependencies for manotorch
pip install chumpy

# Rendering and visualization
pip install trimesh matplotlib pillow k3d
```

### Asset Files

MANO requires model files in `manotorch/assets/mano_v1_2/`:
- `MANO_RIGHT.pkl` - Right hand model
- Shape basis, pose basis, J regressor, etc.

These are included in the manotorch package installation.

---

## References

### Papers
- **MANO**: Romero, J., Tzionas, D., & Black, M. J. (2017). "Embodied hands: Modeling and capturing hands and bodies together." ACM Transactions on Graphics (TOG), 36(6), 1-17.
- **OakInk**: Yang, L., et al. (2022). "OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction." CVPR 2022.

### Code
- **manotorch**: PyTorch implementation of MANO
- **Original MANO**: https://mano.is.tue.mpg.de/

---

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'manotorch'"**
```bash
pip install /workspace/Grasp/manotorch
```

**2. "ModuleNotFoundError: No module named 'chumpy'"**
```bash
pip install chumpy
```

**3. "FileNotFoundError: MANO_RIGHT.pkl not found"**
- Ensure manotorch is installed correctly
- Check assets exist: `ls manotorch/assets/mano_v1_2/`

**4. Contacts not computing**
- Check if manotorch is installed
- Verify `compute_contacts=True` in dataset init
- Check logs for specific error messages

### Performance

- **Forward kinematics**: ~1-2ms per hand
- **Contact computation**: ~5-10ms per sample (with 1024 object points)
- **Rendering**: ~1s per object (6 views)
- **Image loading**: ~1-2ms per object (from pre-rendered)

---

## Future Enhancements

Potential improvements to the MANO integration:

- [ ] Left hand support (mirror MANO parameters)
- [ ] Batch processing for faster FK
- [ ] GPU-accelerated contact computation
- [ ] Mesh simplification for faster rendering
- [ ] Texture support for hand rendering
- [ ] Pose interpolation utilities
- [ ] Joint angle extraction and analysis

