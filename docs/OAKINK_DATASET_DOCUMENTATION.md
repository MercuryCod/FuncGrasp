# OakInk Dataset Documentation

## Overview

OakInk is a large-scale hand-object interaction dataset featuring MANO hand poses with objects. The dataset contains multi-view RGB-D sequences with hand pose annotations for various object manipulation tasks.

- **Project Page**: https://oakink.net
- **Code Repository**: https://github.com/oakink/OakInk
- **Paper**: https://arxiv.org/abs/2203.15709
- **License**: CC-BY-NC-SA-3.0
- **Dataset Root**: `/workspace/data/OakInk/`

---

## Dataset Structure

The dataset is organized into four main directories:

```
/workspace/data/OakInk/
├── OakBase/          # Object part segmentation with affordance labels
├── image/            # Image sequences and hand annotations
├── shape/            # Object meshes and grasp annotations
└── zipped/           # Original compressed data
```

---

## 1. OakBase - Object Part Segmentation

**Location**: `/workspace/data/OakInk/OakBase/`

### Structure
```
OakBase/
├── <category>/        # e.g., mug, bottle, camera, etc.
│   ├── <object_id>/   # e.g., mug_s001, bottle_s002
│   │   ├── part_01.ply      # 3D mesh for part 1
│   │   ├── part_01.json     # Part metadata
│   │   ├── part_02.ply      # 3D mesh for part 2 (if multi-part)
│   │   └── part_02.json     # Part metadata
```

### Categories (29 total)
- **Container Objects**: `mug`, `cup`, `bowl`, `bottle`, `wineglass`, `cylinder_bottle`, `lotion_bottle`
- **Tools**: `hammer`, `screwdriver`, `power_drill`, `knife`, `scissor`, `wrench`, `pincer`
- **Electronics**: `camera`, `mouse`, `game_controller`, `flashlight`, `headphones`, `lightbulb`
- **Others**: `binoculars`, `eyeglasses`, `marker`, `toothbrush`, `trigger_sprayer`, `squeeze_tube`, `frying_pan`, `teapot`

### Part Metadata Format
Each `part_XX.json` contains:
```json
{
    "attr": ["contain_sth", "flow_out_sth", "flow_in_sth", "held_by_hand"],
    "name": "mug_body"
}
```

**Affordance Attributes**:
- `contain_sth`: Can contain liquids/items
- `flow_out_sth`: Can pour out contents
- `flow_in_sth`: Can receive liquids
- `held_by_hand`: Graspable surface
- And others depending on object type

### Statistics
- **Total objects**: ~2,000+ unique object instances
- **Multi-part objects**: Most objects have 1-3 parts (e.g., mug body + handle)
- **File format**: PLY meshes + JSON metadata

---

## 2. Image Sequences - Video Annotations

**Location**: `/workspace/data/OakInk/image/`

### Structure
```
image/
├── obj/                     # Subject-specific hand meshes (OBJ files)
└── anno/
    ├── hand_j/              # Hand joint positions (21 keypoints)
    ├── hand_v/              # Hand mesh vertices (778 vertices)
    ├── cam_intr/            # Camera intrinsic matrices (3x3)
    ├── obj_transf/          # Object transformation matrices (4x4)
    ├── general_info/        # General sequence information
    ├── split/               # Train/test splits for different tasks
    │   ├── split0/          # Split 0: seq_train.json, seq_test.json
    │   ├── split0_ho/       # Split 0 with hand-object pairs
    │   ├── split1/          # Split 1
    │   └── split2/          # Split 2
    ├── split_train_val/     # Train/validation splits
    ├── seq_status.json      # Sequence version information
    └── seq_all.json         # Full sequence listing
```

### File Naming Convention
```
<subject_id>_<object_id>_<task_id>__<timestamp>__<camera_id>__<frame_id>__<hand_side>.pkl

Example:
A01001_0001_0000__2021-09-26-19-59-58__0__16__0.pkl
│      │    │     │                    │  │   │
│      │    │     │                    │  │   └─ Hand side (0=right, 1=left)
│      │    │     │                    │  └───── Frame ID
│      │    │     │                    └────────── Camera ID
│      │    │     └─────────────────────────────── Timestamp
│      │    └───────────────────────────────────── Task variant
│      └────────────────────────────────────────── Object instance
└───────────────────────────────────────────────── Subject ID
```

### Hand Joint Annotations (`hand_j/`)
- **Format**: Pickled NumPy array
- **Shape**: `(21, 3)` - 21 hand joints in 3D space
- **Coordinate System**: Meters (metric units)
- **Joint Order**: Standard MANO hand topology
  - 0: Wrist
  - 1-4: Thumb (root to tip)
  - 5-8: Index finger
  - 9-12: Middle finger
  - 13-16: Ring finger
  - 17-20: Pinky finger

### Hand Mesh Vertices (`hand_v/`)
- **Format**: Pickled NumPy array
- **Shape**: `(778, 3)` - Full MANO mesh vertices
- **Coordinate System**: Meters (metric units)
- **Usage**: Dense hand surface representation for contact analysis

### Camera Intrinsics (`cam_intr/`)
- **Format**: Pickled NumPy array
- **Shape**: `(3, 3)` - Standard camera intrinsic matrix
- **Content**:
  ```
  [[fx,  0, cx],
   [ 0, fy, cy],
   [ 0,  0,  1]]
  ```
  Where:
  - `fx`, `fy`: Focal lengths (pixels)
  - `cx`, `cy`: Principal point (pixels)
- **Example values**: fx≈605, fy≈604, cx≈432, cy≈252

### Object Transformations (`obj_transf/`)
- **Format**: Pickled NumPy array
- **Shape**: `(4, 4)` - Homogeneous transformation matrix
- **Purpose**: Object pose in camera coordinates
- **Content**: SE(3) transformation matrix
  ```
  [[R11, R12, R13, tx],
   [R21, R22, R23, ty],
   [R31, R32, R33, tz],
   [  0,   0,   0,  1]]
  ```
  Where R is rotation and (tx, ty, tz) is translation

### Subject Hand Meshes (`obj/`)
- **Format**: Wavefront OBJ files
- **Files**: One OBJ file per subject (e.g., `A01001.obj`, `A02011.obj`)
- **Purpose**: Subject-specific hand template meshes
- **Usage**: Can be deformed using MANO shape parameters for each subject

### Data Splits (`split/`)
Each split directory contains:
- **`seq_train.json`**: List of training sequences
- **`seq_test.json`**: List of test sequences
- **Format**: JSON array of `[sequence_id, camera_id, frame_id, hand_side]`
  - Example: `["Y35037_0001_0000/2021-10-09-13-20-12", 0, 27, 1]`
  - Hand side: 0 = right hand, 1 = left hand
  - Camera ID: 0-3 (multi-view setup)

**Split Types**:
- **split0**: General hand pose estimation split
- **split0_ho**: Hand-object interaction subset
- **split1, split2**: Alternative splits for cross-validation

### Sequence Status (`seq_status.json`)
- Maps sequence identifiers to dataset versions (v1.0.0, v1.1.0, etc.)
- Format: `{"<subject>_<object>_<task>/<timestamp>": "v1.0.0"}`

---

## 3. Shape - Object Meshes and Grasps

**Location**: `/workspace/data/OakInk/shape/`

### Structure
```
shape/
├── metaV2/                      # Object metadata
│   ├── object_id.json           # Real object IDs and attributes
│   ├── virtual_object_id.json   # Virtual object IDs
│   └── yodaobject_cat.json      # Category mapping
├── OakInkObjectsV2/             # Real scanned objects (~100)
│   └── <object_name>/
│       ├── align/               # Aligned mesh
│       └── align_ds/            # Downsampled aligned mesh
│           ├── textured_simple.obj
│           └── textured_simple.obj.mtl
├── OakInkVirtualObjectsV2/      # Virtual objects (~1700)
│   └── <object_name>/
│       ├── align/
│       └── align_ds/
└── oakink_shape_v2/             # Grasp annotations
    └── <category>/              # e.g., mug, bottle, etc.
        └── <shape_id>/          # e.g., S10001, C10001
            └── <hash>/          # Grasp instance hash
                ├── hand_param.pkl     # MANO parameters (source grasp)
                ├── source.txt         # Source sequence reference
                └── <subject_id>/      # Per-subject refined grasps
                    └── hand_param.pkl
```

### Object Metadata (`metaV2/object_id.json`)
Each object entry contains:
```json
{
    "O01000": {
        "name": "omo_cleaner",
        "class": "container",
        "attr": ["trigger_sprayer"],
        "from": "original",
        "scale": "meter"
    }
}
```

**Fields**:
- `name`: Human-readable object identifier
- `class`: Functional category (container, tool, etc.)
- `attr`: List of functional attributes
- `from`: Data source (original, ShapeNetCore, ContactPose)
- `scale`: Spatial units (meter)

### Object Types
- **OakInkObjectsV2**: ~100 real-world scanned objects
- **OakInkVirtualObjectsV2**: ~1,700 synthetic objects from ShapeNet and other sources

### Object Meshes
- **Format**: Wavefront OBJ with MTL material files
- **Variants**:
  - `align/`: Original resolution aligned mesh
  - `align_ds/`: Downsampled for faster processing
- **Coordinate System**: Meters, aligned to canonical orientation
- **Texture**: Included via MTL material files

### Grasp Annotations (`oakink_shape_v2/`)

#### Hand Parameters (`hand_param.pkl`)
Each grasp is stored as a dictionary with MANO parameters:
```python
{
    'pose': np.ndarray,   # Shape: (48,) - MANO pose parameters
                          # [0:3] global rotation (axis-angle)
                          # [3:48] PCA pose coefficients (15 joints × 3)
    'shape': np.ndarray,  # Shape: (10,) - MANO shape (β) coefficients
    'tsl': np.ndarray     # Shape: (3,) - Global translation (meters)
}
```

**MANO Pose Parameters**:
- **Total dimensions**: 48
- **Global rotation**: First 3 values (axis-angle for wrist orientation)
- **PCA coefficients**: Remaining 45 values (15 hand joints × 3 DOF each, compressed)
- **Units**: Radians for rotations

**MANO Shape Parameters**:
- **Dimensions**: 10 β-coefficients
- **Purpose**: Subject-specific hand shape modeling
- **Usage**: Modulates template mesh to match individual hand geometry

**Translation (tsl)**:
- **Dimensions**: 3 (x, y, z)
- **Units**: Meters
- **Purpose**: Global wrist position in world coordinates

#### Grasp Organization
- **Per-object grasps**: Stored under `<category>/<shape_id>/<hash>/`
- **Source grasp**: Root `hand_param.pkl` (initial grasp from motion capture)
- **Refined grasps**: Per-subject subdirectories with optimized grasps
- **Source tracking**: `source.txt` links back to original image sequence

#### Shape IDs
- Format: `S10001`, `C10001`, etc.
- `S` prefix: ShapeNet-derived objects
- `C` prefix: ContactPose objects
- `O` prefix: Original OakInk scans

---

## 4. Data Statistics

### Object Counts
- **OakBase parts**: ~2,000+ object instances with part segmentation
- **Real objects**: ~100 scanned meshes (OakInkObjectsV2)
- **Virtual objects**: ~1,700 synthetic meshes (OakInkVirtualObjectsV2)
- **Categories**: 29 functional object categories

### Grasp Annotations
- **Estimated grasps**: 10,000+ grasp instances across all objects
- **Subject diversity**: Multiple subjects per object (s10101, s10102, etc.)
- **Grasp variants**: Multiple grasps per object-subject pair

### Image Sequences
- **Sequences**: Thousands of multi-view recordings
- **Subjects**: Multiple human subjects (A01001, A01002, etc.)
- **Frame rate**: Variable (extracted frames from videos)
- **Annotations per frame**: Hand joints (21) + vertices (778)

---

## 5. Coordinate Systems and Units

### Spatial Units
- **All 3D data**: Meters (m)
- **Hand joints**: Meters
- **Hand vertices**: Meters
- **Object meshes**: Meters
- **MANO translation**: Meters

### Coordinate Frame
- **Right-handed coordinate system**
- **Origin**: Varies by data type
  - Hand data: Typically wrist-centered
  - Object meshes: Canonical object-centered
  - Scene data: Camera or world frame

### Rotation Representation
- **MANO pose**: Axis-angle (3 values per joint)
- **Hand orientation**: Quaternions or rotation matrices (depending on usage)

---

## 6. Key Data Formats

### Pickle Files (`.pkl`)
- **Hand annotations**: NumPy arrays or dictionaries
- **MANO parameters**: Dictionary with 'pose', 'shape', 'tsl' keys
- **Loading**: `pickle.load(open(file, 'rb'))`

### JSON Files (`.json`)
- **Metadata**: Object attributes, part information
- **Sequences**: Frame lists, status tracking
- **Loading**: `json.load(open(file))`

### 3D Meshes
- **OBJ format**: Textured object meshes
- **PLY format**: Part segmentation meshes
- **Loading**: Use `trimesh`, `open3d`, or `pyvista`

---

## 7. Intent/Action Modes

The OakInk dataset categorizes grasps by **intent**, representing different manipulation tasks:

| Intent | Action ID | Description |
|--------|-----------|-------------|
| **use** | 0001 | Using the object for its intended purpose |
| **hold** | 0002 | Holding the object statically |
| **liftup** | 0003 | Lifting the object |
| **handover** | 0004 | Handing over the object to another person |

**Intent-specific characteristics:**
- **use**: Functional grasps optimized for object manipulation
- **hold**: Stable grasps for carrying or holding
- **liftup**: Power grasps for lifting heavy objects
- **handover**: Two-hand coordination (includes alternate hand pose)

**Data organization:**
- Intent is encoded in the source sequence path
- Format: `pass1/OBJID_ACTIONID_SUBJECT/timestamp/dom.pkl`
- Example: `pass1/S10001_0001_0002/2021-09-28-17-03-03/dom.pkl`
  - Object: S10001
  - Action: 0001 (use)
  - Subject: 0002

---

## 8. Data Split Convention

**Official split method** (from oikit):
- Based on MD5 hash of object ID: `hash(object_id) % 10`
- **Train**: hash % 10 < 8 (80% of objects)
- **Val**: hash % 10 == 8 (10% of objects)
- **Test**: hash % 10 == 9 (10% of objects)

This ensures:
- Consistent splits across runs
- Object-level split (all grasps of an object stay in same split)
- No data leakage between splits

---

## 9. Object Mesh Centering

**Important convention** (from oikit):

Objects are **centered at their bounding box center** before use:

```python
bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
mesh.vertices = mesh.vertices - bbox_center
```

This means:
- Object point clouds are centered at origin (0, 0, 0)
- Hand poses are relative to the centered object
- Makes the coordinate system more stable and normalized

---

## 10. Hand Center Convention

The hand translation (`tsl`) in MANO parameters represents the **wrist position**.

However, oikit uses **joint index 9** as the hand center:

```python
CENTER_IDX = 9  # Middle finger MCP joint
hand_tsl = hand_joints[CENTER_IDX]  # Update to center joint
```

This provides a more stable reference point than the wrist for grasp alignment.

---

## 11. Usage Examples

### Load Hand Joint Annotation
```python
import pickle
import numpy as np

# Load hand joints
joints = pickle.load(open('hand_j/<filename>.pkl', 'rb'))
# Shape: (21, 3) - 21 joints in 3D space (meters)
```

### Load Hand Mesh Vertices
```python
import pickle

# Load hand vertices
vertices = pickle.load(open('hand_v/<filename>.pkl', 'rb'))
# Shape: (778, 3) - Full MANO mesh (meters)
```

### Load MANO Grasp Parameters
```python
import pickle

# Load grasp parameters
grasp = pickle.load(open('oakink_shape_v2/.../hand_param.pkl', 'rb'))
pose = grasp['pose']    # (48,) - MANO pose
shape = grasp['shape']  # (10,) - MANO shape
tsl = grasp['tsl']      # (3,) - Translation
```

### Load Object Mesh
```python
import trimesh

# Load object mesh
mesh = trimesh.load('OakInkObjectsV2/<obj>/align_ds/textured_simple.obj')
vertices = mesh.vertices  # (N, 3)
faces = mesh.faces        # (M, 3)
```

### Render Hand with MANO
```python
import torch
from manotorch.manolayer import ManoLayer

# Initialize MANO layer
mano_layer = ManoLayer(center_idx=0, mano_assets_root='assets/mano_v1_2')

# Load grasp parameters
grasp = pickle.load(open('hand_param.pkl', 'rb'))

# Forward kinematics
mano_output = mano_layer(
    torch.from_numpy(grasp['pose']).unsqueeze(0),
    torch.from_numpy(grasp['shape']).unsqueeze(0)
)

# Get vertices and apply translation
hand_verts = mano_output.verts.squeeze().numpy() + grasp['tsl'][None]
hand_faces = mano_layer.th_faces.numpy()
```

---

## 12. Comparison with DexFuncGraspNet

| Feature | OakInk | DexFuncGraspNet |
|---------|--------|-----------------|
| **Hand Model** | MANO (human hand) | Shadow Hand (robotic) |
| **Pose Parameters** | 48 (MANO PCA) | 29 (quat + trans + 22 joints) |
| **Data Source** | Motion capture + refinement | Simulation/optimization |
| **Object Meshes** | Real scans + synthetic | Primarily synthetic |
| **Grasp Type** | Contact grasps | Approach poses |
| **Coordinate Units** | Meters | Millimeters |
| **Dataset Size** | ~10K+ grasps, 1800 objects | ~2K grasps, 136 objects |
| **Annotations** | Dense video sequences | Static grasp poses |
| **Part Segmentation** | Yes (OakBase) | No |

---

## 13. Important Notes

1. **MANO Dependency**: To use grasp annotations, you need the MANO hand model. Install `manotorch` or use the official MANO layer.

2. **Coordinate Alignment**: Ensure object and hand are in the same coordinate frame when visualizing. The dataset provides aligned meshes in `align_ds/`.

3. **Scale Consistency**: All spatial data is in meters. Be careful when mixing with other datasets (e.g., DexFuncGraspNet uses millimeters).

4. **Grasp Refinement**: The `oakink_shape_v2` directory contains both source grasps (from motion capture) and per-subject refined grasps. Use refined grasps for better quality.

5. **Multi-view Consistency**: Image sequences may have annotations from multiple camera views. Use camera IDs to track views.

6. **Version Tracking**: Check `seq_status.json` for annotation version information. Some sequences have been updated (v1.0.0 → v1.1.0).

---

## 14. Implementation Status

### Dataset Loader
**Location**: `/workspace/FuncGrasp/dataset/dataset.py`

**Features**:
- ✅ Unified OakInkDataset class
- ✅ Automatic train/val/test splitting (MD5-based)
- ✅ MANO parameter loading (pose, shape, translation)
- ✅ Object point cloud sampling (configurable number of points)
- ✅ Text instruction generation (category + intent)
- ✅ Pre-computed contact labels (7-class finger-specific)
- ✅ Pre-rendered multi-view image loading
- ✅ Metadata extraction (category, intent, IDs)

### Pre-rendering Pipeline
**Location**: `/workspace/FuncGrasp/scripts/prerender_objects.py`

**Features**:
- ✅ Batch rendering of all 68 unique objects
- ✅ 6 viewpoints per object (front, back, left, right, top, bottom)
- ✅ Matplotlib-based orthographic rendering
- ✅ Automatic mesh normalization and centering
- ✅ Configurable image size and zoom
- ✅ Resume support (skip already rendered objects)
- ✅ Error handling and metadata tracking

**Output**: `/workspace/data/OakInk/rendered_objects/`
- 68 object directories
- 408 PNG images (68 × 6 views)
- ~20MB total disk space
- ~77 seconds rendering time

### Visualization Tools
**Location**: `/workspace/FuncGrasp/utils/visualize.py`

**Features**:
- ✅ Interactive 3D visualization (k3d)
- ✅ MANO hand mesh rendering
- ✅ Object point cloud visualization
- ✅ Contact point coloring (7-class)
- ✅ Joint position markers (optional)

### Utilities
- **MANO FK**: `/workspace/FuncGrasp/utils/mano_utils.py`
- **Contact Computation**: `/workspace/FuncGrasp/utils/contact_utils.py`
- **Rendering**: `/workspace/FuncGrasp/utils/render_utils.py`

---

## 15. Recent Updates (2024-10)

### Pre-rendering System
Implemented complete object rendering pipeline:
- Pre-render all objects once (~77s for 68 objects)
- Load images in ~1-2ms vs rendering on-the-fly (~1s per object)
- 6 canonical viewpoints with orthographic projection
- Automatic validation and error handling

### Contact Prediction
Integrated finger-specific contact labels:
- 7-class system (no contact + palm + 5 fingers)
- Pre-computed during dataset loading (optional)
- 10mm distance threshold (configurable)
- Uses MANO mesh vertices for accurate assignment

### Dataset Integration
- Unified loader for all OakInk components
- Single `__getitem__()` returns everything needed
- Optional features via flags (contacts, images)
- Robust error handling with clear messages

See **MANO_HAND_MODEL.md** for hand model details.

---

## 16. References

- **Paper**: Yang, Lixin, et al. "OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction." CVPR 2022.
- **Project**: https://oakink.net
- **GitHub**: https://github.com/oakink/OakInk
- **License**: CC-BY-NC-SA-3.0

---

**Last Updated**: 2024-10-15  
**Documentation Version**: 3.0 (Added implementation status and recent updates)
