# OakInk Dataset Documentation

## Overview

OakInk is a large-scale knowledge repository for hand-object interaction understanding. It contains multi-view RGB images, object 3D models, hand poses, and semantic annotations for dexterous manipulation scenarios.

**Dataset Location**: `/mnt/data/changma/OakInk/`

## Dataset Structure

```
OakInk/
├── OakBase/                    # Object mesh database (organized by category)
│   ├── bottle/                 # Bottle objects with semantic parts
│   ├── bowl/                   # Bowl objects
│   ├── cup/                    # Cup objects  
│   ├── mug/                    # Mug objects
│   ├── knife/                  # Knife objects
│   ├── hammer/                 # Hammer objects
│   ├── screwdriver/            # Screwdriver objects
│   ├── camera/                 # Camera objects
│   └── [25+ other categories]  # Various object types
├── image/                      # Multi-view RGB images and annotations
│   ├── obj/                    # Object mesh files (.obj/.ply)
│   ├── stream_release_v2/      # Multi-view image sequences
│   └── anno/                   # Annotations (poses, camera params, etc.)
├── shape/                      # 3D shape data and metadata
│   ├── metaV2/                 # Object metadata and semantic attributes
│   ├── oakink_shape_v2/        # Processed object meshes
│   ├── OakInkObjectsV2/        # Object shape database
│   └── OakInkVirtualObjectsV2/ # Virtual object variations
└── zipped/                     # Compressed archives
    ├── image/                  # Zipped image data
    ├── OakBase.zip            # Compressed object database
    └── shape/                  # Zipped shape data
```

## Data Components

### 1. Object Database (OakBase/)

**Structure**: Objects organized by functional categories
- Each object has multiple instances/variations (e.g., `bottle_s001`, `bottle_s002`)
- Objects contain semantic parts with functional attributes
- Each part has:
  - `part_XX.json`: Semantic attributes (e.g., "contain_sth", "flow_in_sth", "held_by_hand")
  - `part_XX.ply`: 3D mesh geometry

**Example**: `OakBase/bottle/bottle_s001/`
```json
{"attr": ["contain_sth", "flow_in_sth", "flow_out_sth", "held_by_hand"], "name": "bottle_body"}
```

### 2. Object Metadata (shape/metaV2/)

**Key Files**:
- `object_id.json`: Complete object catalog with semantic attributes
- `virtual_object_id.json`: Virtual object variations
- `yodaobject_cat.json`: Object categorization

**Object Attributes**:
- **Classes**: container, maniptools, wearable, misc, geometry
- **Functional Attributes**: 
  - `pourable`, `squeezable`, `handled`, `revolute`, `pincer_like`
  - `trigger_sprayer`, `lotion_pump`, `symmetric`
- **Sources**: original, AOB, shapenet, contactpose, ycb

**Example Object Entry**:
```json
"S16001": {
    "name": "bottle_s001",
    "class": "container", 
    "attr": ["pourable"],
    "from": "shapenet"
}
```

### 3. Multi-view Images (image/stream_release_v2/)

**Naming Convention**: `{ObjectID}_{SubjectID}_{IntentionID}_{GraspID}/`
- ObjectID: Object identifier (e.g., A01001, S16001, C10001)
- SubjectID: Subject performing the grasp (0001-0004)
- IntentionID: Grasp intention/task (0000-0010+)
- GraspID: Specific grasp instance

**Image Structure**: Each sequence contains:
- Timestamp directory (e.g., `2021-09-26-19-59-58/`)
- Multi-view images: `{camera_position}_color_{frame}.png`
- Camera positions: `north_east`, `south_east`, etc.
- Frame sequences capturing the full grasp motion

### 4. Annotations (image/anno/)

**Annotation Types**:
- `hand_j/`: Hand joint positions (3D keypoints)
- `hand_v/`: Hand vertex positions (detailed mesh)
- `cam_intr/`: Camera intrinsic parameters
- `obj_transf/`: Object transformations
- `general_info/`: General sequence metadata
- `split/`: Train/validation/test splits
- `seq_all.json`: Complete sequence listing
- `seq_status.json`: Sequence processing status

### 5. 3D Shapes (shape/)

**Shape Data**:
- `oakink_shape_v2/`: Processed object meshes for training
- `OakInkObjectsV2/`: Complete object shape database
- `OakInkVirtualObjectsV2/`: Virtual object variations with alignment data
- Alignment files: `manually_transf.pkl`, `scale.pkl`

## Object Categories and Counts

Based on the object metadata, OakInk contains:

### Functional Categories
1. **Containers** (~200+ objects): bottles, mugs, cups, bowls, teapots
   - Attributes: pourable, squeezable, handled, revolute
   - Sources: ShapeNet, ContactPose, YCB, AOB

2. **Manipulation Tools** (~50+ objects): knives, hammers, screwdrivers, scissors
   - Attributes: handled, pincer_like, revolute
   - Focus on tool manipulation and functional grasping

3. **Wearables** (~10+ objects): eyeglasses, headphones, binoculars
   - Attributes: symmetric
   - Emphasis on precise placement and orientation

4. **Miscellaneous** (~20+ objects): cameras, phones, mice
   - Various functional attributes
   - Electronic devices and everyday objects

5. **Geometric Primitives**: spheres, cylinders, torus
   - Basic shapes for controlled experiments

### Source Datasets
- **ShapeNet**: Synthetic 3D models (S-prefixed IDs)
- **ContactPose**: Real-world objects with contact annotations (C-prefixed)
- **YCB**: Yale-CMU-Berkeley object set (Y-prefixed)
- **AOB**: Amazon Object Bank (A-prefixed)
- **Original**: Custom objects (O-prefixed)

## Data Usage in FuncGrasp Pipeline

### Training Data Flow
1. **Images**: Multi-view RGB images from `image/stream_release_v2/`
2. **Semantics**: Generated from object attributes in `shape/metaV2/object_id.json`
3. **Geometry**: Point clouds sampled from meshes in `shape/oakink_shape_v2/`
4. **Hand Poses**: 3D joint positions from `image/anno/hand_j/`
5. **Contact Labels**: Approximated from hand-object proximity (1cm threshold)

### Semantic Prompt Generation
Object attributes are converted to natural language instructions:
- `["pourable", "handled"]` → "grasp the bottle to pour"
- `["squeezable"]` → "squeeze the tube"
- `["pincer_like"]` → "use the scissors to cut"

### Data Splits
- Object-based splits ensure generalization to unseen objects
- Train/validation/test splits available in `image/anno/split/`
- Sequences organized by subject, intention, and grasp variation

## Key Statistics

- **Objects**: 100+ unique objects across 25+ categories
- **Sequences**: 1000+ grasp sequences with multi-view capture
- **Images**: Multi-view RGB images (typically 4-8 camera angles)
- **Subjects**: 4 different subjects performing grasps
- **Intentions**: 10+ different grasp intentions per object
- **Scale**: All objects in meter units for consistent processing

## Critical Issue: Object Image Rendering

### Current Dataset Images vs Pipeline Requirements

**What OakInk Provides**:
- `image/stream_release_v2/`: Multi-view images showing **hand-object interactions**
- These images contain hands grasping objects, not clean object views
- Example: `north_east_color_16.png` shows hands manipulating objects

**What FuncGrasp Pipeline Needs**:
- Clean multi-view object images (left, right, front, etc.) **without hands**
- These serve as visual input to Qwen2.5-VL for semantic understanding
- Required for visual-semantic fusion in the pipeline

### Solution: Object Rendering Pipeline

**Available Mesh Resources**:
1. **Clean Object Meshes**:
   - `image/obj/*.obj`: Individual object meshes (e.g., `A02028.obj`, `S10015.obj`)
   - `OakBase/*/*/*.ply`: Part-based object meshes with semantic annotations
   - `shape/OakInkVirtualObjectsV2/*/align/*.obj`: Aligned object models

2. **Rendering Requirements**:
   - Multi-view rendering from standard viewpoints (front, back, left, right, top, bottom)
   - Clean backgrounds for object focus
   - Consistent lighting and camera parameters
   - Resolution compatible with Qwen2.5-VL (224x224 or higher)

### Required Preprocessing Pipeline

```python
# Complete data preparation workflow
def prepare_funcgrasp_data(oakink_root):
    """
    Prepare OakInk data for FuncGrasp training
    
    Steps:
    1. Render clean object views from meshes
    2. Generate point clouds from meshes  
    3. Extract hand poses from annotations
    4. Generate semantic text from attributes
    5. Compute contact labels from proximity
    """
    
    # 1. Object Rendering (REQUIRED - NOT IN DATASET)
    for object_id in object_ids:
        mesh_path = f"{oakink_root}/image/obj/{object_id}.obj"
        rendered_views = render_multi_view_object(
            mesh_path, 
            views=['front', 'back', 'left', 'right', 'top', 'bottom'],
            resolution=(224, 224),
            clean_background=True
        )
        save_rendered_views(object_id, rendered_views)
    
    # 2. Point Cloud Generation
    point_cloud = sample_points_from_mesh(mesh_path, n_points=1024)
    
    # 3. Hand Pose Extraction
    hand_poses = load_hand_annotations(f"{oakink_root}/image/anno/hand_j/")
    
    # 4. Semantic Text Generation
    attributes = load_object_attributes(f"{oakink_root}/shape/metaV2/object_id.json")
    text = generate_instruction_from_attributes(attributes[object_id])
    
    # 5. Contact Label Computation
    contact_labels = compute_contact_from_proximity(hand_poses, point_cloud, threshold=0.01)
```

### Missing Component: Object Rendering System

**What We Need to Implement**:
1. **3D Rendering Engine**: Use PyTorch3D, Blender, or similar to render clean object views
2. **Camera Setup**: Define standard viewpoints and camera parameters
3. **Lighting**: Consistent illumination for all objects
4. **Background**: Clean or gradient backgrounds (not cluttered scenes)

**Rendering Configuration**:
```python
# Standard viewpoints for object rendering
VIEWPOINTS = {
    'front': (0, 0, 2),      # Front view
    'back': (0, 180, 2),     # Back view  
    'left': (-90, 0, 2),     # Left side
    'right': (90, 0, 2),     # Right side
    'top': (0, -90, 2),      # Top-down
    'bottom': (0, 90, 2)     # Bottom-up
}

# Camera parameters
CAMERA_CONFIG = {
    'resolution': (224, 224),
    'fov': 60,
    'near': 0.1,
    'far': 10.0
}
```

## Integration with FuncGrasp

### Complete Data Loader Requirements
```python
# Expected batch format for training (after rendering)
batch = {
    'images_list': List[List[PIL.Image]],      # Rendered multi-view object images per sample
    'texts_list': List[str],                   # Generated from object attributes  
    'points': torch.Tensor[B, 1024, 3],       # Sampled from object meshes (object frame)
    'contact_labels': torch.LongTensor[B, 1024],  # 7-way per-point labels (0..6)
    'pose': torch.Tensor[B, 63]               # 21 joints × 3 coordinates
}
```

### Data Processing Workflow
1. **Object Rendering** (NEW - REQUIRED):
   - Load mesh from `image/obj/{object_id}.obj`
   - Render 6 standard views with clean backgrounds
   - Save as multi-view image set per object

2. **Point Cloud Generation**:
   - Sample 1024 points from same mesh using FPS
   - Normalize and center point cloud

3. **Hand Pose Processing**:
   - Load from `image/anno/hand_j/{sequence}.pkl`
   - Convert to 63D representation (21 joints × 3), transformed to object frame

4. **Contact Label Generation (7-way finger/palm)**:
   - Use MANO hand mesh vertices `image/anno/hand_v/{sequence}.pkl` (778×3) transformed to object frame
   - Assign each MANO vertex to a hand part (thumb/index/middle/ring/little/palm) via skinning tree
   - For each object point, compute nearest distance to each part mesh and choose argmin
   - Threshold at 1cm for contact; else label as `no_contact`

5. **Semantic Text Generation**:
   - Load attributes from `shape/metaV2/object_id.json`
   - Convert to natural language (e.g., "grasp the bottle to pour")

## File Formats

- **Meshes**: `.ply` and `.obj` files for 3D geometry
- **Images**: `.png` files for RGB captures
- **Annotations**: `.pkl` files for poses, camera parameters, transformations
- **Metadata**: `.json` files for object attributes and sequence information

## Usage Notes

- All objects are scaled to meter units
- Hand poses use consistent coordinate frames
- Multi-view images provide comprehensive object coverage
- Semantic attributes enable rich natural language conditioning
- Contact annotations support supervision for contact prediction

## Action Items for FuncGrasp Training

### Immediate Requirements
1. **Implement Object Rendering System**:
   - Create script to render clean multi-view images from OakInk meshes
   - Use PyTorch3D, Blender Python API, or similar rendering engine
   - Generate 6 standard views per object with consistent parameters

2. **Update Data Loader**:
   - Modify `dataset/oakink_loader.py` to use rendered images instead of interaction images
   - Implement point cloud sampling from the same meshes used for rendering
   - Ensure mesh-image-pointcloud alignment

3. **Rendering Script Priority**:
   - Start with `image/obj/*.obj` files (100+ objects)
   - Implement batch rendering for efficiency
   - Store rendered views in organized directory structure

### Rendering Implementation Suggestions

**Option 1: PyTorch3D**
```python
import torch
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PerspectiveCameras, PointLights, RasterizationSettings
)

def render_object_views(mesh_path, output_dir):
    # Load mesh, setup cameras for 6 views, render with consistent lighting
    pass
```

**Option 2: Blender Python API**
```python
import bpy

def render_object_blender(mesh_path, output_dir):
    # Import mesh, position cameras, render views, export images
    pass
```

**Option 3: Open3D**
```python
import open3d as o3d

def render_object_open3d(mesh_path, output_dir):
    # Load mesh, setup visualizer, capture views programmatically  
    pass
```

### Current Status
- ✅ **Point Clouds**: Generated from meshes via FPS
- ✅ **Text Instructions**: Generated from semantic attributes
- ✅ **Hand Poses**: 21 joints available and transformed to object frame (63D)
- ✅ **Contact Labels**: 7-way finger/palm labels via MANO mesh proximity with 1cm threshold
- ✅ **Object Images**: Rendered multi-view images implemented under `dataset/prepare_data.py`
- ✅ **Loader Robustness**: During training, sequences with missing rendered images are filtered out to keep the pipeline running with partial renders.

**Bottom Line**: With the rendering utility in place, the loader expects rendered views, 63D joint poses, and 7-way contact labels for training. Missing renders are skipped during development-only training runs.

## Hand Pose Representation Analysis

### OakInk Hand Pose Formats

**Format 1: Joint Positions** (`image/anno/hand_j/*.pkl`)
- **Structure**: NumPy array `[21, 3]`
- **Content**: 3D joint coordinates (x, y, z) in meters
- **Coverage**: Standard hand skeleton with 21 joints
- **Usage**: Direct 3D keypoint positions

**Format 2: MANO Parameters** (`shape/oakink_shape_v2/*/hand_param.pkl`)
- **Structure**: Dictionary with keys `['pose', 'shape', 'tsl']`
- **pose**: `[48]` - MANO hand pose parameters (joint rotations)
- **shape**: `[10]` - MANO shape parameters (hand geometry)
- **tsl**: `[3]` - Translation vector (wrist position)
- **Usage**: Parametric hand model representation

### Decision for FuncGrasp (Current Setting)

We use direct 21-joint positions (63D) transformed to the object frame:
- Pose vector: `DPOSE = 63` composed of `21 × (x,y,z)`
- Benefits: simpler pipeline, direct interpretability, no MANO dependency in the model

This dataset supports training with:
1. Rendered object images (multi-view) for semantics
2. 63D joint poses in object frame
3. 7-way finger/palm contact labels derived from MANO mesh proximity
