import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple, Optional
import trimesh
from pathlib import Path
from torch_geometric.nn import fps
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Note: rtree and pyembree are not required in the current pipeline path
# They can be reintroduced if spatial indexing/acceleration is needed


class OakInkDataset(Dataset):
    """
    OakInk dataset loader for functional grasp training.
    Loads rendered object images, hand poses, object meshes, and computes contact maps.
    Uses 21-joint representation (63D flattened) for direct joint prediction.
    """
    
    # Standard views for rendered objects
    OBJECT_VIEWS = ["front", "left", "right", "back", "top", "bottom"]
    
    def __init__(
        self,
        root_dir: str,
        render_dir: str,
        split: str = "train",
        split_mode: str = "split0",
        n_points: int = 1024,
        contact_threshold: float = 0.01,  # 1cm threshold for contact
        use_cache: bool = True,
        num_views: int = 3,  # Number of object views to use
        semantic_prompts: Optional[Dict] = None,
        transform_to_object_frame: bool = True
    ):
        """
        Args:
            root_dir: Path to OakInk dataset root
            render_dir: Path to directory with rendered object images
            split: 'train' or 'test'
            split_mode: 'split0', 'split0_ho', 'split1', or 'split2'
            n_points: Number of points to sample from object mesh
            contact_threshold: Distance threshold for contact detection (meters)
            use_cache: Whether to cache processed data
            num_views: Number of object views to use (1-6)
            semantic_prompts: Dict mapping object IDs to functional descriptions
            transform_to_object_frame: Whether to transform to object frame
        """
        self.root_dir = Path(root_dir)
        self.render_dir = Path(render_dir)
        self.split = split
        self.n_points = n_points
        self.contact_threshold = contact_threshold
        self.use_cache = use_cache
        self.num_views = min(num_views, len(self.OBJECT_VIEWS))
        self.semantic_prompts = semantic_prompts or {}
        self.transform_to_object_frame = transform_to_object_frame
        

        
        # Load split sequences
        split_file = self.root_dir / "image" / "anno" / "split" / split_mode / f"seq_{split}.json"
        with open(split_file, 'r') as f:
            all_sequences = json.load(f)
        
        # Load object metadata
        meta_file = self.root_dir / "shape" / "metaV2" / "object_id.json"
        with open(meta_file, 'r') as f:
            self.object_meta = json.load(f)
        
        # Filter sequences to only include those with rendered images
        print(f"Filtering sequences for available rendered images...")
        self.sequences = []
        missing_objects = set()
        
        for seq in all_sequences:
            # Extract object ID from sequence
            seq_id = seq[0] if isinstance(seq, list) else seq
            obj_id = self._get_object_id(seq_id)
            
            # Check if rendered images exist
            obj_render_dir = self.render_dir / obj_id
            if obj_render_dir.exists():
                # Check if at least the first view exists
                first_view = self.OBJECT_VIEWS[0] if self.OBJECT_VIEWS else "front"
                if (obj_render_dir / f"{first_view}.png").exists():
                    self.sequences.append(seq)
                else:
                    missing_objects.add(obj_id)
            else:
                missing_objects.add(obj_id)
        
        print(f"Filtered sequences: {len(all_sequences)} -> {len(self.sequences)}")
        print(f"Missing rendered objects: {len(missing_objects)}")
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found with rendered images in {self.render_dir}")
        
        # Filter to final grasp frames only (last frame per sequence)
        print(f"Filtering to final grasp frames (last frame per sequence)...")
        from collections import defaultdict
        seq_groups = defaultdict(list)
        
        for seq in self.sequences:
            seq_id, timestamp, _, _ = seq  # frame_idx and view_idx used in seq itself
            # Group by (seq_id, timestamp) to identify unique grasp sequences
            key = (seq_id, timestamp)
            seq_groups[key].append(seq)
        
        # Keep only the last frame (max frame_idx) of each sequence, with view 0
        final_sequences = []
        for key, frames in seq_groups.items():
            # Find maximum frame index
            max_frame_idx = max(f[2] for f in frames)
            # Get all frames at max_frame_idx
            final_frames = [f for f in frames if f[2] == max_frame_idx]
            # Prefer view 0, fallback to first available view
            view_0_frames = [f for f in final_frames if f[3] == 0]
            if view_0_frames:
                final_sequences.append(view_0_frames[0])
            elif final_frames:
                final_sequences.append(final_frames[0])
        
        print(f"Final grasp filtering: {len(self.sequences)} frames -> {len(final_sequences)} final grasps")
        print(f"  Unique grasp sequences: {len(seq_groups)}")
        self.sequences = final_sequences
        
        
        # Load MANO topology (once per dataset)
        self.mano_faces, self.mano_weights, self.mano_kintree = self._load_mano_topology(
            mano_model_path=Config.DATA.get('mano_model_path', 'assets/mano_v1_2/models/MANO_RIGHT.pkl')
        )
        
        # Cache directory for multiclass contact labels
        self.cache_dir = self.root_dir / "cache" / split_mode / split
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict containing:
                - images_list: List[List[PIL.Image]] - Batch format with B=1
                - texts_list: List[str] - Batch format with B=1
                - points: Object point cloud [1, N, 3]
                - contact_labels: 7-class finger/palm contact labels [1, N] (LongTensor, values 0..6)
                - pose: Hand pose 63D joint representation [1, 63]
                - meta: Additional metadata
        """
        # Parse sequence info
        seq_info = self.sequences[idx]
        seq_id, timestamp, frame_idx, view_idx = seq_info
        
        # Create frame identifier
        # The frame_id format matches the hand_j file naming convention
        frame_id = f"{seq_id.replace('/', '__')}__{timestamp}__{frame_idx}__{view_idx}"
        
        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f"{frame_id}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Extract object ID from sequence
        obj_id = self._get_object_id(seq_id)
        
        # Load rendered object images
        images = self._load_rendered_images(obj_id)
        
        # Load annotations
        hand_joints = self._load_hand_joints(frame_id)  # (21, 3) in world coordinates
        hand_vertices = self._load_hand_vertices(frame_id)  # (778, 3) MANO hand mesh vertices (world)
        obj_transform = self._load_obj_transform(frame_id)
        
        # Always transform hand joints and MANO vertices to object frame
        hand_joints = self._transform_joints_to_object_frame(hand_joints, obj_transform)
        hand_vertices_h = np.hstack([hand_vertices, np.ones((hand_vertices.shape[0], 1))])
        obj_transform_inv = np.linalg.inv(obj_transform)
        hand_vertices = (obj_transform_inv @ hand_vertices_h.T).T[:, :3]
        
        # Load object mesh points in object frame (no world transform applied)
        points = self._load_object_points(obj_id, obj_transform)
        
        # Compute 7-way contact labels using MANO hand mesh (preferred)
        contact_labels = self._compute_contact_labels_mano(hand_vertices, hand_joints, points)
        
        # Generate semantic prompt
        text = self._generate_prompt(obj_id, seq_id)
        
        # Convert joints to 63D flattened representation
        pose = hand_joints.flatten()  # (21, 3) -> (63,)
        
        # Prepare output in batch format for pipeline compatibility
        # Contact labels are always LongTensor for 7-way classification
        batch = {
            "images_list": [images],  # List[List[PIL.Image]] with B=1
            "texts_list": [text],     # List[str] with B=1
            "points": torch.tensor(points, dtype=torch.float32).unsqueeze(0),  # [1, N, 3]
            "contact_labels": torch.tensor(contact_labels, dtype=torch.long).unsqueeze(0),  # [1, N]
            "pose": torch.tensor(pose, dtype=torch.float32).unsqueeze(0),  # [1, 63]
            "meta": {
                "seq_id": seq_id,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "obj_transform": obj_transform
            }
        }
        
        # Cache processed data
        if self.use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(batch, f)
        
        return batch
    
    def _load_hand_joints(self, frame_id: str) -> np.ndarray:
        """Load 3D hand joint positions."""
        path = self.root_dir / "image" / "anno" / "hand_j" / f"{frame_id}.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)  # (21, 3)

    def _load_hand_vertices(self, frame_id: str) -> np.ndarray:
        """Load MANO hand mesh vertices if available; else raise.
        Expected path: image/anno/hand_v/{frame_id}.pkl with shape (778, 3).
        """
        path = self.root_dir / "image" / "anno" / "hand_v" / f"{frame_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"MANO hand vertices not found: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)  # (778, 3)
    
    def _load_obj_transform(self, frame_id: str) -> np.ndarray:
        """Load object 6D pose transformation matrix."""
        path = self.root_dir / "image" / "anno" / "obj_transf" / f"{frame_id}.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)  # (4, 4)
    
    def _load_cam_intrinsic(self, frame_id: str) -> np.ndarray:
        """Load camera intrinsic matrix."""
        path = self.root_dir / "image" / "anno" / "cam_intr" / f"{frame_id}.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)  # (3, 3)
    
    def _load_hand_vertices(self, frame_id: str) -> np.ndarray:
        """Load MANO hand mesh vertices (778, 3) in world coordinates."""
        path = self.root_dir / "image" / "anno" / "hand_v" / f"{frame_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"MANO hand vertices not found: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)  # (778, 3)
    
    def _load_mano_topology(self, mano_model_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load MANO topology from model file.
        
        Returns:
            faces: (F, 3) triangle faces
            weights: (778, J) skinning weights per vertex
            kintree_table: (2, J) parent-child relationships
        """
        path = Path(mano_model_path)
        if not path.exists():
            raise FileNotFoundError(f"MANO model file not found at: {path}")
        with open(path, 'rb') as f:
            mano_data = pickle.load(f, encoding='latin1')
        
        # Extract topology
        faces = mano_data['f']  # (F, 3)
        weights = mano_data['weights']  # (778, J)
        kintree_table = mano_data['kintree_table']  # (2, J)
        
        return faces, weights, kintree_table
    
    def _assign_vertex_parts(self, weights: np.ndarray, kintree_table: np.ndarray) -> np.ndarray:
        """
        Assign each MANO vertex to a hand part (0..5) using skinning weights and kinematic tree.
        
        Args:
            weights: (778, J) skinning weights per vertex
            kintree_table: (2, J) parent-child relationships [parent_idx, child_idx]
            
        Returns:
            part_assignment: (778,) array with values 0..5 (thumb/index/middle/ring/little/palm)
        """
        # Find finger branch roots (direct children of wrist/root)
        # MANO kinematic tree: joint 0 is wrist/root
        root_joint = 0
        children_of_root = kintree_table[1][kintree_table[0] == root_joint]
        
        # Map each joint to its finger branch
        joint_to_branch = {}
        joint_to_branch[root_joint] = 5  # wrist -> palm
        
        # Assign direct children to finger branches
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        for i, child in enumerate(children_of_root):
            if i < len(finger_names):
                joint_to_branch[child] = i  # 0..4 for fingers
            else:
                joint_to_branch[child] = 5  # extra joints -> palm
        
        # Propagate branch assignment to all descendants
        for joint in range(len(kintree_table[0])):
            if joint not in joint_to_branch:
                # Find root ancestor
                current = joint
                while current != root_joint and kintree_table[0][current] != -1:
                    parent = kintree_table[0][current]
                    if parent in joint_to_branch:
                        joint_to_branch[joint] = joint_to_branch[parent]
                        break
                    current = parent
                else:
                    joint_to_branch[joint] = 5  # default to palm if no clear branch
        
        # Assign vertices to parts based on max skinning weight
        part_assignment = np.zeros(len(weights), dtype=np.int64)
        for vi, w in enumerate(weights):
            max_joint = np.argmax(w)
            part_assignment[vi] = joint_to_branch.get(max_joint, 5)  # default to palm
            
        return part_assignment
    
    def _load_rendered_images(self, obj_id: str) -> List[Image.Image]:
        """Load rendered object images."""
        images = []
        obj_render_dir = self.render_dir / obj_id
        
        # Check if object render directory exists
        if not obj_render_dir.exists():
            raise FileNotFoundError(
                f"Rendered object directory not found: {obj_render_dir}\n"
                f"This object ({obj_id}) should have been filtered out during dataset initialization.\n"
                f"Please check your rendered_objects directory."
            )
        
        # Load specified number of views
        for i, view in enumerate(self.OBJECT_VIEWS[:self.num_views]):
            img_path = obj_render_dir / f"{view}.png"
            if img_path.exists():
                images.append(Image.open(img_path).convert('RGB'))
            else:
                # List available files for debugging
                available_files = list(obj_render_dir.glob("*.png"))
                raise FileNotFoundError(
                    f"Rendered image not found: {img_path}\n"
                    f"Available files in {obj_render_dir}: {[f.name for f in available_files]}"
                )
        
        return images
    
    def _transform_joints_to_object_frame(self, joints: np.ndarray, obj_transform: np.ndarray) -> np.ndarray:
        """Transform hand joints from world frame to object frame.
        
        Args:
            joints: (21, 3) hand joints in world coordinates
            obj_transform: (4, 4) object-to-world transformation matrix
            
        Returns:
            joints_obj: (21, 3) hand joints in object coordinates
        """
        try:
            # Compute inverse transformation (world-to-object)
            obj_transform_inv = np.linalg.inv(obj_transform)
            
            # Transform joints
            joints_homo = np.hstack([joints, np.ones((21, 1))])  # (21, 4)
            joints_obj = (obj_transform_inv @ joints_homo.T).T[:, :3]
            
            return joints_obj
        except np.linalg.LinAlgError:
            # If inverse fails, return joints in world frame
            print(f"Warning: Failed to invert object transform, using world coordinates")
            return joints
    
    def _get_object_id(self, seq_id: str) -> str:
        """Extract object ID from sequence ID and align with rendered folder names.

        We render per-object folders using the mesh stem or shape dir name:
          - image/obj:  AXXXXX → folder 'AXXXXX'
          - shape/*:    e.g., 'bottle_s001' → folder 'bottle_s001'

        For compatibility, prefer returning the leading token (e.g., 'AXXXXX'),
        which directly matches the render_dir scheme when rendering from image/obj.
        """
        parts = seq_id.split('_')
        token = parts[0] if parts else seq_id
        # Common OakInk prefixes include A,S,Y,C,O followed by digits
        if token and len(token) > 1 and token[0].isalpha() and token[1:].isdigit():
            return token
        # Fallback to original token or legacy mapping if provided
        if token.startswith('A') and len(token) > 1:
            return token
        return token
    
    def _load_object_points(self, obj_id: str, transform: np.ndarray) -> np.ndarray:
        """Load object mesh and sample points using FPS.
        
        The meshes in image/obj/ are already in the object's canonical frame,
        NOT in world coordinates. The meshes in shape/*/align/ are also in 
        canonical frame but may need scaling.
        """
        # Try different mesh locations and formats
        mesh_paths = [
            self.root_dir / "image" / "obj" / f"{obj_id}.obj",
            self.root_dir / "image" / "obj" / f"{obj_id}.ply",  # Also try PLY format
            self.root_dir / "shape" / "OakInkObjectsV2" / obj_id / "align" / "model_align.obj",
            self.root_dir / "shape" / "OakInkObjectsV2" / obj_id / "align" / "model_align.ply",
            self.root_dir / "shape" / "OakInkVirtualObjectsV2" / obj_id / "align" / "model_align.obj",
            self.root_dir / "shape" / "OakInkVirtualObjectsV2" / obj_id / "align" / "model_align.ply",
        ]

        mesh = None
        chosen_path = None
        for path in mesh_paths:
            if path.exists():
                mesh = trimesh.load(str(path), force='mesh')
                chosen_path = path
                break

        if mesh is None:
            raise FileNotFoundError(f"Object mesh not found for {obj_id}. Tried paths: {mesh_paths}")

        # Sample more points initially for FPS
        initial_points, _ = trimesh.sample.sample_surface(mesh, self.n_points * 10)

        # Convert to torch for FPS
        points_torch = torch.tensor(initial_points, dtype=torch.float32)
        batch = torch.zeros(len(points_torch), dtype=torch.long)

        # Apply FPS to get exactly n_points
        fps_idx = fps(points_torch, batch, ratio=float(self.n_points) / len(points_torch))
        points = points_torch[fps_idx].numpy()

        # Handle scaling for aligned meshes
        if chosen_path is not None:
            path_str = str(chosen_path).replace("\\", "/")
            # If mesh comes from shape/*/align, it may need scaling
            if "/shape/" in path_str and "/align/" in path_str:
                scale_file = chosen_path.parent / "scale.pkl"
                if scale_file.exists():
                    with open(scale_file, 'rb') as f:
                        scale_value = pickle.load(f)
                    # Apply scale
                    if isinstance(scale_value, (float, int)):
                        points = points * float(scale_value)
                    else:
                        scale_arr = np.array(scale_value).reshape(-1)
                        if scale_arr.size == 1:
                            points = points * float(scale_arr[0])
                        elif scale_arr.size == 3:
                            points = points * scale_arr[:3]
                        else:
                            raise ValueError(f"Unsupported scale format: {scale_value}")

        # All meshes are already in object frame, no transform needed
        return points
    
    
    # _expand_hand_surface removed: replaced by MANO mesh-based labeling
    
    # _compute_contact_labels_multiclass removed: replaced by MANO mesh-based labeling

    def _compute_contact_labels_mano(self, hand_vertices: np.ndarray, hand_joints: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Compute 7-way contact labels using MANO mesh topology and point-to-mesh distances.
        Parts: 0..4=fingers (thumb..little), 5=palm, 6=no_contact.
        
        Args:
            hand_vertices: (778, 3) MANO vertices in object frame
            hand_joints: (21, 3) joints (for reference, not used in computation)
            points: (N, 3) object points in object frame
            
        Returns:
            labels: (N,) array with values 0..6
        """
        # Assign vertices to parts using skinning weights and kinematic tree
        part_assignment = self._assign_vertex_parts(self.mano_weights, self.mano_kintree)
        
        # Build part submeshes
        part_meshes = []
        for part_id in range(6):  # 0..5 (thumb/index/middle/ring/little/palm)
            # Get vertices for this part
            part_vertex_mask = (part_assignment == part_id)
            if not np.any(part_vertex_mask):
                raise ValueError(f"No vertices assigned to part {part_id} ({Config.CONTACT_CLASSES[part_id]})")
            
            part_vertices = hand_vertices[part_vertex_mask]
            
            # Filter faces to only include triangles where all 3 vertices belong to this part
            valid_faces = []
            vertex_indices = np.where(part_vertex_mask)[0]
            vertex_set = set(vertex_indices)
            
            for face in self.mano_faces:
                if all(v in vertex_set for v in face):
                    # Remap face indices to local part vertex indices
                    local_face = [np.where(vertex_indices == v)[0][0] for v in face]
                    valid_faces.append(local_face)
            
            if len(valid_faces) == 0:
                raise ValueError(
                    f"No valid faces for part {part_id} ({Config.CONTACT_CLASSES[part_id]}). "
                    f"Check MANO topology and vertex assignment."
                )
            # Create proper mesh
            part_mesh = trimesh.Trimesh(vertices=part_vertices, faces=np.asarray(valid_faces), process=False)
            # Basic validity checks without using non-existent attributes
            if part_mesh.faces is None or len(part_mesh.faces) == 0 or part_mesh.vertices is None or len(part_mesh.vertices) == 0:
                raise ValueError(
                    f"Invalid mesh for part {part_id} ({Config.CONTACT_CLASSES[part_id]}): empty vertices/faces."
                )
            
            part_meshes.append(part_mesh)
        
        # Compute distances from object points to each part mesh
        labels = np.full(len(points), Config.NO_CONTACT_INDEX, dtype=np.int64)
        
        for i, point in enumerate(points):
            best_part = Config.NO_CONTACT_INDEX
            best_distance = float('inf')
            for part_id, part_mesh in enumerate(part_meshes):
                # Proper mesh distance using trimesh nearest surface
                _, distance, _ = part_mesh.nearest.on_surface([point])
                distance = float(distance[0])
                if distance < best_distance:
                    best_distance = distance
                    best_part = part_id
            if best_distance < self.contact_threshold:
                labels[i] = best_part
        
        return labels
    
    def _generate_prompt(self, obj_id: str, seq_id: str) -> str:
        """Generate functional description for the object."""
        if obj_id in self.semantic_prompts:
            return self.semantic_prompts[obj_id]
        
        # Parse intent from sequence ID
        intent_code = seq_id.split('_')[1] if '_' in seq_id else "0000"
        
        # Generate default prompts based on object category
        category_prompts = {
            "bottle": "grasp the bottle to pour liquid",
            "bowl": "hold the bowl to contain food",
            "cup": "grip the cup handle to drink",
            "hammer": "hold the hammer handle to strike",
            "scissors": "grip the scissors to cut paper",
            "camera": "hold the camera to take photos",
        }
        
        # Extract category from object ID
        for cat, prompt in category_prompts.items():
            if cat in obj_id.lower():
                return prompt
        
        # Default prompt
        return f"functionally grasp the {obj_id.replace('_', ' ')}"


class OakInkPartDataset(OakInkDataset):
    """
    Extended dataset that uses part-based semantic annotations from OakBase.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.part_annotations = self._load_part_annotations()
    
    def _load_part_annotations(self) -> Dict:
        """Load part-based semantic attributes from OakBase."""
        annotations = {}
        oakbase_dir = self.root_dir / "OakBase"
        
        if oakbase_dir.exists():
            for cat_dir in oakbase_dir.iterdir():
                if cat_dir.is_dir():
                    for obj_dir in cat_dir.iterdir():
                        if obj_dir.is_dir():
                            obj_parts = []
                            for part_file in obj_dir.glob("part_*.json"):
                                with open(part_file, 'r') as f:
                                    part_data = json.load(f)
                                    obj_parts.append(part_data)
                            if obj_parts:
                                annotations[obj_dir.name] = obj_parts
        
        return annotations
    
    def _generate_prompt(self, obj_id: str, seq_id: str) -> str:
        """Generate prompt using part-based semantic attributes."""
        if obj_id in self.part_annotations:
            parts = self.part_annotations[obj_id]
            
            # Build functional description from part attributes
            actions = []
            for part in parts:
                attrs = part.get("attr", [])
                name = part.get("name", "part")
                
                if "held_by_hand" in attrs:
                    actions.append(f"grasp the {name}")
                if "flow_out_sth" in attrs:
                    actions.append(f"use the {name} to pour")
                if "contain_sth" in attrs:
                    actions.append(f"use the {name} to hold contents")
                if "support_sth" in attrs:
                    actions.append(f"use the {name} as support")
            
            if actions:
                return " and ".join(actions[:2])  # Combine top 2 actions
        
        # Fallback to parent method
        return super()._generate_prompt(obj_id, seq_id)


def create_oakink_loaders(
    root_dir: str,
    render_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split_mode: str = "split0",
    use_part_dataset: bool = True,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test data loaders for OakInk dataset.

    Args:
        root_dir: Path to OakInk dataset root
        render_dir: Path to rendered object images
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        split_mode: Data split to use
        use_part_dataset: Whether to use part-based semantic annotations
        distributed: Whether to use DistributedSampler
        world_size: Number of distributed processes
        rank: Current process rank
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        train_loader, test_loader
    """
    dataset_class = OakInkPartDataset if use_part_dataset else OakInkDataset

    train_dataset = dataset_class(
        root_dir=root_dir,
        render_dir=render_dir,
        split="train",
        split_mode=split_mode,
        **dataset_kwargs
    )

    test_dataset = dataset_class(
        root_dir=root_dir,
        render_dir=render_dir,
        split="test",
        split_mode=split_mode,
        **dataset_kwargs
    )

    # Custom collate function to handle our batch format
    def collate_fn(batch_list):
        """Collate function that maintains List[List[PIL.Image]] format."""
        batch = {
            'images_list': [item['images_list'][0] for item in batch_list],
            'texts_list': [item['texts_list'][0] for item in batch_list],
            'points': torch.cat([item['points'] for item in batch_list], dim=0),
            'contact_labels': torch.cat([item['contact_labels'] for item in batch_list], dim=0),
            'pose': torch.cat([item['pose'] for item in batch_list], dim=0),
            'meta': [item['meta'] for item in batch_list]
        }
        return batch

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader
