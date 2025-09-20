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
            self.sequences = json.load(f)
        
        # Load object metadata
        meta_file = self.root_dir / "shape" / "metaV2" / "object_id.json"
        with open(meta_file, 'r') as f:
            self.object_meta = json.load(f)
        
        # Cache directory
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
                - contact_labels: Binary contact labels [1, N]
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
        obj_transform = self._load_obj_transform(frame_id)
        
        # Transform hand joints to object frame if requested
        if self.transform_to_object_frame:
            hand_joints = self._transform_joints_to_object_frame(hand_joints, obj_transform)
        
        # Load and transform object mesh
        points = self._load_object_points(obj_id, obj_transform)
        
        # Compute contact labels using hand joints (expanded surface)
        contact_labels = self._compute_contact_labels(hand_joints, points)
        
        # Generate semantic prompt
        text = self._generate_prompt(obj_id, seq_id)
        
        # Convert joints to 63D flattened representation
        pose = hand_joints.flatten()  # (21, 3) -> (63,)
        
        # Prepare output in batch format for pipeline compatibility
        batch = {
            "images_list": [images],  # List[List[PIL.Image]] with B=1
            "texts_list": [text],     # List[str] with B=1
            "points": torch.tensor(points, dtype=torch.float32).unsqueeze(0),  # [1, N, 3]
            "contact_labels": torch.tensor(contact_labels, dtype=torch.float32).unsqueeze(0),  # [1, N]
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
    
    def _load_rendered_images(self, obj_id: str) -> List[Image.Image]:
        """Load rendered object images."""
        images = []
        obj_render_dir = self.render_dir / obj_id
        
        # Load specified number of views
        for i, view in enumerate(self.OBJECT_VIEWS[:self.num_views]):
            img_path = obj_render_dir / f"{view}.png"
            if img_path.exists():
                images.append(Image.open(img_path).convert('RGB'))
            else:
                raise FileNotFoundError(f"Rendered image not found: {img_path}")
        
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
    
    def _map_object_code(self, obj_code: str) -> str:
        """Map object code to actual object ID."""
        # This would map codes like "01001" to actual object names
        # For now, return as-is or implement mapping logic
        code_to_id = {
            "01001": "bottle_s001",
            "01002": "bottle_s002",
            "02001": "bowl_s001",
            # Add more mappings as needed
        }
        return code_to_id.get(obj_code, f"object_{obj_code}")
    
    def _load_object_points(self, obj_id: str, transform: np.ndarray) -> np.ndarray:
        """Load object mesh and sample points using FPS."""
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
        for path in mesh_paths:
            if path.exists():
                mesh = trimesh.load(str(path), force='mesh')
                break
        
        if mesh is None:
            raise FileNotFoundError(f"Object mesh not found for {obj_id}. Tried paths: {mesh_paths}")
        else:
            # Sample more points initially for FPS
            initial_points, _ = trimesh.sample.sample_surface(mesh, self.n_points * 10)
            
            # Convert to torch for FPS
            points_torch = torch.tensor(initial_points, dtype=torch.float32)
            batch = torch.zeros(len(points_torch), dtype=torch.long)
            
            # Apply FPS to get exactly n_points
            fps_idx = fps(points_torch, batch, ratio=float(self.n_points) / len(points_torch))
            points = points_torch[fps_idx].numpy()
            
            # Apply transformation to world coordinates
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            points = (transform @ points_homo.T).T[:, :3]
        
        return points
    
    def _compute_contact_labels(self, hand_joints: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Compute binary contact labels based on hand-object proximity."""
        # Compute pairwise distances between hand joints and object points
        # hand_joints: (21, 3), points: (N, 3)
        
        # Expand hand joints to include interpolated palm/finger surfaces
        hand_points = self._expand_hand_surface(hand_joints)
        
        # Compute minimum distance from each object point to hand
        contact_labels = np.zeros(len(points))
        for i, point in enumerate(points):
            distances = np.linalg.norm(hand_points - point, axis=1)
            min_dist = np.min(distances)
            contact_labels[i] = 1.0 if min_dist < self.contact_threshold else 0.0
        
        return contact_labels
    
    def _expand_hand_surface(self, joints: np.ndarray, n_interp: int = 5) -> np.ndarray:
        """Expand hand joints to approximate hand surface."""
        # MANO joint order: wrist(1) + thumb(4) + fingers(4x4)
        expanded = [joints[0]]  # Wrist
        
        # Finger chains
        finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 1):
                # Interpolate between consecutive joints
                start = joints[chain[i]]
                end = joints[chain[i + 1]]
                for t in np.linspace(0, 1, n_interp):
                    expanded.append(start + t * (end - start))
        
        return np.array(expanded)
    
    
    
    
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
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
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
        distributed: Whether to use DistributedSampler for FSDP
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

