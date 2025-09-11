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


class OakInkDataset(Dataset):
    """
    OakInk dataset loader for functional grasp training.
    Loads multi-view images, hand poses, object meshes, and computes contact maps.
    """
    
    VIEWS = ["north_east", "south_east", "north_west", "south_west"]
    HAND_JOINTS = 21  # MANO convention
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_mode: str = "split0",
        n_points: int = 1024,
        contact_threshold: float = 0.01,  # 1cm threshold for contact
        use_cache: bool = True,
        single_view: bool = True,
        view_idx: int = 0,
        semantic_prompts: Optional[Dict] = None
    ):
        """
        Args:
            root_dir: Path to OakInk dataset root
            split: 'train' or 'test'
            split_mode: 'split0', 'split0_ho', 'split1', or 'split2'
            n_points: Number of points to sample from object mesh
            contact_threshold: Distance threshold for contact detection (meters)
            use_cache: Whether to cache processed data
            single_view: Use single view or multi-view
            view_idx: Which view to use if single_view
            semantic_prompts: Dict mapping object IDs to functional descriptions
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_points = n_points
        self.contact_threshold = contact_threshold
        self.use_cache = use_cache
        self.single_view = single_view
        self.view_idx = view_idx
        self.semantic_prompts = semantic_prompts or {}
        
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
                - images: PIL Images or list of images
                - texts: Functional description string
                - points: Object point cloud (N, 3)
                - contact_labels: Binary contact labels (N,)
                - pose: Hand pose (wrist + joints)
                - meta: Additional metadata
        """
        # Parse sequence info
        seq_info = self.sequences[idx]
        seq_id, timestamp, frame_idx, view_idx = seq_info
        
        # Create frame identifier
        frame_id = f"{seq_id.replace('/', '__')}__{timestamp}__{frame_idx}__{view_idx}"
        
        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f"{frame_id}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Load annotations
        hand_joints = self._load_hand_joints(frame_id)
        obj_transform = self._load_obj_transform(frame_id)
        cam_intrinsic = self._load_cam_intrinsic(frame_id)
        
        # Load image(s)
        images = self._load_images(seq_id, timestamp, frame_idx)
        
        # Extract object ID from sequence
        obj_id = self._get_object_id(seq_id)
        
        # Load and transform object mesh
        points = self._load_object_points(obj_id, obj_transform)
        
        # Compute contact labels
        contact_labels = self._compute_contact_labels(hand_joints, points)
        
        # Generate semantic prompt
        text = self._generate_prompt(obj_id, seq_id)
        
        # Convert hand joints to pose representation
        pose = self._joints_to_pose(hand_joints)
        
        # Prepare output
        batch = {
            "images": images,
            "texts": text,
            "points": torch.tensor(points, dtype=torch.float32),
            "contact_labels": torch.tensor(contact_labels, dtype=torch.float32),
            "pose": torch.tensor(pose, dtype=torch.float32),
            "meta": {
                "seq_id": seq_id,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "cam_intrinsic": cam_intrinsic
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
    
    def _load_images(self, seq_id: str, timestamp: str, frame_idx: int) -> List[Image.Image]:
        """Load RGB images from specified views."""
        images = []
        base_path = self.root_dir / "image" / "stream_release_v2" / seq_id / timestamp
        
        if self.single_view:
            view = self.VIEWS[self.view_idx]
            img_path = base_path / f"{view}_color_{frame_idx}.png"
            if img_path.exists():
                images.append(Image.open(img_path).convert('RGB'))
        else:
            for view in self.VIEWS:
                img_path = base_path / f"{view}_color_{frame_idx}.png"
                if img_path.exists():
                    images.append(Image.open(img_path).convert('RGB'))
        
        # Return single image if single view, else list
        return images[0] if self.single_view and images else images
    
    def _get_object_id(self, seq_id: str) -> str:
        """Extract object ID from sequence ID."""
        # Format: A{object_code}_{intent}_{seq}
        # Example: A01001_0002_0000 -> object 01001
        parts = seq_id.split('_')
        if parts[0].startswith('A'):
            obj_code = parts[0][1:]  # Remove 'A' prefix
            # Map to actual object ID if needed
            return self._map_object_code(obj_code)
        return parts[0]
    
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
        """Load object mesh and sample points."""
        # Try different mesh locations
        mesh_paths = [
            self.root_dir / "shape" / "OakInkObjectsV2" / obj_id / "align" / "model_align.obj",
            self.root_dir / "shape" / "OakInkVirtualObjectsV2" / obj_id / "align" / "model_align.obj",
        ]
        
        mesh = None
        for path in mesh_paths:
            if path.exists():
                mesh = trimesh.load(str(path), force='mesh')
                break
        
        if mesh is None:
            # Fallback: create dummy point cloud
            points = np.random.randn(self.n_points, 3) * 0.1
        else:
            # Sample points from mesh surface
            points, _ = trimesh.sample.sample_surface_even(mesh, self.n_points)
            
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
    
    def _joints_to_pose(self, joints: np.ndarray) -> np.ndarray:
        """Convert 3D joint positions to pose representation."""
        # Simple representation: wrist position + relative joint positions
        wrist = joints[0]
        relative_joints = joints[1:] - wrist
        
        # Flatten to vector: [wrist_xyz, relative_joints_flat]
        pose = np.concatenate([wrist, relative_joints.flatten()])
        return pose[:28]  # Truncate to match DPOSE=28 from pipeline
    
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
    batch_size: int = 8,
    num_workers: int = 4,
    split_mode: str = "split0",
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test data loaders for OakInk dataset.
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = OakInkPartDataset(
        root_dir=root_dir,
        split="train",
        split_mode=split_mode,
        **dataset_kwargs
    )
    
    test_dataset = OakInkPartDataset(
        root_dir=root_dir,
        split="test",
        split_mode=split_mode,
        **dataset_kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader