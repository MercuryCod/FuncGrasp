"""
OakInk Dataset Loader

Unified PyTorch dataset for loading OakInk grasp data with:
- Text instructions (generated from category, affordances, and intent)
- MANO hand poses (with pre-computed joints and vertices)
- Object point clouds (centered and sampled from mesh)
- Metadata
"""

import os
import pickle
import json
import hashlib
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


# Constants from oikit
ALL_CATEGORIES = [
    "apple",
    "banana",
    "binoculars",
    "bottle",
    "bowl",
    "cameras",
    "can",
    "cup",
    "cylinder_bottle",
    "donut",
    "eyeglasses",
    "flashlight",
    "fryingpan",
    "gamecontroller",
    "hammer",
    "headphones",
    "knife",
    "lightbulb",
    "lotion_pump",
    "mouse",
    "mug",
    "pen",
    "phone",
    "pincer",
    "power_drill",
    "scissors",
    "screwdriver",
    "squeezable",
    "stapler",
    "teapot",
    "toothbrush",
    "trigger_sprayer",
    "wineglass",
    "wrench",
]

ALL_INTENTS = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}

CENTER_IDX = 9  # MANO joint index used as hand center


class OakInkDataset(Dataset):
    """
    Unified OakInk Dataset for grasp learning.
    
    Each sample contains:
    - Text instruction (generated from object category and intent)
    - MANO hand pose (48-dim pose + 10-dim shape + 3-dim translation)
    - Object point cloud (sampled from mesh)
    - Contact labels (finger-specific, 7 classes)
    - Multi-view object images (optional, pre-rendered from 6 viewpoints)
    - Metadata (category, object ID, intent, subject)
    
    Note: Hand joints and vertices are NOT pre-computed. Use mano_utils.mano_forward()
    to compute them from the MANO parameters when needed.
    
    Args:
        data_root: Path to OakInk root directory
        split: 'train', 'val', 'test', or 'all'
        num_points: Number of points to sample from object mesh
        compute_contacts: Whether to pre-compute contact labels (default: True)
        contact_threshold: Distance threshold for contact in meters (default: 0.01)
        load_object_images: Whether to load pre-rendered multi-view images (default: False)
        rendered_dir: Path to rendered images directory (default: {data_root}/rendered_objects)
        transform: Optional transform to apply to samples
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_points: int = 1024,
        compute_contacts: bool = True,
        contact_threshold: float = 0.01,
        load_object_images: bool = False,
        rendered_dir: Optional[str] = None,
        transform=None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.compute_contacts = compute_contacts
        self.contact_threshold = contact_threshold
        self.load_object_images = load_object_images
        self.transform = transform

        # Paths
        self.shape_root = self.data_root / "shape"
        self.grasp_root = self.shape_root / "oakink_shape_v2"
        self.objects_root = self.shape_root / "OakInkObjectsV2"
        self.virtual_objects_root = self.shape_root / "OakInkVirtualObjectsV2"
        self.meta_root = self.shape_root / "metaV2"
        
        # Rendered images directory
        if rendered_dir is None:
            self.rendered_dir = self.data_root / "rendered_objects"
        else:
            self.rendered_dir = Path(rendered_dir)

        # Load metadata
        self._load_metadata()

        # Build index
        self._build_index()
        
        # Validate rendered images if requested
        if self.load_object_images:
            self._validate_rendered_images()

        logging.info(f"Loaded {len(self)} grasp samples from OakInk ({self.split})")

    def _load_metadata(self):
        """Load object metadata."""
        object_id_file = self.meta_root / "object_id.json"
        virtual_id_file = self.meta_root / "virtual_object_id.json"

        self.object_metadata = {}

        if object_id_file.exists():
            with open(object_id_file, "r") as f:
                self.object_metadata.update(json.load(f))

        if virtual_id_file.exists():
            with open(virtual_id_file, "r") as f:
                self.object_metadata.update(json.load(f))

    def _get_object_split(self, obj_id: str) -> str:
        """Determine split based on object ID hash (following official oikit)."""
        obj_id_hash = int(hashlib.md5(obj_id.encode("utf-8")).hexdigest(), 16)
        hash_mod = obj_id_hash % 10

        if hash_mod < 8:
            return "train"
        elif hash_mod == 8:
            return "val"
        else:  # hash_mod == 9
            return "test"
    
    def _validate_rendered_images(self):
        """Validate that rendered images exist for all objects in the dataset."""
        if not self.rendered_dir.exists():
            logging.error(f"Rendered objects directory not found: {self.rendered_dir}")
            logging.error(f"Please run: bash scripts/prepare_renders.sh")
            raise FileNotFoundError(
                f"Rendered objects directory not found: {self.rendered_dir}\n"
                f"Please run: bash scripts/prepare_renders.sh"
            )
        
        # Check metadata file
        metadata_file = self.rendered_dir / "render_metadata.json"
        if not metadata_file.exists():
            logging.error(f"Render metadata not found: {metadata_file}")
            logging.error(f"The rendered_objects directory exists but appears incomplete")
            logging.error(f"Please run: bash scripts/prepare_renders.sh")
            raise FileNotFoundError(
                f"Render metadata not found: {metadata_file}\n"
                f"Please run: bash scripts/prepare_renders.sh"
            )
        
        # Load and log metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        logging.info(f"Using pre-rendered images from: {self.rendered_dir}")
        logging.info(f"  Rendered objects: {metadata.get('success', 0)}/{metadata.get('total', 0)}")
        if metadata.get('failed', 0) > 0:
            logging.warning(f"  {metadata['failed']} objects failed to render")
        
        # Check if all unique objects in dataset have renders
        unique_objects = set(sample['shape_id'] for sample in self.samples)
        missing_renders = []
        for obj_id in unique_objects:
            obj_dir = self.rendered_dir / obj_id
            if not obj_dir.exists():
                missing_renders.append(obj_id)
        
        if missing_renders:
            logging.warning(f"Missing renders for {len(missing_renders)} objects: {missing_renders[:5]}...")
            logging.warning(f"Please re-run: bash scripts/prepare_renders.sh")
        else:
            logging.info(f"All {len(unique_objects)} unique objects have pre-rendered images")
    
    def _load_rendered_images(self, shape_id: str) -> Dict[str, torch.Tensor]:
        """
        Load pre-rendered multi-view images for an object.
        
        Args:
            shape_id: Object identifier
        
        Returns:
            Dictionary mapping view names to image tensors (H, W, 3) in range [0, 1]
        
        Raises:
            FileNotFoundError: If rendered images directory or specific views don't exist
        """
        from PIL import Image
        
        obj_dir = self.rendered_dir / shape_id
        if not obj_dir.exists():
            logging.error(f"Rendered images not found for object: {shape_id}")
            logging.error(f"Expected location: {obj_dir}")
            logging.error(f"Please run: bash scripts/prepare_renders.sh")
            raise FileNotFoundError(f"Rendered images not found for object: {shape_id} at {obj_dir}")
        
        images = {}
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        missing_views = []
        
        for view in views:
            img_path = obj_dir / f"{view}.png"
            if not img_path.exists():
                missing_views.append(view)
                continue
            
            try:
                img = Image.open(img_path)
                img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
                images[view] = torch.from_numpy(img_array)  # (H, W, 3)
            except Exception as e:
                logging.error(f"Failed to load image {img_path}: {e}")
                missing_views.append(view)
        
        if missing_views:
            logging.error(f"Missing or corrupted views for object {shape_id}: {missing_views}")
            logging.error(f"Please re-run: bash scripts/prepare_renders.sh --force")
            raise FileNotFoundError(f"Views {missing_views} not found or corrupted for object {shape_id}")
        
        return images

    def _build_index(self):
        """Build an index of all grasp samples."""
        self.samples = []

        if not self.grasp_root.exists():
            raise FileNotFoundError(f"Grasp root not found: {self.grasp_root}")

        # Iterate through all categories
        for category in tqdm(ALL_CATEGORIES, desc="Building dataset index"):
            category_path = self.grasp_root / category
            if not category_path.exists():
                continue

            # Iterate through shape IDs
            for shape_dir in category_path.iterdir():
                if not shape_dir.is_dir():
                    continue

                shape_id = shape_dir.name  # e.g., S10001, C10001, O00001

                # Determine object split based on hash
                obj_split = self._get_object_split(shape_id)

                # Filter by split
                if self.split != "all" and obj_split != self.split:
                    continue

                # Iterate through grasp instances
                for grasp_instance_dir in shape_dir.iterdir():
                    if not grasp_instance_dir.is_dir():
                        continue

                    grasp_hash = grasp_instance_dir.name

                    # Get source info to check intent
                    source_file = grasp_instance_dir / "source.txt"
                    if not source_file.exists():
                        continue

                    with open(source_file, "r") as f:
                        source_seq = f.read().strip()

                    # Parse action_id from source (format: pass1/OBJID_ACTIONID_SUBJECT/timestamp/dom.pkl)
                    try:
                        parts = source_seq.split("/")
                        seq_parts = parts[1].split("_")
                        action_id = seq_parts[1]
                    except:
                        continue

                    # Load refined grasps (per-subject subdirectories)
                    for subject_dir in grasp_instance_dir.iterdir():
                        if not subject_dir.is_dir():
                            continue

                        subject_id = subject_dir.name
                        hand_param_file = subject_dir / "hand_param.pkl"

                        if hand_param_file.exists():
                            self.samples.append(
                                {
                                    "category": category,
                                    "shape_id": shape_id,
                                    "grasp_hash": grasp_hash,
                                    "hand_param_file": str(hand_param_file),
                                    "subject_id": subject_id,
                                    "action_id": action_id,
                                    "source_seq": source_seq,
                                }
                            )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_object_mesh_path(self, shape_id: str) -> Optional[str]:
        """Find object mesh path."""
        # Get object name from metadata
        if shape_id in self.object_metadata:
            obj_name = self.object_metadata[shape_id]["name"]
        else:
            obj_name = shape_id

        # Try both real and virtual object directories, both align_ds and align
        for base_root in [self.objects_root, self.virtual_objects_root]:
            for mesh_suffix in ["align_ds", "align"]:
                base_path = base_root / obj_name / mesh_suffix

                if base_path.exists():
                    # Look for OBJ or PLY files
                    for ext in ["*.obj", "*.ply"]:
                        mesh_files = list(base_path.glob(ext))
                        if mesh_files:
                            # Prefer files with 'align' or 'textured' in name
                            for f in mesh_files:
                                if "align" in f.name or "textured" in f.name:
                                    return str(f)
                            return str(mesh_files[0])

        return None

    def _load_object_pointcloud(self, shape_id: str) -> np.ndarray:
        """Load and sample object point cloud."""
        mesh_path = self._get_object_mesh_path(shape_id)

        if mesh_path is None:
            logging.warning(f"Object mesh not found for {shape_id}")
            return np.zeros((self.num_points, 3), dtype=np.float32)

        try:
            # Load mesh
            mesh = trimesh.load(
                mesh_path, process=False, force="mesh", skip_materials=True
            )

            # Center the object at bbox center
            bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
            mesh.vertices = mesh.vertices - bbox_center

            # Sample points from surface
            points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            return points.astype(np.float32)

        except Exception as e:
            logging.warning(f"Error loading mesh {mesh_path}: {e}")
            return np.zeros((self.num_points, 3), dtype=np.float32)

    def _generate_instruction(self, category: str, intent: str) -> str:
        """Generate text instruction from category and intent."""
        # Clean category name
        category_name = category.replace("_", " ")

        # Intent descriptions
        intent_map = {
            "0001": "use",
            "0002": "hold",
            "0003": "lift up",
            "0004": "hand over",
        }
        intent_verb = intent_map.get(intent, "grasp")

        # Build instruction
        if intent == "0004":  # handover
            return f"Hand over the {category_name}."
        elif intent == "0003":  # liftup
            return f"Lift up the {category_name}."
        elif intent == "0002":  # hold
            return f"Hold the {category_name}."
        else:  # use
            return f"Use the {category_name}."

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Load a single sample.

        Returns:
            Dictionary containing:
                - instruction: Text instruction string
                - mano_pose: (48,) MANO pose parameters
                - mano_shape: (10,) MANO shape parameters
                - mano_trans: (3,) Wrist translation in meters
                - obj_points: (N, 3) Object point cloud in meters (centered)
                - category: Object category name
                - shape_id: Shape identifier (object ID)
                - subject_id: Subject identifier
                - action_id: Action/intent ID ('0001', '0002', '0003', '0004')
                - intent: Intent name ('use', 'hold', 'liftup', 'handover')
                - contact_labels: (N,) finger contact labels 0-6 (if compute_contacts=True)
                    0=no_contact, 1=palm, 2=thumb, 3=index, 4=middle, 5=ring, 6=pinky
                - contact_mask: (N,) boolean, True if point is in contact (if compute_contacts=True)
                - contact_distances: (N,) float, distance to closest hand point in meters (if compute_contacts=True)
                - object_images: Dict[str, Tensor] mapping view names to images (H,W,3) in [0,1] (if load_object_images=True)
                    Views: 'front', 'back', 'left', 'right', 'top', 'bottom'
                
            Note: To compute hand geometry (joints/vertices), use:
                from utils.mano_utils import mano_forward
                joints, verts, faces = mano_forward(
                    sample['mano_pose'], sample['mano_shape'], sample['mano_trans']
                )
        """
        sample_info = self.samples[idx]

        # Load hand parameters
        with open(sample_info["hand_param_file"], "rb") as f:
            hand_params = pickle.load(f)

        # Extract MANO parameters
        mano_pose = hand_params["pose"].astype(np.float32)
        mano_shape = hand_params["shape"].astype(np.float32)
        mano_trans_raw = hand_params["tsl"].astype(np.float32)

        # Use raw translation from dataset (wrist position)
        # Note: To get hand joints/vertices, use mano_utils.mano_forward() separately
        mano_trans = mano_trans_raw

        # Load object point cloud
        obj_points = self._load_object_pointcloud(sample_info["shape_id"])

        # Get intent name
        intent_name = next(
            (k for k, v in ALL_INTENTS.items() if v == sample_info["action_id"]), "use"
        )

        # Generate instruction
        instruction = self._generate_instruction(
            sample_info["category"], sample_info["action_id"]
        )

        sample = {
            "instruction": instruction,
            "mano_pose": torch.from_numpy(mano_pose),
            "mano_shape": torch.from_numpy(mano_shape),
            "mano_trans": torch.from_numpy(mano_trans),
            "obj_points": torch.from_numpy(obj_points),
            "category": sample_info["category"],
            "shape_id": sample_info["shape_id"],
            "subject_id": sample_info["subject_id"],
            "action_id": sample_info["action_id"],
            "intent": intent_name,
        }
        
        # Compute contact labels if requested
        if self.compute_contacts:
            try:
                from utils.contact_utils import compute_contact_points
                
                contact_info = compute_contact_points(
                    mano_pose,
                    mano_shape,
                    mano_trans,
                    obj_points,
                    contact_threshold=self.contact_threshold,
                    use_vertices=True
                )
                
                # Add full contact info to sample
                sample["contact_labels"] = torch.from_numpy(contact_info['finger_labels'].astype(np.int64))
                sample["contact_mask"] = torch.from_numpy(contact_info['contact_mask'])
                sample["contact_distances"] = torch.from_numpy(contact_info['contact_distances'].astype(np.float32))
                sample["per_finger_distances"] = torch.from_numpy(contact_info['per_finger_distances'].astype(np.float32))
                
            except Exception as e:
                logging.warning(f"Contact computation failed for sample {idx}: {e}")
                # Fallback: all points labeled as no_contact (0)
                sample["contact_labels"] = torch.zeros(self.num_points, dtype=torch.int64)
                sample["contact_mask"] = torch.zeros(self.num_points, dtype=torch.bool)
                sample["contact_distances"] = torch.full((self.num_points,), float('inf'), dtype=torch.float32)
                sample["per_finger_distances"] = torch.full((self.num_points, 6), float('inf'), dtype=torch.float32)
        
        # Load pre-rendered object images if requested
        if self.load_object_images:
            try:
                sample["object_images"] = self._load_rendered_images(sample_info["shape_id"])
            except FileNotFoundError as e:
                # Re-raise with context about the sample
                logging.error(f"Failed to load rendered images for sample {idx} (object: {sample_info['shape_id']})")
                raise
            except Exception as e:
                # Unexpected error - log and re-raise
                logging.error(f"Unexpected error loading images for sample {idx} (object: {sample_info['shape_id']}): {e}")
                raise
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def get_statistics(self) -> Dict:
        """Compute dataset statistics."""
        stats = {
            "total_samples": len(self),
            "split": self.split,
            "categories": {},
            "intents": {},
        }

        # Count per category and intent
        for sample in self.samples:
            cat = sample["category"]
            action = sample["action_id"]
            intent = next((k for k, v in ALL_INTENTS.items() if v == action), "unknown")

            if cat not in stats["categories"]:
                stats["categories"][cat] = 0
            stats["categories"][cat] += 1

            if intent not in stats["intents"]:
                stats["intents"][intent] = 0
            stats["intents"][intent] += 1

        return stats


def get_dataloader(
    data_root: str,
    split: str = "train",
    num_points: int = 1024,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for OakInk dataset.

    Args:
        data_root: Path to OakInk root directory
        split: 'train', 'val', 'test', or 'all'
        num_points: Number of points to sample from object mesh
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for OakInkDataset

    Returns:
        PyTorch DataLoader
    """
    dataset = OakInkDataset(
        data_root=data_root, split=split, num_points=num_points, **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
