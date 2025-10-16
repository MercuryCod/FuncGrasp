"""
Data collation utilities for batching OakInk samples.

Converts dataset format to model-expected format:
- Stacks tensors (point clouds, MANO params, contact labels)
- Converts image tensors → PIL Images (for Qwen2.5-VL processor)
- Computes soft contact targets from per-finger distances
- Collects text instructions and metadata
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Dict


def compute_soft_targets_from_per_finger_distances(
    per_finger_distances: np.ndarray,
    tau: float = 0.01,
    min_prob: float = 1e-6
) -> np.ndarray:
    """
    Convert per-finger distances to soft probability distributions.
    
    Creates independent probability for each hand part based on distance.
    Closer distances → higher probabilities.
    
    Args:
        per_finger_distances: [N, 6] distances in meters
            Column 0: palm, 1: thumb, 2: index, 3: middle, 4: ring, 5: pinky
        tau: Temperature (default 0.01 = 10mm). Smaller = sharper probabilities
        min_prob: Minimum probability floor for numerical stability
    
    Returns:
        soft_targets: [N, 6] probability distribution (each row sums to ~1)
    
    Example:
        distances = [[0.002, 0.005, 0.050, 0.080, 0.100, 0.120]]  # 1 point
                     # palm   thumb  index  middle ring   pinky
        
        probs = exp(-distances / 0.01) = [0.819, 0.607, 0.007, 0.0003, ...]
        normalized = [0.573, 0.425, 0.005, ...] # Sums to 1.0
        
        Interpretation: Point is 57% likely palm contact, 42% likely thumb
    """
    # Gaussian kernel: P ∝ exp(-distance / tau)
    unnormalized_probs = np.exp(-per_finger_distances / tau)  # [N, 6]
    
    # Floor to prevent numerical issues
    unnormalized_probs = np.maximum(unnormalized_probs, min_prob)
    
    # Normalize so each point's probabilities sum to 1
    row_sums = unnormalized_probs.sum(axis=1, keepdims=True)
    soft_targets = unnormalized_probs / row_sums
    
    return soft_targets.astype(np.float32)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert image tensor to PIL Image.
    
    Args:
        tensor: [H, W, 3] float32 in range [0, 1]
    
    Returns:
        PIL.Image in RGB mode, uint8 in range [0, 255]
    """
    img_array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='RGB')


def collate_oakink_batch(batch: List[Dict], soft_target_tau: float = 0.01) -> Dict:
    """
    Collate OakInk samples into a batch for training.
    
    Performs the following conversions:
    1. Stack tensors (point clouds, MANO params, contact labels)
    2. Convert image tensors → PIL Images (for Qwen2.5-VL processor)
    3. Collect text instructions and metadata
    
    Args:
        batch: List of samples from OakInkDataset.__getitem__()
            Each sample is a dict with keys:
                - obj_points: [N, 3] object point cloud
                - mano_pose: [48] MANO pose parameters
                - mano_shape: [10] MANO shape parameters
                - mano_trans: [3] wrist translation
                - contact_labels: [N] finger-specific labels (0-6)
                - object_images: Dict[str, Tensor] (optional)
                - instruction: str
                - category, shape_id, subject_id, etc.
    
    Returns:
        Dictionary with:
            pts: [B, N, 3] object point clouds
            mano_params: [B, 61] MANO parameters (pose+shape+trans concatenated)
            contact_labels: [B, N] finger-specific labels (0-6)
                0=no_contact, 1=palm, 2-6=thumb/index/middle/ring/pinky
            images_list: List[List[PIL.Image]] for Qwen2.5-VL processor
            texts_list: List[str] text instructions
            metadata: Dict with categories, intents, shape_ids, subject_ids
    """
    B = len(batch)
    
    # Stack point clouds
    pts = torch.stack([s['obj_points'] for s in batch])  # [B, N, 3]
    
    # Stack and concatenate MANO parameters
    # Each sample has pose (48), shape (10), trans (3) → concatenate to 61-dim
    mano_params = torch.cat([
        torch.stack([s['mano_pose'] for s in batch]),     # [B, 48]
        torch.stack([s['mano_shape'] for s in batch]),    # [B, 10]
        torch.stack([s['mano_trans'] for s in batch])     # [B, 3]
    ], dim=-1)  # [B, 61]
    
    # Verify dimension
    assert mano_params.shape[1] == 61, \
        f"MANO params should be 61-dim, got {mano_params.shape[1]}"
    
    # Compute soft contact targets from per-finger distances
    soft_targets_list = []
    hard_labels_list = []  # Keep for visualization/metrics
    
    for s in batch:
        # Get per-finger distances [N, 6]
        per_finger_dists = s['per_finger_distances'].numpy()
        
        # Convert to soft probability distribution
        soft_targets = compute_soft_targets_from_per_finger_distances(
            per_finger_dists,
            tau=soft_target_tau
        )
        
        soft_targets_list.append(torch.from_numpy(soft_targets))
        hard_labels_list.append(s['contact_labels'])  # Keep original for visualization
    
    # Stack
    contact_soft_targets = torch.stack(soft_targets_list)  # [B, N, 6] - probability distributions
    contact_hard_labels = torch.stack(hard_labels_list)    # [B, N] - integer labels for visualization
    
    # Convert images: torch.Tensor → PIL.Image
    images_list = []
    for s in batch:
        if 'object_images' in s and s['object_images']:
            # Single-view strategy: use front view only
            # For multi-view, uncomment the alternative below
            front_img = tensor_to_pil(s['object_images']['front'])
            images_list.append([front_img])
            
            # Multi-view strategy (alternative):
            # views = ['front', 'left', 'right', 'top']
            # imgs = [tensor_to_pil(s['object_images'][v]) for v in views]
            # images_list.append(imgs)
        else:
            # No images available - text-only mode
            images_list.append([])
    
    # Collect text instructions
    texts_list = [s['instruction'] for s in batch]
    
    # Collect metadata (for logging and debugging)
    metadata = {
        'categories': [s['category'] for s in batch],
        'intents': [s['intent'] for s in batch],
        'shape_ids': [s['shape_id'] for s in batch],
        'subject_ids': [s['subject_id'] for s in batch],
    }
    
    return {
        'pts': pts,                           # [B, N, 3] object point clouds
        'mano_params': mano_params,           # [B, 61] MANO parameters (GT)
        'contact_soft_targets': contact_soft_targets,  # [B, N, 6] soft probability distributions
        'contact_hard_labels': contact_hard_labels,    # [B, N] hard labels for visualization
        'images_list': images_list,           # List[List[PIL.Image]] for Qwen
        'texts_list': texts_list,             # List[str] instructions
        'metadata': metadata,                 # Dict
    }


def collate_oakink_batch_multiview(batch: List[Dict], views: List[str] = None) -> Dict:
    """
    Collate OakInk samples with multiple views per object.
    
    Args:
        batch: List of samples from OakInkDataset.__getitem__()
        views: List of view names to use (default: ['front', 'left', 'right', 'top'])
    
    Returns:
        Same as collate_oakink_batch but with multiple images per sample
    """
    if views is None:
        views = ['front', 'left', 'right', 'top']
    
    # Same as single-view, but load multiple views
    B = len(batch)
    
    pts = torch.stack([s['obj_points'] for s in batch])
    
    mano_params = torch.cat([
        torch.stack([s['mano_pose'] for s in batch]),
        torch.stack([s['mano_shape'] for s in batch]),
        torch.stack([s['mano_trans'] for s in batch])
    ], dim=-1)
    
    contact_labels = torch.stack([s['contact_labels'] for s in batch])
    
    # Multi-view image loading
    images_list = []
    for s in batch:
        if 'object_images' in s and s['object_images']:
            imgs = [tensor_to_pil(s['object_images'][v]) for v in views if v in s['object_images']]
            images_list.append(imgs if imgs else [])
        else:
            images_list.append([])
    
    texts_list = [s['instruction'] for s in batch]
    
    metadata = {
        'categories': [s['category'] for s in batch],
        'intents': [s['intent'] for s in batch],
        'shape_ids': [s['shape_id'] for s in batch],
        'subject_ids': [s['subject_id'] for s in batch],
    }
    
    return {
        'pts': pts,
        'mano_params': mano_params,
        'contact_labels': contact_labels,
        'images_list': images_list,
        'texts_list': texts_list,
        'metadata': metadata,
    }

