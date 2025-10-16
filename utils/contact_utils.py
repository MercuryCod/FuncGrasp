"""
Contact Point Computation Utilities

Functions for computing contact points between MANO hand and object point clouds,
with finger-specific labeling.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, Union
from utils.mano_utils import mano_forward


# MANO joint topology - which joints belong to which finger
FINGER_JOINTS = {
    "thumb": [1, 2, 3, 4],  # Thumb joints
    "index": [5, 6, 7, 8],  # Index finger joints
    "middle": [9, 10, 11, 12],  # Middle finger joints
    "ring": [13, 14, 15, 16],  # Ring finger joints
    "pinky": [17, 18, 19, 20],  # Pinky finger joints
    "palm": [0],  # Wrist/palm
}

# Finger ID mapping for classification (7 classes total)
FINGER_IDS = {
    "no_contact": 0,  # No contact
    "palm": 1,  # Palm contact
    "thumb": 2,  # Thumb contact
    "index": 3,  # Index finger contact
    "middle": 4,  # Middle finger contact
    "ring": 5,  # Ring finger contact
    "pinky": 6,  # Pinky finger contact
}


def compute_contact_points(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Union[np.ndarray, torch.Tensor],
    obj_points: Union[np.ndarray, torch.Tensor],
    contact_threshold: float = 0.005,  # 5mm in meters
    use_vertices: bool = True,  # Use hand vertices (778) instead of just joints (21)
) -> Dict[str, np.ndarray]:
    """
    Compute contact points between hand and object.

    For each object point, finds the closest hand point and determines:
    1. Whether it's in contact (distance < threshold)
    2. Which finger it's closest to
    3. The distance to that finger

    Args:
        mano_pose: (48,) MANO pose parameters
        mano_shape: (10,) MANO shape parameters
        mano_trans: (3,) Translation
        obj_points: (N, 3) Object point cloud
        contact_threshold: Distance threshold for contact (in meters)
        use_vertices: Use hand mesh vertices (778) for more accurate contact,
                     or just joints (21) for faster computation

    Returns:
        Dictionary containing:
            - contact_mask: (N,) boolean array, True if point is in contact
            - contact_distances: (N,) distance to closest hand point
            - finger_labels: (N,) integer labels indicating which finger (0-5)
            - finger_names: (N,) string labels ('palm', 'thumb', 'index', etc.)
            - contact_points: (M, 3) subset of object points in contact
            - contact_finger_labels: (M,) finger labels for contact points only
    """
    # Convert to numpy
    if isinstance(obj_points, torch.Tensor):
        obj_points = obj_points.cpu().numpy()

    # Compute hand geometry
    joints, vertices, faces = mano_forward(
        mano_pose, mano_shape, mano_trans, return_numpy=True
    )

    # Choose hand representation
    if use_vertices:
        hand_points = vertices  # (778, 3) - more accurate
        # Create mapping from vertex index to finger
        vertex_to_finger = _create_vertex_to_finger_mapping(joints, vertices)
    else:
        hand_points = joints  # (21, 3) - faster
        # Create mapping from joint index to finger
        joint_to_finger = _create_joint_to_finger_mapping()

    # Compute distances from each object point to all hand points
    # obj_points: (N, 3), hand_points: (M, 3)
    # distances: (N, M)
    distances = np.linalg.norm(obj_points[:, None, :] - hand_points[None, :, :], axis=2)

    # Find closest hand point for each object point
    min_distances = distances.min(axis=1)  # (N,)
    closest_hand_idx = distances.argmin(axis=1)  # (N,)

    # Determine contact
    contact_mask = min_distances < contact_threshold

    # Assign finger labels (1-6 for hand parts)
    if use_vertices:
        finger_labels_contact = np.array(
            [vertex_to_finger[idx] for idx in closest_hand_idx]
        )
    else:
        finger_labels_contact = np.array(
            [joint_to_finger[idx] for idx in closest_hand_idx]
        )

    # Create final labels: 0 for no contact, 1-6 for finger/palm contact
    finger_labels = np.where(contact_mask, finger_labels_contact, 0)

    # Get finger names
    finger_id_to_name = {v: k for k, v in FINGER_IDS.items()}
    finger_names = np.array([finger_id_to_name[label] for label in finger_labels])

    # Extract contact points
    contact_points = obj_points[contact_mask]
    contact_finger_labels = finger_labels[contact_mask]
    contact_finger_names = finger_names[contact_mask]

    # Compute per-finger distances for soft target computation
    per_finger_distances = _compute_per_finger_distances(
        obj_points, hand_points, vertex_to_finger if use_vertices else None
    )
    
    return {
        "contact_mask": contact_mask,
        "contact_distances": min_distances,
        "finger_labels": finger_labels,
        "finger_names": finger_names,
        "contact_points": contact_points,
        "contact_finger_labels": contact_finger_labels,
        "contact_finger_names": contact_finger_names,
        "per_finger_distances": per_finger_distances,  # NEW: [N, 6] distances to each part
    }


def _compute_per_finger_distances(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    point_to_finger: Dict[int, int] = None
) -> np.ndarray:
    """
    Compute distance from each object point to each of the 6 hand parts separately.
    
    Args:
        obj_points: [N, 3] object point cloud
        hand_points: [M, 3] hand points (vertices or joints)
        point_to_finger: Dict mapping hand point index → finger ID (1-6)
                        If None, creates joint mapping
    
    Returns:
        per_finger_distances: [N, 6] minimum distance from each object point to each hand part
            Index 0: distance to palm (finger_id=1)
            Index 1: distance to thumb (finger_id=2)
            Index 2: distance to index (finger_id=3)
            Index 3: distance to middle (finger_id=4)
            Index 4: distance to ring (finger_id=5)
            Index 5: distance to pinky (finger_id=6)
    """
    N = len(obj_points)
    
    # Create mapping if not provided
    if point_to_finger is None:
        point_to_finger = _create_joint_to_finger_mapping()
    
    # Group hand points by finger
    finger_points = {finger_id: [] for finger_id in range(1, 7)}
    for point_idx, finger_id in point_to_finger.items():
        finger_points[finger_id].append(hand_points[point_idx])
    
    # Convert to arrays
    for finger_id in range(1, 7):
        if finger_points[finger_id]:
            finger_points[finger_id] = np.array(finger_points[finger_id])
        else:
            # No points for this finger (shouldn't happen with valid MANO)
            finger_points[finger_id] = np.array([[0, 0, 0]])  # Placeholder
    
    # Compute distance from each object point to each finger
    per_finger_distances = np.zeros((N, 6), dtype=np.float32)
    
    for finger_idx, finger_id in enumerate(range(1, 7)):  # 1=palm, 2=thumb, ..., 6=pinky
        finger_pts = finger_points[finger_id]  # [M_finger, 3]
        
        # Compute distances: [N, M_finger]
        dists = np.linalg.norm(
            obj_points[:, None, :] - finger_pts[None, :, :],
            axis=2
        )
        
        # Minimum distance to this finger
        min_dist_to_finger = dists.min(axis=1)  # [N]
        
        # Store in column (finger_id 1-6 → index 0-5)
        per_finger_distances[:, finger_idx] = min_dist_to_finger
    
    return per_finger_distances


def _create_joint_to_finger_mapping() -> Dict[int, int]:
    """
    Create mapping from joint index to finger ID.
    Returns IDs 1-6 (excluding 0 which is 'no_contact').
    """
    mapping = {}
    for finger_name, joint_indices in FINGER_JOINTS.items():
        finger_id = FINGER_IDS[
            finger_name
        ]  # Will be 1-6 (palm, thumb, index, middle, ring, pinky)
        for joint_idx in joint_indices:
            mapping[joint_idx] = finger_id
    return mapping


def _create_vertex_to_finger_mapping(
    joints: np.ndarray, vertices: np.ndarray
) -> Dict[int, int]:
    """
    Create mapping from vertex index to finger ID.

    For each vertex, finds the closest joint and assigns it to that joint's finger.

    Args:
        joints: (21, 3) hand joints
        vertices: (778, 3) hand mesh vertices

    Returns:
        Dictionary mapping vertex index to finger ID
    """
    # For each vertex, find closest joint
    distances = np.linalg.norm(
        vertices[:, None, :] - joints[None, :, :], axis=2
    )  # (778, 21)

    closest_joints = distances.argmin(axis=1)  # (778,)

    # Map joints to fingers
    joint_to_finger = _create_joint_to_finger_mapping()

    # Map vertices to fingers via closest joint
    vertex_to_finger = {}
    for vertex_idx, joint_idx in enumerate(closest_joints):
        vertex_to_finger[vertex_idx] = joint_to_finger[joint_idx]

    return vertex_to_finger


def compute_finger_contact_statistics(
    contact_info: Dict[str, np.ndarray],
) -> Dict[str, any]:
    """
    Compute statistics about finger-specific contacts.

    Args:
        contact_info: Output from compute_contact_points()

    Returns:
        Dictionary with contact statistics per finger
    """
    stats = {
        "total_contact_points": len(contact_info["contact_points"]),
        "contact_rate": contact_info["contact_mask"].mean(),
        "per_finger": {},
    }

    # Count contacts per finger
    for finger_name in FINGER_IDS.keys():
        finger_id = FINGER_IDS[finger_name]

        # Count total object points closest to this finger
        total_closest = (contact_info["finger_labels"] == finger_id).sum()

        # Count contact points for this finger
        contact_count = (contact_info["contact_finger_labels"] == finger_id).sum()

        # Average distance for points closest to this finger
        finger_mask = contact_info["finger_labels"] == finger_id
        if finger_mask.sum() > 0:
            avg_distance = contact_info["contact_distances"][finger_mask].mean()
            min_distance = contact_info["contact_distances"][finger_mask].min()
        else:
            avg_distance = float("inf")
            min_distance = float("inf")

        stats["per_finger"][finger_name] = {
            "contact_points": int(contact_count),
            "closest_points": int(total_closest),
            "avg_distance": float(avg_distance),
            "min_distance": float(min_distance),
        }

    return stats


def compute_per_finger_distances(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Union[np.ndarray, torch.Tensor],
    obj_points: Union[np.ndarray, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    For each object point, compute distance to each finger separately.

    This provides detailed per-finger distance information useful for training
    finger-specific contact prediction models.

    Args:
        mano_pose: (48,) MANO pose parameters
        mano_shape: (10,) MANO shape parameters
        mano_trans: (3,) Translation
        obj_points: (N, 3) Object point cloud

    Returns:
        Dictionary with:
            - per_finger_distances: Dict[finger_name, (N,)] distances to each finger
            - finger_contact_masks: Dict[finger_name, (N,)] boolean contact masks
    """
    # Convert to numpy
    if isinstance(obj_points, torch.Tensor):
        obj_points = obj_points.cpu().numpy()

    # Compute hand geometry
    joints, vertices, faces = mano_forward(
        mano_pose, mano_shape, mano_trans, return_numpy=True
    )

    # Create vertex to finger mapping
    vertex_to_finger = _create_vertex_to_finger_mapping(joints, vertices)

    # Group vertices by finger
    finger_vertices = {finger: [] for finger in FINGER_IDS.keys()}
    for vertex_idx, finger_id in vertex_to_finger.items():
        finger_name = next(k for k, v in FINGER_IDS.items() if v == finger_id)
        finger_vertices[finger_name].append(vertices[vertex_idx])

    # Convert to arrays
    for finger in finger_vertices:
        if finger_vertices[finger]:
            finger_vertices[finger] = np.array(finger_vertices[finger])

    # Compute distances from each object point to each finger
    per_finger_distances = {}
    finger_contact_masks = {}

    for finger_name, finger_verts in finger_vertices.items():
        if len(finger_verts) == 0:
            per_finger_distances[finger_name] = np.full(
                len(obj_points), np.inf, dtype=np.float32
            )
            finger_contact_masks[finger_name] = np.zeros(len(obj_points), dtype=bool)
            continue

        # Compute distances to all vertices of this finger
        distances = np.linalg.norm(
            obj_points[:, None, :] - finger_verts[None, :, :], axis=2
        )  # (N, num_finger_verts)

        # Minimum distance to this finger
        min_distances = distances.min(axis=1)  # (N,)
        per_finger_distances[finger_name] = min_distances

        # Contact mask (within 5mm)
        finger_contact_masks[finger_name] = min_distances < 0.005

    return {
        "per_finger_distances": per_finger_distances,
        "finger_contact_masks": finger_contact_masks,
    }


def get_contact_heatmap(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Union[np.ndarray, torch.Tensor],
    obj_points: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Create a contact heatmap showing distance from each object point to hand.

    Args:
        mano_pose: (48,) MANO pose parameters
        mano_shape: (10,) MANO shape parameters
        mano_trans: (3,) Translation
        obj_points: (N, 3) Object point cloud

    Returns:
        heatmap: (N,) distance values (in meters)
    """
    contact_info = compute_contact_points(
        mano_pose,
        mano_shape,
        mano_trans,
        obj_points,
        contact_threshold=float("inf"),  # Include all points
        use_vertices=True,
    )

    return contact_info["contact_distances"]
