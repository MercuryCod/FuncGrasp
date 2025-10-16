"""
MANO Utilities

Functions for computing and rendering MANO hand meshes from pose parameters.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, Union


def get_mano_layer(mano_assets_root: Optional[str] = None):
    """
    Get MANO layer instance.

    Args:
        mano_assets_root: Path to MANO assets. If None, uses default location.

    Returns:
        ManoLayer instance
    """
    from manotorch.manolayer import ManoLayer

    if mano_assets_root is None:
        # Default to local manotorch assets (in parent directory)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mano_assets_root = os.path.join(script_dir, "manotorch", "assets", "mano_v1_2")

    mano_layer = ManoLayer(
        center_idx=0, mano_assets_root=mano_assets_root  # Use wrist as center
    )

    return mano_layer


def mano_forward(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mano_layer=None,
    return_numpy: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass through MANO to get hand joints, vertices, and faces.

    Args:
        mano_pose: MANO pose parameters, shape (48,) or (B, 48)
        mano_shape: MANO shape parameters, shape (10,) or (B, 10)
        mano_trans: Optional translation to apply, shape (3,) or (B, 3)
        mano_layer: ManoLayer instance (created if None)
        return_numpy: Whether to return numpy arrays (True) or torch tensors (False)

    Returns:
        Tuple of (joints, vertices, faces):
            - joints: (21, 3) or (B, 21, 3) hand joint positions
            - vertices: (778, 3) or (B, 778, 3) hand mesh vertices
            - faces: (1538, 3) triangle face indices (same for all batches)
    """
    # Convert to torch if needed
    if isinstance(mano_pose, np.ndarray):
        mano_pose = torch.from_numpy(mano_pose)
    if isinstance(mano_shape, np.ndarray):
        mano_shape = torch.from_numpy(mano_shape)
    if mano_trans is not None and isinstance(mano_trans, np.ndarray):
        mano_trans = torch.from_numpy(mano_trans)

    # Ensure float32
    mano_pose = mano_pose.float()
    mano_shape = mano_shape.float()
    if mano_trans is not None:
        mano_trans = mano_trans.float()

    # Add batch dimension if needed
    single_sample = False
    if mano_pose.ndim == 1:
        mano_pose = mano_pose.unsqueeze(0)
        mano_shape = mano_shape.unsqueeze(0)
        if mano_trans is not None:
            mano_trans = mano_trans.unsqueeze(0)
        single_sample = True

    # Create MANO layer if not provided
    if mano_layer is None:
        mano_layer = get_mano_layer()

    # Forward pass
    with torch.no_grad():
        mano_output = mano_layer(mano_pose, mano_shape)

    # Get outputs
    joints = mano_output.joints  # (B, 21, 3)
    vertices = mano_output.verts  # (B, 778, 3)
    faces = mano_layer.th_faces  # (1538, 3)

    # Apply translation if provided
    if mano_trans is not None:
        joints = joints + mano_trans.unsqueeze(1)
        vertices = vertices + mano_trans.unsqueeze(1)

    # Remove batch dimension if single sample
    if single_sample:
        joints = joints.squeeze(0)
        vertices = vertices.squeeze(0)

    # Convert to numpy if requested
    if return_numpy:
        joints = joints.cpu().numpy()
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()

    return joints, vertices, faces


def render_hand_mesh(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mano_layer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render hand mesh from MANO parameters.

    This is a simplified wrapper around mano_forward that returns only
    vertices and faces (what you need for rendering).

    Args:
        mano_pose: MANO pose parameters, shape (48,) or (B, 48)
        mano_shape: MANO shape parameters, shape (10,) or (B, 10)
        mano_trans: Optional translation, shape (3,) or (B, 3)
        mano_layer: ManoLayer instance (created if None)

    Returns:
        Tuple of (vertices, faces):
            - vertices: (778, 3) or (B, 778, 3) mesh vertices
            - faces: (1538, 3) triangle faces
    """
    _, vertices, faces = mano_forward(
        mano_pose, mano_shape, mano_trans, mano_layer=mano_layer, return_numpy=True
    )

    return vertices, faces


def joints_to_mesh(
    hand_joints: Union[np.ndarray, torch.Tensor],
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_layer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given hand joints, compute the full hand mesh.

    Note: This function assumes you have MANO pose and shape parameters.
    The joints are primarily used to determine the translation.

    Args:
        hand_joints: Hand joint positions, shape (21, 3)
        mano_pose: MANO pose parameters, shape (48,)
        mano_shape: MANO shape parameters, shape (10,)
        mano_layer: ManoLayer instance (created if None)

    Returns:
        Tuple of (vertices, faces):
            - vertices: (778, 3) mesh vertices
            - faces: (1538, 3) triangle faces
    """
    # Convert to numpy if needed
    if isinstance(hand_joints, torch.Tensor):
        hand_joints = hand_joints.cpu().numpy()

    # Compute mesh without translation first
    _, verts_no_trans, faces = mano_forward(
        mano_pose, mano_shape, mano_trans=None, mano_layer=mano_layer, return_numpy=True
    )

    # Compute joints without translation to find the offset
    joints_no_trans, _, _ = mano_forward(
        mano_pose, mano_shape, mano_trans=None, mano_layer=mano_layer, return_numpy=True
    )

    # Calculate translation as difference between given joints and computed joints
    # Use wrist (joint 0) as reference
    translation = hand_joints[0] - joints_no_trans[0]

    # Apply translation to vertices
    vertices = verts_no_trans + translation[None, :]

    return vertices, faces


def create_hand_trimesh(
    vertices: np.ndarray, faces: np.ndarray, color: Optional[np.ndarray] = None
):
    """
    Create a trimesh object from hand vertices and faces.

    Args:
        vertices: (778, 3) or (N, 3) vertex positions
        faces: (1538, 3) or (M, 3) triangle faces
        color: Optional color array (3,) RGB or (N, 3) per-vertex colors

    Returns:
        trimesh.Trimesh object
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Add color if provided
    if color is not None:
        if color.ndim == 1:  # Single color for all vertices
            color = np.tile(color, (len(vertices), 1))
        mesh.visual.vertex_colors = (color * 255).astype(np.uint8)

    return mesh


def visualize_hand_mesh_k3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str = "hand_mesh.html",
    color: int = 0xFFB6C1,  # Light pink
    point_size: float = 2.0,
) -> str:
    """
    Visualize hand mesh using k3d and save to HTML.

    Args:
        vertices: (778, 3) mesh vertices
        faces: (1538, 3) triangle faces
        output_path: Output HTML file path
        color: Color of the mesh (hexadecimal)
        point_size: Size of points if rendering as point cloud

    Returns:
        Path to saved HTML file
    """
    import k3d

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    # Create plot
    plot = k3d.plot()

    # Add mesh
    mesh = k3d.mesh(vertices=vertices, indices=faces, color=color, side="double")
    plot += mesh

    # Set camera
    plot.camera_auto_fit = True

    # Save to HTML
    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    print(f"Hand mesh visualization saved to: {output_path}")
    return output_path


def visualize_hand_and_object_k3d(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    obj_points: np.ndarray,
    output_path: str = "hand_object.html",
    hand_color: int = 0xFFB6C1,  # Light pink
    obj_color: int = 0x87CEEB,  # Sky blue
    point_size: float = 2.0,
) -> str:
    """
    Visualize hand mesh and object point cloud together using k3d.

    Args:
        hand_vertices: (778, 3) hand mesh vertices
        hand_faces: (1538, 3) triangle faces
        obj_points: (N, 3) object point cloud
        output_path: Output HTML file path
        hand_color: Color of hand mesh (hexadecimal)
        obj_color: Color of object points (hexadecimal)
        point_size: Size of object points

    Returns:
        Path to saved HTML file
    """
    import k3d

    hand_vertices = hand_vertices.astype(np.float32)
    hand_faces = hand_faces.astype(np.uint32)
    obj_points = obj_points.astype(np.float32)

    # Create plot
    plot = k3d.plot()

    # Add hand mesh
    hand_mesh = k3d.mesh(
        vertices=hand_vertices, indices=hand_faces, color=hand_color, side="double"
    )
    plot += hand_mesh

    # Add object point cloud
    obj_cloud = k3d.points(
        positions=obj_points, point_size=point_size, color=obj_color, shader="3d"
    )
    plot += obj_cloud

    # Set camera
    plot.camera_auto_fit = True

    # Save to HTML
    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    print(f"Hand-object visualization saved to: {output_path}")
    return output_path
