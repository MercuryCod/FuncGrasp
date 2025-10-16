"""
Visualization utilities for OakInk dataset.

Provides functions to visualize hand meshes, object point clouds,
and hand-object interactions using k3d (HTML output).
"""

import numpy as np
import torch
from typing import Optional, Union, Dict
import k3d
from utils.mano_utils import mano_forward


def visualize_hand_from_mano(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Union[np.ndarray, torch.Tensor],
    output_path: str = "hand_visualization.html",
    hand_color: int = 0xFFB6C1,  # Light pink
) -> str:
    """
    Visualize hand mesh from MANO parameters.

    Args:
        mano_pose: (48,) MANO pose parameters
        mano_shape: (10,) MANO shape parameters
        mano_trans: (3,) Translation
        output_path: Output HTML file path
        hand_color: Color of hand mesh (hexadecimal)

    Returns:
        Path to saved HTML file
    """
    # Compute hand mesh
    joints, vertices, faces = mano_forward(
        mano_pose, mano_shape, mano_trans, return_numpy=True
    )

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    # Create plot
    plot = k3d.plot()

    # Add hand mesh
    hand_mesh = k3d.mesh(
        vertices=vertices, indices=faces, color=hand_color, side="double"
    )
    plot += hand_mesh

    # Optionally add joints as spheres
    joints = joints.astype(np.float32)
    for i, joint in enumerate(joints):
        sphere = k3d.points(
            positions=joint.reshape(1, 3),
            point_size=0.01,
            color=0xFF0000,  # Red
            shader="3d",
        )
        plot += sphere

    plot.camera_auto_fit = True

    # Save to HTML
    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    print(f"Hand visualization saved to: {output_path}")
    return output_path


def visualize_object_pointcloud(
    obj_points: Union[np.ndarray, torch.Tensor],
    output_path: str = "object_pointcloud.html",
    color: int = 0x87CEEB,  # Sky blue
    point_size: Optional[float] = None,
    contact_info: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    Visualize object point cloud, optionally with contact information.

    Args:
        obj_points: (N, 3) object point cloud
        output_path: Output HTML file path
        color: Color of points (hexadecimal) - used only if contact_info is None
        point_size: Size of points (None = auto-scale)
        contact_info: Optional dictionary from compute_contact_points() to color by finger

    Returns:
        Path to saved HTML file
    """
    if isinstance(obj_points, torch.Tensor):
        obj_points = obj_points.cpu().numpy()

    obj_points = obj_points.astype(np.float32)

    # Auto-scale point size based on object dimensions
    if point_size is None:
        extent = np.max(obj_points, axis=0) - np.min(obj_points, axis=0)
        extent_norm = float(np.linalg.norm(extent))
        point_size = float(max(0.005, extent_norm / 100.0))

    # Create plot
    plot = k3d.plot()

    # Determine coloring
    if contact_info is not None:
        # Color by finger contact
        # Color map for 7-class finger labels
        finger_colors = {
            0: 0x404040,  # no_contact - dark gray
            1: 0x0000FF,  # palm - blue
            2: 0xFF0000,  # thumb - red
            3: 0x00FF00,  # index - green
            4: 0xFFA500,  # middle - orange
            5: 0xFFFF00,  # ring - yellow
            6: 0xFF00FF,  # pinky - magenta
        }

        # Create per-point colors directly from finger labels (0-6)
        point_colors = np.array(
            [finger_colors[label] for label in contact_info["finger_labels"]],
            dtype=np.uint32,
        )

        # Add colored point cloud
        point_cloud = k3d.points(
            positions=obj_points,
            colors=point_colors,
            point_size=point_size,
            shader="3d",
        )
    else:
        # Single color
        point_cloud = k3d.points(
            positions=obj_points, point_size=point_size, color=color, shader="3d"
        )

    plot += point_cloud
    plot.camera_auto_fit = True

    # Save to HTML
    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    print(f"Object visualization saved to: {output_path}")
    if contact_info is not None:
        print(f"  Contact points: {contact_info['contact_mask'].sum()}")
    return output_path


def visualize_grasp(
    mano_pose: Union[np.ndarray, torch.Tensor],
    mano_shape: Union[np.ndarray, torch.Tensor],
    mano_trans: Union[np.ndarray, torch.Tensor],
    obj_points: Union[np.ndarray, torch.Tensor],
    output_path: str = "grasp_visualization.html",
    hand_color: int = 0xFFB6C1,  # Light pink
    obj_color: int = 0x87CEEB,  # Sky blue
    show_joints: bool = True,
    point_size: Optional[float] = None,
    contact_info: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    Visualize complete grasp: hand mesh + object point cloud.

    Args:
        mano_pose: (48,) MANO pose parameters
        mano_shape: (10,) MANO shape parameters
        mano_trans: (3,) Translation
        obj_points: (N, 3) object point cloud
        output_path: Output HTML file path
        hand_color: Color of hand mesh
        obj_color: Color of object points (used only if contact_info is None)
        show_joints: Whether to show hand joints as spheres
        point_size: Size of object points (None = auto-scale)
        contact_info: Optional dictionary from compute_contact_points() to color by finger

    Returns:
        Path to saved HTML file
    """
    # Convert to numpy
    if isinstance(obj_points, torch.Tensor):
        obj_points = obj_points.cpu().numpy()

    # Compute hand mesh
    joints, vertices, faces = mano_forward(
        mano_pose, mano_shape, mano_trans, return_numpy=True
    )

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)
    obj_points = obj_points.astype(np.float32)

    # Auto-scale point size
    if point_size is None:
        extent = np.max(obj_points, axis=0) - np.min(obj_points, axis=0)
        extent_norm = float(np.linalg.norm(extent))
        point_size = float(max(0.005, extent_norm / 100.0))

    # Create plot
    plot = k3d.plot()

    # Add hand mesh
    hand_mesh = k3d.mesh(
        vertices=vertices, indices=faces, color=hand_color, side="double", opacity=0.9
    )
    plot += hand_mesh

    # Add object point cloud with optional contact coloring
    if contact_info is not None:
        # Color by finger contact (7-class system)
        finger_colors = {
            0: 0x303030,  # no_contact - dark gray
            1: 0x808080,  # palm - gray
            2: 0xFF0000,  # thumb - red
            3: 0x00FF00,  # index - green
            4: 0x0000FF,  # middle - blue
            5: 0xFFFF00,  # ring - yellow
            6: 0xFF00FF,  # pinky - magenta
        }

        # Create per-point colors directly from finger labels (0-6)
        finger_labels = contact_info["finger_labels"]
        if isinstance(finger_labels, torch.Tensor):
            finger_labels = finger_labels.cpu().numpy()
        finger_labels = finger_labels.astype(np.int32)
        
        point_colors = np.array([finger_colors[label] for label in finger_labels], dtype=np.uint32)

        obj_cloud = k3d.points(
            positions=obj_points,
            colors=point_colors,
            point_size=point_size,
            shader="3d",
        )
    else:
        # Single color
        obj_cloud = k3d.points(
            positions=obj_points, point_size=point_size, color=obj_color, shader="3d"
        )

    plot += obj_cloud

    # Optionally add joints
    if show_joints:
        joints = joints.astype(np.float32)
        joint_cloud = k3d.points(
            positions=joints,
            point_size=point_size * 2,
            color=0xFF0000,  # Red
            shader="3d",
        )
        plot += joint_cloud

    plot.camera_auto_fit = True

    # Save to HTML
    with open(output_path, "w") as f:
        f.write(plot.get_snapshot())

    if contact_info is not None:
        print(f"Grasp visualization with contacts saved to: {output_path}")
        print(f"  Contact points: {contact_info['contact_mask'].sum()}")
    else:
        print(f"Grasp visualization saved to: {output_path}")

    return output_path

