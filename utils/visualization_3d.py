"""
3D visualization utilities for hand-object interaction.
Provides functions for creating 3D plots with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple


def create_hand_object_figure(
    object_points: np.ndarray,
    contact_labels: np.ndarray,
    hand_vertices: Optional[np.ndarray] = None,
    hand_joints: Optional[np.ndarray] = None,
    show_joints: bool = False,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create a figure with 3D visualization of hand-object interaction.
    
    Args:
        object_points: (N, 3) array of object points
        contact_labels: (N,) array of contact labels (0-6)
        hand_vertices: Optional (V, 3) array of hand mesh vertices
        hand_joints: Optional (21, 3) array of hand joints
        show_joints: Whether to show hand joints and skeleton
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Define colors for each contact class
    contact_colors = get_contact_colors()
    
    # Create subplots
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Plot 3D view
    plot_3d_view(ax1, object_points, contact_labels, hand_vertices, 
                 hand_joints if show_joints else None, contact_colors)
    
    # Plot top-down view (XY)
    plot_2d_view(ax2, object_points, contact_labels, hand_vertices,
                 hand_joints if show_joints else None, contact_colors, 
                 dims=(0, 1), title='Top-down View (XY)')
    
    # Plot side view (XZ)
    plot_2d_view(ax3, object_points, contact_labels, hand_vertices,
                 hand_joints if show_joints else None, contact_colors,
                 dims=(0, 2), title='Side View (XZ)')
    
    # Add statistics
    contact_points = np.sum(contact_labels != 6)
    total_points = len(contact_labels)
    fig.suptitle(f'Hand-Object Interaction Analysis\nContact Points: {contact_points}/{total_points} ({contact_points/total_points*100:.1f}%)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def get_contact_colors() -> Dict[int, str]:
    """Get color mapping for contact classes."""
    return {
        0: '#FF0000',   # thumb - red
        1: '#FF8C00',   # index - orange
        2: '#FFD700',   # middle - gold
        3: '#00FF00',   # ring - green
        4: '#0000FF',   # little - blue
        5: '#9400D3',   # palm - violet
        6: '#C0C0C0'    # no_contact - silver/gray
    }


def get_contact_class_names() -> List[str]:
    """Get names for contact classes."""
    return ['Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Palm', 'No Contact']


def plot_3d_view(ax, object_points: np.ndarray, contact_labels: np.ndarray,
                 hand_vertices: Optional[np.ndarray], hand_joints: Optional[np.ndarray],
                 contact_colors: Dict[int, str]):
    """Plot 3D view of hand-object interaction."""
    class_names = get_contact_class_names()
    
    # Plot object points with contact colors
    for class_idx in range(7):
        mask = contact_labels == class_idx
        if np.any(mask):
            ax.scatter(object_points[mask, 0], object_points[mask, 1], object_points[mask, 2],
                      c=contact_colors[class_idx], s=8, alpha=0.7,
                      label=f"{class_names[class_idx]} ({np.sum(mask)})")
    
    # Add hand mesh if available
    if hand_vertices is not None:
        ax.scatter(hand_vertices[:, 0], hand_vertices[:, 1], hand_vertices[:, 2],
                  c='red', alpha=0.3, s=1, label='Hand mesh')
    
    # Add hand joints if requested
    if hand_joints is not None:
        plot_hand_skeleton(ax, hand_joints, plot_3d=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D View')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.view_init(elev=20, azim=45)


def plot_2d_view(ax, object_points: np.ndarray, contact_labels: np.ndarray,
                 hand_vertices: Optional[np.ndarray], hand_joints: Optional[np.ndarray],
                 contact_colors: Dict[int, str], dims: Tuple[int, int], title: str):
    """Plot 2D projection of hand-object interaction."""
    dim1, dim2 = dims
    
    # Plot object points (contact points with higher alpha)
    for class_idx in range(7):
        mask = contact_labels == class_idx
        if np.any(mask) and class_idx != 6:  # Don't plot no_contact for clarity
            ax.scatter(object_points[mask, dim1], object_points[mask, dim2],
                      c=contact_colors[class_idx], s=10, alpha=0.7)
    
    # Plot no_contact points with lower alpha
    mask_no_contact = contact_labels == 6
    ax.scatter(object_points[mask_no_contact, dim1], object_points[mask_no_contact, dim2],
              c=contact_colors[6], s=5, alpha=0.3)
    
    # Plot hand mesh if available
    if hand_vertices is not None:
        ax.scatter(hand_vertices[:, dim1], hand_vertices[:, dim2],
                  c='red', alpha=0.3, s=1)
    
    # Plot hand joints if requested
    if hand_joints is not None:
        plot_hand_skeleton_2d(ax, hand_joints, dims)
    
    ax.set_xlabel(['X', 'Y', 'Z'][dim1])
    ax.set_ylabel(['X', 'Y', 'Z'][dim2])
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)


def plot_hand_skeleton(ax, hand_joints: np.ndarray, plot_3d: bool = True):
    """Plot hand skeleton with joint connections."""
    # Plot joints
    if plot_3d:
        ax.scatter(hand_joints[:, 0], hand_joints[:, 1], hand_joints[:, 2],
                  c='red', s=80, marker='o', edgecolors='black', linewidth=2, alpha=0.9)
    
    # Draw skeleton connections
    connections = get_hand_connections()
    for finger in connections:
        for i in range(len(finger)-1):
            if finger[i] < len(hand_joints) and finger[i+1] < len(hand_joints):
                if plot_3d:
                    ax.plot([hand_joints[finger[i], 0], hand_joints[finger[i+1], 0]],
                           [hand_joints[finger[i], 1], hand_joints[finger[i+1], 1]],
                           [hand_joints[finger[i], 2], hand_joints[finger[i+1], 2]], 
                           'r-', linewidth=2, alpha=0.8)


def plot_hand_skeleton_2d(ax, hand_joints: np.ndarray, dims: Tuple[int, int]):
    """Plot hand skeleton in 2D projection."""
    dim1, dim2 = dims
    
    # Plot joints
    ax.scatter(hand_joints[:, dim1], hand_joints[:, dim2],
              c='red', s=100, marker='o', edgecolors='black', linewidth=2, alpha=0.9)
    
    # Draw connections
    connections = get_hand_connections()
    for finger in connections:
        for i in range(len(finger)-1):
            if finger[i] < len(hand_joints) and finger[i+1] < len(hand_joints):
                ax.plot([hand_joints[finger[i], dim1], hand_joints[finger[i+1], dim1]],
                       [hand_joints[finger[i], dim2], hand_joints[finger[i+1], dim2]], 
                       'r-', linewidth=2, alpha=0.8)


def get_hand_connections() -> List[List[int]]:
    """Get hand joint connections for skeleton visualization."""
    return [
        [0, 1, 2, 3, 4],        # Thumb
        [0, 5, 6, 7, 8],        # Index
        [0, 9, 10, 11, 12],     # Middle
        [0, 13, 14, 15, 16],    # Ring
        [0, 17, 18, 19, 20]     # Little
    ]


def create_contact_distribution_plot(contact_labels: np.ndarray, ax=None) -> None:
    """Create a bar plot showing contact distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    contact_counts = [np.sum(contact_labels == i) for i in range(7)]
    class_names = get_contact_class_names()
    colors = get_contact_colors()
    
    bars = ax.bar(class_names, contact_counts, color=[colors[i] for i in range(7)])
    ax.set_ylabel('Number of Points')
    ax.set_title('Contact Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, contact_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   str(count), ha='center', va='bottom')
    
    if ax is None:
        plt.tight_layout()
        return fig
