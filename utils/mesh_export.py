"""
3D mesh export utilities for hand-object scenes.
Provides functions for exporting to various 3D formats.
"""

import numpy as np
import trimesh
import pickle
from typing import Optional, Dict, Any
from pathlib import Path


def export_hand_object_scene(
    object_points: np.ndarray,
    contact_labels: np.ndarray,
    hand_vertices: np.ndarray,
    hand_faces: Optional[np.ndarray] = None,
    mano_model_path: str = 'assets/mano_v1_2/models/MANO_RIGHT.pkl',
    output_dir: str = 'output',
    filename_prefix: str = 'scene'
) -> Dict[str, Path]:
    """
    Export hand-object scene to various 3D formats.
    
    Args:
        object_points: (N, 3) array of object points
        contact_labels: (N,) array of contact labels (0-6)
        hand_vertices: (V, 3) array of hand mesh vertices
        hand_faces: Optional (F, 3) array of hand mesh faces. If None, loads from MANO model
        mano_model_path: Path to MANO model file
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filenames
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load MANO faces if not provided
    if hand_faces is None:
        hand_faces = load_mano_faces(mano_model_path)
    
    # Create hand mesh
    hand_mesh = create_hand_mesh(hand_vertices, hand_faces)
    
    # Create object point cloud with colors
    object_cloud = create_colored_point_cloud(object_points, contact_labels)
    
    # Create scene
    scene = trimesh.Scene([hand_mesh, object_cloud])
    
    # Export to various formats
    output_paths = {}
    
    # GLB format (widely supported)
    glb_path = output_dir / f"{filename_prefix}.glb"
    scene.export(str(glb_path))
    output_paths['glb'] = glb_path
    
    # PLY format (good for point clouds)
    ply_path = output_dir / f"{filename_prefix}.ply"
    scene.export(str(ply_path))
    output_paths['ply'] = ply_path
    
    # OBJ format (widely supported, but may lose colors)
    obj_path = output_dir / f"{filename_prefix}.obj"
    try:
        scene.export(str(obj_path))
        output_paths['obj'] = obj_path
    except:
        pass  # OBJ export might fail for point clouds
    
    # Export individual components
    hand_mesh_path = output_dir / f"{filename_prefix}_hand.ply"
    hand_mesh.export(str(hand_mesh_path))
    output_paths['hand_mesh'] = hand_mesh_path
    
    object_cloud_path = output_dir / f"{filename_prefix}_object.ply"
    object_cloud.export(str(object_cloud_path))
    output_paths['object_cloud'] = object_cloud_path
    
    return output_paths


def load_mano_faces(mano_model_path: str) -> np.ndarray:
    """Load MANO face indices from model file."""
    with open(mano_model_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']


def create_hand_mesh(vertices: np.ndarray, faces: np.ndarray, 
                     color: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Create a trimesh object for the hand.
    
    Args:
        vertices: (V, 3) array of vertices
        faces: (F, 3) array of face indices
        color: Optional RGBA color (default: light red with transparency)
        
    Returns:
        trimesh.Trimesh object
    """
    if color is None:
        color = [255, 200, 200, 200]  # Light red with transparency
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = color
    return mesh


def create_colored_point_cloud(points: np.ndarray, labels: np.ndarray) -> trimesh.PointCloud:
    """
    Create a colored point cloud based on contact labels.
    
    Args:
        points: (N, 3) array of points
        labels: (N,) array of labels (0-6)
        
    Returns:
        trimesh.PointCloud object
    """
    # Color mapping for contact classes
    color_map = {
        0: [255, 0, 0, 255],      # thumb - red
        1: [255, 140, 0, 255],    # index - orange
        2: [255, 215, 0, 255],    # middle - gold
        3: [0, 255, 0, 255],      # ring - green
        4: [0, 0, 255, 255],      # little - blue
        5: [148, 0, 211, 255],    # palm - violet
        6: [192, 192, 192, 100]   # no_contact - gray with transparency
    }
    
    colors = np.array([color_map[label] for label in labels])
    return trimesh.PointCloud(vertices=points, colors=colors)


def export_contact_info(
    contact_labels: np.ndarray,
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export contact information to a text file.
    
    Args:
        contact_labels: (N,) array of contact labels
        output_path: Path to save the info file
        additional_info: Optional dictionary with additional information
    """
    class_names = ['thumb', 'index', 'middle', 'ring', 'little', 'palm', 'no_contact']
    
    with open(output_path, 'w') as f:
        f.write("Contact Information\n")
        f.write("==================\n\n")
        
        # Write contact statistics
        total_points = len(contact_labels)
        f.write(f"Total points: {total_points}\n\n")
        
        f.write("Contact distribution:\n")
        for i, name in enumerate(class_names):
            count = np.sum(contact_labels == i)
            percentage = (count / total_points) * 100
            f.write(f"  {name}: {count} points ({percentage:.1f}%)\n")
        
        # Write summary
        contact_points = np.sum(contact_labels != 6)
        contact_percentage = (contact_points / total_points) * 100
        f.write(f"\nTotal contact points: {contact_points} ({contact_percentage:.1f}%)\n")
        
        # Write additional info if provided
        if additional_info:
            f.write("\nAdditional Information:\n")
            for key, value in additional_info.items():
                f.write(f"  {key}: {value}\n")
