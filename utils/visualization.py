"""
Visualization utilities for hand grasps.
Minimal utilities for joint-based hand visualization.
"""

import numpy as np
from scipy.spatial import ConvexHull


def create_hand_mesh_from_joints(joints, hand_scale=1.0):
    """
    Create a simplified hand mesh from 21 joints.
    
    Args:
        joints: (21, 3) array of joint positions
        hand_scale: scale factor for hand thickness
        
    Returns:
        vertices: (N, 3) array of mesh vertices
        faces: (M, 3) array of triangle faces
    """
    # Define finger connections
    fingers = {
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20]
    }
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Create cylinders for finger bones
    for finger_name, joint_indices in fingers.items():
        for i in range(len(joint_indices) - 1):
            start = joints[joint_indices[i]]
            end = joints[joint_indices[i + 1]]
            
            # Skip if bone is too short
            bone_vec = end - start
            bone_len = np.linalg.norm(bone_vec)
            if bone_len < 0.001:
                continue
            
            # Create cylinder
            radius = 0.008 * hand_scale if finger_name == 'thumb' else 0.006 * hand_scale
            n_segments = 6  # Reduced for cleaner mesh
            
            # Generate cylinder vertices
            bone_dir = bone_vec / bone_len
            perp1 = np.cross(bone_dir, [0, 0, 1] if abs(bone_dir[2]) < 0.9 else [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(bone_dir, perp1)
            
            for t in [0, 1]:  # Start and end
                center = start + t * bone_vec
                for k in range(n_segments):
                    angle = 2 * np.pi * k / n_segments
                    offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                    all_vertices.append(center + offset)
            
            # Create faces
            for k in range(n_segments):
                v1 = vertex_offset + k
                v2 = vertex_offset + (k + 1) % n_segments
                v3 = vertex_offset + n_segments + k
                v4 = vertex_offset + n_segments + (k + 1) % n_segments
                
                all_faces.extend([[v1, v2, v3], [v2, v4, v3]])
            
            vertex_offset += 2 * n_segments
    
    # Create palm mesh
    palm_joints = joints[[0, 1, 5, 9, 13, 17]]
    palm_center = palm_joints.mean(axis=0)
    
    # Add palm vertices with thickness
    palm_normal = np.cross(palm_joints[2] - palm_joints[0], palm_joints[4] - palm_joints[0])
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    
    palm_thickness = 0.005 * hand_scale
    for joint in palm_joints:
        all_vertices.append(joint + palm_thickness * palm_normal)
        all_vertices.append(joint - palm_thickness * palm_normal)
    
    # Create palm faces using convex hull
    palm_vertices = np.array(all_vertices[vertex_offset:])
    if len(palm_vertices) >= 4:
        try:
            hull = ConvexHull(palm_vertices)
            for face in hull.simplices:
                all_faces.append([vertex_offset + face[0], 
                                vertex_offset + face[1], 
                                vertex_offset + face[2]])
        except:
            pass  # Skip palm if convex hull fails
    
    return np.array(all_vertices), np.array(all_faces)


# Note: For interactive 3D visualization, use visualize_grasp.py instead
# This module now only contains the hand mesh generation utility
