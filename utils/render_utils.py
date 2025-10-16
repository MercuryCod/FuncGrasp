#!/usr/bin/env python3
"""
Utilities for rendering object meshes from multiple viewpoints.
"""
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


VIEWS = {
    'front':  (0,   0,   2.5),
    'back':   (180, 0,   2.5),
    'left':   (-90, 0,   2.5),  # Looking from the left side
    'right':  (90,  0,   2.5),  # Looking from the right side
    'top':    (0,   90,  2.5),
    'bottom': (0,  -90,  2.5),
}


def normalize_mesh(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize mesh vertices to unit sphere.
    
    Args:
        vertices: (N, 3) vertex positions
    
    Returns:
        Normalized vertices centered at origin and scaled to unit sphere
    """
    center = vertices.mean(axis=0)
    verts = vertices - center
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts = verts / scale
    return verts


def render_mesh_view(vertices: np.ndarray, faces: np.ndarray, 
                     azim: float, elev: float,
                     image_size: int = 512, zoom: float = 1.6) -> np.ndarray:
    """
    Render a single view of a mesh.
    
    Args:
        vertices: (N, 3) normalized vertex positions
        faces: (M, 3) triangle face indices
        azim: Azimuth angle in degrees
        elev: Elevation angle in degrees
        image_size: Output image size in pixels
        zoom: Zoom factor - higher values fill more of frame
    
    Returns:
        (image_size, image_size, 3) RGB image as uint8 array
    """
    dpi = 100
    figsize = image_size / dpi
    lim = 1.0 / zoom  # Higher zoom -> tighter crop
    
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)  # Remove margins
    ax = fig.add_subplot(111, projection='3d', facecolor=(1, 1, 1))
    
    # Set camera view
    ax.view_init(elev=elev, azim=azim)
    
    # Use orthographic projection
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    
    # Render mesh surface
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces,
        color=(0.7, 0.7, 0.7),
        edgecolor='none',
        shade=True,
        alpha=1.0,
        antialiased=True,
    )
    
    # Set tight limits (zoomed in)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    # Force cubic aspect ratio
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    
    # Clean appearance - hide axes
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_visible(False)
        axis.set_ticks([])
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(image_size, image_size, 4)[:, :, :3]  # Drop alpha
    
    plt.close(fig)
    return img


def render_mesh_multiview(mesh_path: str, output_folder: str, 
                          object_id: str = None,
                          image_size: int = 512, 
                          zoom: float = 1.6,
                          views: Optional[list] = None,
                          verbose: bool = False) -> Dict[str, str]:
    """
    Render mesh from multiple viewpoints and save images.
    
    Args:
        mesh_path: Path to .obj mesh file
        output_folder: Output directory for images
        object_id: Object identifier (if None, uses mesh filename)
        image_size: Output image size (default 512)
        zoom: Zoom factor - higher values fill more of frame (default 1.6)
        views: List of view names to render (default: all 6 views)
        verbose: Print progress messages
    
    Returns:
        Dictionary mapping view names to saved image paths
    """
    if views is None:
        views = list(VIEWS.keys())
    
    # Load and normalize mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = normalize_mesh(mesh.vertices)
    faces = mesh.faces
    
    # Create output directory
    if object_id is None:
        object_id = Path(mesh_path).stem
    
    obj_out_dir = Path(output_folder) / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    for view_name in views:
        if view_name not in VIEWS:
            continue
        
        azim, elev, _ = VIEWS[view_name]
        
        # Render view
        img = render_mesh_view(vertices, faces, azim, elev, image_size, zoom)
        
        # Save to file
        output_path = obj_out_dir / f"{view_name}.png"
        Image.fromarray(img).save(output_path)
        
        saved_paths[view_name] = str(output_path)
        
        if verbose:
            print(f"  {view_name}: {output_path}")
    
    return saved_paths

