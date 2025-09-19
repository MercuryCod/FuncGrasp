#!/usr/bin/env python3
"""
OakInk data preparation script.

Renders clean multi-view images for all objects using matplotlib.
Saves renders under: <output_dir>/<obj_id>/{front,left,right,back,top,bottom}.png
Writes mapping JSON: <output_dir>/object_renders.json mapping obj_id → dir

Usage:
  python -m dataset.prepare_data \
    --oakink_root /DATA/disk0/OakInk \
    --output_dir ./rendered_objects \
    --max_objects 200

  python -m dataset.prepare_data --oakink_root /DATA/disk0/OakInk --sample --num_samples 3
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


VIEWS = {
    'front': (0, 0, 2.5),
    'back': (180, 0, 2.5),
    'left': (90, 0, 2.5),
    'right': (-90, 0, 2.5),
    'top': (0, 90, 2.5),    # Positive elevation = looking down from above
    'bottom': (0, -90, 2.5)  # Negative elevation = looking up from below
}


class ObjectRenderer:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.dpi = 100
        self.figsize = image_size / self.dpi

    def _normalize_mesh(self, vertices: np.ndarray) -> np.ndarray:
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.max(np.linalg.norm(vertices, axis=1))
        if scale > 0:
            vertices = vertices / scale
        return vertices

    def render_views(self, mesh_path: str, views: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if views is None:
            views = list(VIEWS.keys())
        
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = self._normalize_mesh(mesh.vertices)
        faces = mesh.faces
        
        rendered_views = {}
        for view_name in views:
            if view_name not in VIEWS:
                continue
            
            azim, elev, _ = VIEWS[view_name]
            
            fig = plt.figure(figsize=(self.figsize, self.figsize), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                color=(0.7, 0.7, 0.7),
                edgecolor='none',
                shade=True,
                alpha=1.0
            )
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_axis_off()
            ax.grid(False)
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(self.image_size, self.image_size, 4)[:, :, :3]
            rendered_views[view_name] = image
            
            plt.close(fig)
        
        return rendered_views


def collect_object_meshes(oakink_root: str, max_objects: Optional[int] = None) -> Dict[str, Path]:
    root = Path(oakink_root)
    object_meshes: Dict[str, Path] = {}

    # image/obj/*.obj (A*, S*, Y*, etc.)
    obj_dir = root / "image" / "obj"
    if obj_dir.exists():
        for mesh_path in obj_dir.glob("*.obj"):
            object_meshes[mesh_path.stem] = mesh_path

    # shape/OakInkObjectsV2/*/align/model_align.obj preferred
    shape_dirs = [
        root / "shape" / "OakInkObjectsV2",
        root / "shape" / "OakInkVirtualObjectsV2",
    ]
    for sdir in shape_dirs:
        if not sdir.exists():
            continue
        for obj_dir in sdir.iterdir():
            if not obj_dir.is_dir():
                continue
            align_dir = obj_dir / "align"
            if align_dir.exists():
                meshes = list(align_dir.glob("*.obj"))
                if meshes:
                    object_meshes[obj_dir.name] = meshes[0]
                    continue
            meshes = list(obj_dir.glob("*.obj"))
            if meshes:
                object_meshes[obj_dir.name] = meshes[0]

    if max_objects:
        object_ids = list(object_meshes.keys())[:max_objects]
        object_meshes = {k: object_meshes[k] for k in object_ids}

    return object_meshes


def render_objects(oakink_root: str, output_dir: str, max_objects: Optional[int] = None,
                   views: Optional[List[str]] = None, image_size: int = 224) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    object_meshes = collect_object_meshes(oakink_root, max_objects=max_objects)
    print(f"Found {len(object_meshes)} unique objects to render")

    renderer = ObjectRenderer(image_size=image_size)
    render_mapping: Dict[str, str] = {}
    failed: List[str] = []

    for obj_id, mesh_path in tqdm(object_meshes.items(), desc="Rendering objects"):
        try:
            obj_out = output / obj_id
            obj_out.mkdir(exist_ok=True)
            imgs = renderer.render_views(str(mesh_path), views=views)
            for name, arr in imgs.items():
                Image.fromarray(arr).save(obj_out / f"{name}.png")
            render_mapping[obj_id] = str(obj_out)
        except Exception as e:
            print(f"Error rendering {obj_id}: {e}")
            failed.append(obj_id)

    mapping_file = output / "object_renders.json"
    with open(mapping_file, 'w') as f:
        json.dump(render_mapping, f, indent=2)

    if failed:
        failed_file = output / "failed_objects.json"
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"Failed to render {len(failed)} objects; see {failed_file}")

    print(f"Rendered {len(render_mapping)} objects → {output}")
    return render_mapping


def render_samples(oakink_root: str, output_dir: str, num_samples: int = 3, image_size: int = 224):
    mapping = collect_object_meshes(oakink_root, max_objects=None)
    keys = list(mapping.keys())[:num_samples]
    subset = {k: mapping[k] for k in keys}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderer = ObjectRenderer(image_size=image_size)
    for i, (obj_id, mesh_path) in enumerate(subset.items()):
        print(f"Rendering sample {i+1}/{len(subset)}: {obj_id}")
        imgs = renderer.render_views(str(mesh_path), views=['front', 'left', 'top'])
        obj_out = out / f"sample_{i}_{obj_id}"
        obj_out.mkdir(exist_ok=True)
        for name, arr in imgs.items():
            Image.fromarray(arr).save(obj_out / f"{name}.png")
    print(f"Samples saved to {out}")


def parse_views(views_arg: Optional[str]) -> Optional[List[str]]:
    if not views_arg:
        return None
    parts = [v.strip() for v in views_arg.split(',') if v.strip()]
    return parts if parts else None


def main():
    parser = argparse.ArgumentParser(description="Prepare OakInk data: render multi-view object images")
    parser.add_argument("--oakink_root", type=str, required=True, help="Path to OakInk dataset root")
    parser.add_argument("--output_dir", type=str, default="rendered_objects", help="Directory to save renders")
    parser.add_argument("--views", type=str, default=None, help="Comma-separated views (default: all)")
    parser.add_argument("--image_size", type=int, default=1024, help="Output image size")
    parser.add_argument("--max_objects", type=int, default=None, help="Maximum number of objects to render")
    parser.add_argument("--sample", action="store_true", help="Render a small sample instead of all objects")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples when --sample is set")

    args = parser.parse_args()

    views = parse_views(args.views)
    if args.sample:
        render_samples(args.oakink_root, args.output_dir, num_samples=args.num_samples,
                       image_size=args.image_size)
    else:
        render_objects(args.oakink_root, args.output_dir, max_objects=args.max_objects,
                       views=views, image_size=args.image_size)


if __name__ == "__main__":
    main()


