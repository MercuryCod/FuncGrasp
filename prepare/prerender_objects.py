#!/usr/bin/env python3
"""
Pre-render all unique objects in OakInk dataset from multiple viewpoints.
This script generates and saves multi-view images for faster training.
"""
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.render_utils import render_mesh_multiview, VIEWS


def collect_unique_objects(oakink_root: Path) -> Set[str]:
    """
    Collect all unique object IDs from the OakInk dataset using the dataset API.
    
    Args:
        oakink_root: Root directory of OakInk dataset
    
    Returns:
        Set of unique object IDs
    """
    print("Discovering unique objects in dataset...")
    
    # Import dataset here to avoid circular import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.dataset import OakInkDataset
    
    # Load datasets from all splits to get complete object set
    unique_objects = set()
    
    for split in ['train', 'val', 'test']:
        print(f"  Scanning {split} split...")
        dataset = OakInkDataset(
            data_root=str(oakink_root),
            split=split,
            num_points=256,  # Small number for faster loading
            compute_contacts=False,  # Don't compute contacts
            load_object_images=False,  # Don't load images
        )
        
        # Extract unique shape_ids from this split
        split_objects = set(sample['shape_id'] for sample in dataset.samples)
        print(f"    {split}: {len(split_objects)} unique objects")
        unique_objects.update(split_objects)
    
    print(f"\nTotal: {len(unique_objects)} unique objects across all splits")
    return unique_objects


def find_object_mesh_path(oakink_root: Path, object_id: str) -> Optional[Path]:
    """
    Find mesh file path for a given object ID.
    
    Args:
        oakink_root: Root directory of OakInk dataset
        object_id: Object identifier
    
    Returns:
        Path to mesh file, or None if not found
    """
    # Try both real and virtual object directories
    shape_roots = [
        oakink_root / "shape" / "OakInkObjectsV2",
        oakink_root / "shape" / "OakInkVirtualObjectsV2",
    ]
    
    for shape_root in shape_roots:
        if not shape_root.exists():
            continue
        
        obj_dir = shape_root / object_id
        if not obj_dir.exists():
            continue
        
        # Try align_ds first, then align
        for mesh_suffix in ["align_ds", "align"]:
            mesh_dir = obj_dir / mesh_suffix
            if mesh_dir.exists():
                mesh_files = list(mesh_dir.glob("*.obj"))
                if mesh_files:
                    # Prefer files with 'align' or 'textured' in name
                    for mesh_file in mesh_files:
                        if "align" in mesh_file.name or "textured" in mesh_file.name:
                            return mesh_file
                    # Otherwise return first mesh
                    return mesh_files[0]
        
        # Try root directory
        mesh_files = list(obj_dir.glob("*.obj"))
        if mesh_files:
            return mesh_files[0]
    
    return None


def render_all_objects(oakink_root: str, output_dir: str,
                       image_size: int = 512,
                       zoom: float = 1.6,
                       views: Optional[List[str]] = None,
                       force: bool = False) -> Dict[str, dict]:
    """
    Render all unique objects in the dataset.
    
    Args:
        oakink_root: Root directory of OakInk dataset
        output_dir: Output directory for rendered images
        image_size: Output image size in pixels
        zoom: Zoom factor for rendering
        views: List of view names (default: all)
        force: If True, re-render existing objects
    
    Returns:
        Dictionary with rendering statistics
    """
    from dataset.dataset import OakInkDataset
    
    oakink_root = Path(oakink_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect unique objects from all splits
    unique_objects = collect_unique_objects(oakink_root)
    
    # Create a dataset instance to access mesh path lookup (uses metadata mapping)
    print("\nInitializing dataset for mesh path lookups...")
    dataset = OakInkDataset(
        data_root=str(oakink_root),
        split='train',  # Any split works, we just need the _get_object_mesh_path method
        num_points=256,
        compute_contacts=False,
        load_object_images=False,
    )
    
    if views is None:
        views = list(VIEWS.keys())
    
    print(f"\nRendering {len(unique_objects)} objects with views: {', '.join(views)}")
    print(f"Image size: {image_size}x{image_size}, Zoom: {zoom}")
    print(f"Output directory: {output_dir}\n")
    
    stats = {
        'total': len(unique_objects),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'failed_objects': [],
        'render_params': {
            'image_size': image_size,
            'zoom': zoom,
            'views': views,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    for object_id in tqdm(unique_objects, desc="Rendering objects"):
        try:
            # Check if already rendered (unless force=True)
            obj_out_dir = output_dir / object_id
            if not force and obj_out_dir.exists():
                # Check if all views exist
                all_views_exist = all((obj_out_dir / f"{v}.png").exists() for v in views)
                if all_views_exist:
                    stats['skipped'] += 1
                    continue
            
            # Find mesh file using dataset's metadata mapping
            mesh_path = dataset._get_object_mesh_path(object_id)
            if mesh_path is None:
                tqdm.write(f"WARNING: Mesh not found for object {object_id}")
                stats['failed'] += 1
                stats['failed_objects'].append({
                    'object_id': object_id,
                    'reason': 'mesh_not_found'
                })
                continue
            
            # Render views
            render_mesh_multiview(
                mesh_path=str(mesh_path),
                output_folder=str(output_dir),
                object_id=object_id,
                image_size=image_size,
                zoom=zoom,
                views=views,
                verbose=False
            )
            
            stats['success'] += 1
            
        except Exception as e:
            tqdm.write(f"ERROR: Failed to render {object_id}: {e}")
            stats['failed'] += 1
            stats['failed_objects'].append({
                'object_id': object_id,
                'reason': str(e)
            })
    
    # Save metadata
    metadata_file = output_dir / "render_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Rendering complete!")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Metadata: {metadata_file}")
    
    if stats['failed'] > 0:
        failed_file = output_dir / "failed_objects.json"
        with open(failed_file, 'w') as f:
            json.dump(stats['failed_objects'], f, indent=2)
        print(f"  Failed objects list: {failed_file}")
    
    print(f"{'='*80}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-render all objects in OakInk dataset from multiple viewpoints"
    )
    parser.add_argument(
        "--oakink_root",
        type=str,
        required=True,
        help="Path to OakInk dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for rendered images (default: {oakink_root}/rendered_objects)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Output image size in pixels (default: 512)"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.6,
        help="Zoom factor - higher values fill more of frame (default: 1.6)"
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Comma-separated list of views to render (default: all 6 views)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render objects even if they already exist"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.oakink_root) / "rendered_objects")
    
    # Parse views
    views = None
    if args.views:
        views = [v.strip() for v in args.views.split(',')]
    
    # Run rendering
    render_all_objects(
        oakink_root=args.oakink_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        zoom=args.zoom,
        views=views,
        force=args.force
    )


if __name__ == "__main__":
    main()

