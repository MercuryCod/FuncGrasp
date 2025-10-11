from dataset.oakink_loader import OakInkDataset
from utils.visualization_3d import create_hand_object_figure
from utils.mesh_export import export_contact_info
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def save_images(images_list, frame_idx, output_dir="output"):
    """Save all views from images_list."""
    os.makedirs(output_dir, exist_ok=True)
    
    if images_list and len(images_list) > 0:
        views = images_list[0]  # Get first batch
        view_names = ["front", "left", "right", "back", "top", "bottom"]
        
        for i, img in enumerate(views):
            view_name = view_names[i] if i < len(view_names) else f"view_{i}"
            img_path = os.path.join(output_dir, f"frame_{frame_idx}_{view_name}.png")
            img.save(img_path)
            print(f"  Saved: {img_path}")


def visualize_hand_object_3d(sample, dataset, sample_idx, frame_idx, output_dir="output"):
    """Create 3D visualization of hand-object interaction with full hand mesh."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    points = sample.get("points", None)
    labels = sample.get("contact_labels", None)
    pose = sample.get("pose", None)
    meta = sample.get("meta", {})
    
    if points is None or labels is None or pose is None:
        print("  Missing data for 3D visualization")
        return
    
    # Convert to numpy
    points_np = points[0].cpu().numpy() if hasattr(points, 'cpu') else points[0]
    labels_np = labels[0].cpu().numpy() if hasattr(labels, 'cpu') else labels[0]
    pose_np = pose[0].cpu().numpy() if hasattr(pose, 'cpu') else pose[0]
    
    # Reshape pose to (21, 3) for hand joints
    hand_joints = pose_np.reshape(21, 3)
    
    # Load hand vertices directly from the dataset
    seq_info = dataset.sequences[sample_idx]
    seq_id, timestamp, frame_idx_orig, view_idx = seq_info
    frame_id = f"{seq_id.replace('/', '__')}__{timestamp}__{frame_idx_orig}__{view_idx}"
    
    try:
        # Load hand vertices in world coordinates
        hand_vertices_world = dataset._load_hand_vertices(frame_id)
        obj_transform = dataset._load_obj_transform(frame_id)
        
        # Transform to object frame
        obj_transform_inv = np.linalg.inv(obj_transform)
        hand_vertices_h = np.hstack([hand_vertices_world, np.ones((hand_vertices_world.shape[0], 1))])
        hand_vertices = (obj_transform_inv @ hand_vertices_h.T).T[:, :3]
    except Exception as e:
        print(f"  Warning: Could not load hand vertices: {e}")
        hand_vertices = None
    
    # Create visualization using utility function
    fig = create_hand_object_figure(
        object_points=points_np,
        contact_labels=labels_np,
        hand_vertices=hand_vertices,
        hand_joints=hand_joints,
        show_joints=True
    )
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"frame_{frame_idx}_3d_visualization.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 3D visualization: {plot_path}")
    
    # Export contact info with data values
    info_path = os.path.join(output_dir, f"frame_{frame_idx}_contact_info.txt")
    additional_info = {
        'seq_id': meta.get('seq_id', 'unknown'),
        'frame_idx': meta.get('frame_idx', -1),
        'obj_id': meta.get('obj_id', 'unknown'),
        'object_points_shape': points_np.shape,
        'hand_joints_shape': hand_joints.shape,
        'hand_vertices_shape': hand_vertices.shape if hand_vertices is not None else None,
        "object_points": points_np.tolist(),
        "hand_vertices": hand_vertices.tolist(),
        "hand_joints": hand_joints.tolist(),
    }
    
    if hand_vertices is not None:
        additional_info['hand_bounds'] = {
            'min': hand_vertices.min(axis=0).tolist(),
            'max': hand_vertices.max(axis=0).tolist(),
            'center': hand_vertices.mean(axis=0).tolist()
        }
    
    export_contact_info(labels_np, info_path, additional_info)
    print(f"  Saved contact info: {info_path}")


def summarize_sample(sample, dataset=None, sample_idx=0, frame_idx=0, save_visuals=False):
    meta = sample.get("meta", {})
    pose = sample.get("pose", None)
    points = sample.get("points", None)
    labels = sample.get("contact_labels", None)
    images_list = sample.get("images_list", None)
    texts_list = sample.get("texts_list", None)

    print("=== Sample Summary ===")
    if meta:
        print("meta:")
        for k in ["seq_id", "frame_idx", "obj_id"]:
            if k in meta:
                print(f"  {k}: {meta[k]}")

    if images_list is not None:
        # images_list is List[List[PIL.Image]] with B=1
        num_views = len(images_list[0]) if len(images_list) > 0 else 0
        print(f"images_list: B={len(images_list)}, views={num_views}")
        
        if save_visuals:
            print("Saving images...")
            save_images(images_list, frame_idx)

    if texts_list is not None:
        print(f"texts_list: B={len(texts_list)}")

    if points is not None:
        print(f"points shape: {tuple(points.shape)}  dtype: {points.dtype}")

    if pose is not None:
        print(f"pose shape: {tuple(pose.shape)}  dtype: {pose.dtype}")

    if labels is not None:
        arr = labels[0].cpu().numpy() if hasattr(labels, 'cpu') else labels[0]
        uniq, cnt = np.unique(arr, return_counts=True)
        print(f"contact_labels shape: {tuple(labels.shape)}  dtype: {labels.dtype}")
        print("contact label distribution:")
        for u, c in zip(uniq, cnt):
            print(f"  class {int(u)}: {int(c)}")

    if save_visuals and points is not None and labels is not None and dataset is not None:
        print("Creating 3D visualization...")
        visualize_hand_object_3d(sample, dataset, sample_idx, frame_idx)
        


def main():
    print("Starting dataset initialization...")
    import sys
    sys.stdout.flush()
    
    dataset = OakInkDataset(
        root_dir="/mnt/data/changma/OakInk",
        render_dir="/mnt/data/changma/OakInk/rendered_objects",
        split="train",
        split_mode="split0",
        n_points=1024,
        contact_threshold=0.005,  # 1cm threshold
        num_views=6,
        use_cache=True,
        transform_to_object_frame=True,
    )
    
    print("Dataset initialized successfully!")
    sys.stdout.flush()

    # Test frame 48 which has full grasp with multi-finger contact
    print("Testing frame 48 (full grasp with multi-finger contact):\n")
    sys.stdout.flush()
    
    sample_idx = 47
    print(f"Loading sample {sample_idx}...")
    sys.stdout.flush()
    
    sample = dataset[sample_idx]
    print("Sample loaded, summarizing...")
    sys.stdout.flush()
    
    summarize_sample(sample, dataset=dataset, sample_idx=sample_idx, frame_idx=48, save_visuals=True)


if __name__ == "__main__":
    
    print("Showing sample...")
    main()
