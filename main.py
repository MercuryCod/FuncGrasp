#!/usr/bin/env python3
"""
Example script demonstrating OakInk dataset loading with pre-rendered images.
"""
import os
import numpy as np
import torch
from dataset.dataset import OakInkDataset, get_dataloader
from utils.visualize import visualize_grasp
from PIL import Image


if __name__ == '__main__':
    # Load dataset with pre-computed contact labels AND pre-rendered images
    dataset = OakInkDataset(
        data_root='/workspace/data/OakInk',
        split='train',
        num_points=1024,
        compute_contacts=True,      # Pre-compute contact labels
        contact_threshold=0.01,     # 10mm threshold
        load_object_images=True,    # Load pre-rendered multi-view images
    )
    
    print(f"\nDataset loaded with {len(dataset)} samples")
    print(f"Contact labels are pre-computed during loading")
    
    # Get a sample - contact labels AND images are already included!
    sample_id = 20
    sample = dataset[sample_id]
    
    print(f"\nSample {sample_id}:")
    print(f"  Category: {sample['category']}")
    print(f"  Intent: {sample['intent']}")
    print(f"  Instruction: '{sample['instruction']}'")
    
    # Contact info is directly available from dataset (pre-computed)
    contact_labels = sample['contact_labels']      # (N,) tensor with values 0-6
    contact_mask = sample['contact_mask']          # (N,) boolean tensor
    contact_distances = sample['contact_distances'] # (N,) float tensor
    
    print(f"\nContact info (pre-computed in dataset):")
    print(f"  Labels shape: {contact_labels.shape}, range: {contact_labels.min()}-{contact_labels.max()}")
    print(f"  Contact rate: {contact_mask.float().mean():.1%}")
    print(f"  Min distance: {contact_distances.min():.4f}m")
    
    # Object images are also available (pre-rendered)
    if 'object_images' in sample:
        object_images = sample['object_images']  # Dict of 6 views
        print(f"\nObject images (pre-rendered):")
        print(f"  Available views: {list(object_images.keys())}")
        print(f"  Image shape: {object_images['front'].shape}")
        print(f"  Value range: [{object_images['front'].min():.3f}, {object_images['front'].max():.3f}]")
    else:
        print(f"\nWARNING: No object images found in sample")
    
    obj_points = sample['obj_points']

    # Create contact_info dictionary for visualization
    contact_info = {
        'finger_labels': contact_labels,
        'contact_mask': contact_mask,
        'contact_distances': contact_distances,
    }

    # Visualize hand-object interaction
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    contact_output_path = os.path.join(output_folder, 'contact_with_hand.html')
    
    print(f"\nGenerating 3D visualization...")
    visualize_grasp(
        sample['mano_pose'], 
        sample['mano_shape'], 
        sample['mano_trans'], 
        obj_points, 
        output_path=contact_output_path,
        contact_info=contact_info,
        show_joints=False
    )
    print(f"Saved to: {contact_output_path}")
    
    # Save pre-rendered multi-view images to output folder
    if 'object_images' in sample:
        print(f"\n" + "=" * 80)
        print("Saving pre-rendered multi-view images...")
        print("=" * 80)
        
        images_folder = os.path.join(output_folder, 'multiview')
        os.makedirs(images_folder, exist_ok=True)
        
        print(f"\nSaving 6 viewpoints:")
        for view_name, img_tensor in object_images.items():
            # Convert from torch tensor [0, 1] to uint8 [0, 255]
            img_array = (img_tensor.numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            
            output_path = os.path.join(images_folder, f'sample_24889_{view_name}.png')
            img_pil.save(output_path)
            print(f"  {view_name}: {output_path}")
        
        print(f"\n" + "=" * 80)
        print("All visualizations complete!")
        print(f"  3D visualization (hand + object): {contact_output_path}")
        print(f"  2D images (object only): {images_folder}/")
        print(f"\nNote: Images were loaded from pre-rendered cache, not rendered on-the-fly")
        print("=" * 80)
    else:
        print(f"\n" + "=" * 80)
        print("Could not save images - no pre-rendered images available")
        print("Please run: bash scripts/prepare_renders.sh")
        print("=" * 80)
