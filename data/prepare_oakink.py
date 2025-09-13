#!/usr/bin/env python3
"""
Integration script for OakInk dataset with functional grasp training pipeline.
Validates data loading and prepares batches for training.
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from data.oakink_loader import create_oakink_loaders, OakInkPartDataset
from models.functional_grasp_model import FunctionalGraspModel


def validate_batch(batch):
    """Validate that batch has correct format for training."""
    required_keys = ["images", "texts", "points", "contact_labels", "pose"]
    
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Missing required key: {key}")
    
    # Check shapes
    B = batch["points"].size(0)
    N = batch["points"].size(1)
    
    assert batch["points"].shape == (B, N, 3), f"Invalid points shape: {batch['points'].shape}"
    assert batch["contact_labels"].shape == (B, N), f"Invalid contact_labels shape: {batch['contact_labels'].shape}"
    assert batch["pose"].shape[0] == B, f"Invalid pose batch size: {batch['pose'].shape[0]}"
    
    # Check types
    assert isinstance(batch["texts"], list), "texts should be a list of strings"
    assert len(batch["texts"]) == B, f"texts length {len(batch['texts'])} != batch size {B}"
    
    print(f"✓ Batch validation passed: B={B}, N={N}")
    return True


def test_data_loading(root_dir, num_samples=5):
    """Test loading individual samples from the dataset."""
    print("\n=== Testing Data Loading ===")
    
    dataset = OakInkPartDataset(
        root_dir=root_dir,
        split="train",
        split_mode="split0",
        n_points=1024,
        use_cache=False,
        single_view=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\nLoading sample {i}...")
        try:
            sample = dataset[i]
            
            print(f"  Sequence: {sample['meta']['seq_id']}")
            print(f"  Object: {sample['meta']['obj_id']}")
            print(f"  Text: {sample['texts']}")
            print(f"  Points shape: {sample['points'].shape}")
            print(f"  Contact ratio: {sample['contact_labels'].mean():.2%}")
            print(f"  Pose shape: {sample['pose'].shape}")
            
            # Check if image loaded
            if sample["images"] is not None:
                if hasattr(sample["images"], 'size'):
                    print(f"  Image size: {sample['images'].size}")
                else:
                    print(f"  Images: {len(sample['images'])} views")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    return dataset


def test_dataloader(root_dir, batch_size=4):
    """Test the dataloader with batching."""
    print("\n=== Testing DataLoader ===")
    
    train_loader, test_loader = create_oakink_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for debugging
        split_mode="split0",
        n_points=1024,
        use_cache=True,
        single_view=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test first batch
    print("\nLoading first training batch...")
    for batch in train_loader:
        validate_batch(batch)
        
        print(f"  Batch size: {batch['points'].size(0)}")
        print(f"  Points: {batch['points'].shape}")
        print(f"  Contact labels: {batch['contact_labels'].shape}")
        print(f"  Poses: {batch['pose'].shape}")
        print(f"  Sample texts: {batch['texts'][:2]}")
        
        break  # Only test first batch
    
    return train_loader, test_loader


def test_model_forward(root_dir, device="cpu"):
    """Test forward pass through the model with real data."""
    print("\n=== Testing Model Forward Pass ===")
    
    # Create minimal model
    model = FunctionalGraspModel(
        qwen_name="Qwen/Qwen2.5-VL-3B-Instruct",
        CSEM=256,
        CGEO=256,
        DPOSE=28,
        K_CONTACT=1
    ).to(device)
    
    # Get a batch
    train_loader, _ = create_oakink_loaders(
        root_dir=root_dir,
        batch_size=2,
        num_workers=0,
        n_points=1024,
        use_cache=True
    )
    
    for batch in train_loader:
        # Move to device
        points = batch["points"].to(device)
        poses = batch["pose"].to(device)
        contact_labels = batch["contact_labels"].to(device)
        
        print(f"Input shapes:")
        print(f"  Points: {points.shape}")
        print(f"  Poses: {poses.shape}")
        print(f"  Contact labels: {contact_labels.shape}")
        
        # Forward pass (without images for speed)
        try:
            # Use dummy images for testing
            from PIL import Image
            dummy_images = [Image.new('RGB', (224, 224), color='white')] * points.size(0)
            
            with torch.no_grad():
                f_fuse, logits_c, c = model.forward_backbone(
                    images=dummy_images,
                    texts=batch["texts"],
                    pts=points
                )
            
            print(f"\nOutput shapes:")
            print(f"  F_fuse: {f_fuse.shape}")
            print(f"  Logits_c: {logits_c.shape}")
            print(f"  Conditioning: {c.shape}")
            print(f"✓ Model forward pass successful!")
            
        except Exception as e:
            print(f"✗ Model forward failed: {str(e)}")
        
        break  # Only test one batch
    
    return model


def compute_dataset_statistics(root_dir, split="train", max_samples=100):
    """Compute statistics over the dataset."""
    print("\n=== Computing Dataset Statistics ===")
    
    dataset = OakInkPartDataset(
        root_dir=root_dir,
        split=split,
        n_points=1024,
        use_cache=True
    )
    
    contact_ratios = []
    pose_stats = []
    
    print(f"Analyzing {min(max_samples, len(dataset))} samples...")
    
    for i in tqdm(range(min(max_samples, len(dataset)))):
        try:
            sample = dataset[i]
            contact_ratios.append(sample["contact_labels"].mean().item())
            pose_stats.append(sample["pose"].numpy())
        except:
            continue
    
    if contact_ratios:
        print(f"\nContact statistics:")
        print(f"  Mean contact ratio: {np.mean(contact_ratios):.2%}")
        print(f"  Std contact ratio: {np.std(contact_ratios):.2%}")
        print(f"  Min contact ratio: {np.min(contact_ratios):.2%}")
        print(f"  Max contact ratio: {np.max(contact_ratios):.2%}")
    
    if pose_stats:
        pose_stats = np.array(pose_stats)
        print(f"\nPose statistics:")
        print(f"  Mean pose: {pose_stats.mean(axis=0)[:3]}")  # First 3 values
        print(f"  Std pose: {pose_stats.std(axis=0)[:3]}")
    
    return contact_ratios, pose_stats


def generate_semantic_prompts(root_dir):
    """Generate semantic prompts from part annotations."""
    print("\n=== Generating Semantic Prompts ===")
    
    oakbase_dir = Path(root_dir) / "OakBase"
    prompts = {}
    
    if oakbase_dir.exists():
        for cat_dir in oakbase_dir.iterdir():
            if cat_dir.is_dir():
                category = cat_dir.name
                print(f"\nCategory: {category}")
                
                for obj_dir in list(cat_dir.iterdir())[:3]:  # Sample 3 objects
                    if obj_dir.is_dir():
                        obj_id = obj_dir.name
                        
                        # Load part annotations
                        parts = []
                        for part_file in obj_dir.glob("part_*.json"):
                            import json
                            with open(part_file, 'r') as f:
                                part = json.load(f)
                                parts.append(part)
                        
                        if parts:
                            # Generate prompt from parts
                            actions = []
                            for part in parts:
                                name = part.get("name", "part")
                                attrs = part.get("attr", [])
                                
                                if "held_by_hand" in attrs:
                                    actions.append(f"grasp the {name}")
                                if "flow_out_sth" in attrs:
                                    actions.append(f"pour using the {name}")
                                if "contain_sth" in attrs:
                                    actions.append(f"hold contents in the {name}")
                            
                            if actions:
                                prompt = " to ".join(actions[:2])
                                prompts[obj_id] = prompt
                                print(f"  {obj_id}: {prompt}")
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Prepare OakInk dataset for training")
    parser.add_argument("--root_dir", type=str, default="/Users/mercury/Developer/FuncGrasp/OakInk",
                        help="Path to OakInk dataset root")
    parser.add_argument("--test_loading", action="store_true", help="Test data loading")
    parser.add_argument("--test_dataloader", action="store_true", help="Test dataloader")
    parser.add_argument("--test_model", action="store_true", help="Test model forward pass")
    parser.add_argument("--compute_stats", action="store_true", help="Compute dataset statistics")
    parser.add_argument("--generate_prompts", action="store_true", help="Generate semantic prompts")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not Path(args.root_dir).exists():
        print(f"Error: Dataset root not found: {args.root_dir}")
        return
    
    print(f"OakInk Dataset Root: {args.root_dir}")
    
    if args.all or args.test_loading:
        test_data_loading(args.root_dir)
    
    if args.all or args.test_dataloader:
        test_dataloader(args.root_dir)
    
    if args.all or args.test_model:
        # Note: This requires the model files to exist
        try:
            test_model_forward(args.root_dir)
        except ImportError as e:
            print(f"Skipping model test: {e}")
    
    if args.all or args.compute_stats:
        compute_dataset_statistics(args.root_dir)
    
    if args.all or args.generate_prompts:
        generate_semantic_prompts(args.root_dir)
    
    print("\n=== Integration Complete ===")


if __name__ == "__main__":
    main()
