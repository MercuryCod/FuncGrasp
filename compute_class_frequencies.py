#!/usr/bin/env python3
"""
Compute class frequencies from training data for pos_weight calculation.

Usage:
    python compute_class_frequencies.py --data_path /mnt/data/OakInk --render_dir /mnt/data/OakInk/rendered_objects
    
Or via setup script:
    bash scripts/setup_training.sh
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.oakink_loader import OakInkDataset
from config import Config


def compute_class_frequencies(data_path, render_dir=None, split='train'):
    """
    Compute class frequencies from the training/test dataset.
    
    Args:
        data_path: Path to OakInk dataset
        render_dir: Path to rendered objects (if None, use data_path/rendered_objects)
        split: 'train' or 'test'
    
    Returns:
        class_counts: dict mapping class_id -> count
        pos_weight: list of pos_weight values for each class
    """
    print(f"Computing class frequencies for {split} split...")
    print(f"Data path: {data_path}")
    
    # Get config
    cfg = Config.get_config()
    
    # Determine render_dir
    if render_dir is None:
        render_dir = os.path.join(data_path, 'rendered_objects')
    
    print(f"Render dir: {render_dir}")
    
    # Create dataset (just training split for frequency computation)
    dataset = OakInkDataset(
        root_dir=data_path,
        render_dir=render_dir,
        split=split,
        split_mode=cfg['data']['split_mode'],
        n_points=cfg['model']['n_points'],
        contact_threshold=cfg['data']['contact_threshold'],
        use_cache=cfg['data']['use_cache'],
        num_views=1,  # Use single view for speed
        use_soft_targets=False,  # Only need hard labels for frequency
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize counters
    class_counts = np.zeros(Config.NUM_CONTACT_CLASSES, dtype=np.int64)
    
    # Iterate through dataset
    for idx in tqdm(range(len(dataset)), desc="Computing frequencies"):
        try:
            batch = dataset[idx]
            labels = batch['contact_labels'].numpy().flatten()  # [N]
            
            # Count each class
            for class_id in range(Config.NUM_CONTACT_CLASSES):
                class_counts[class_id] += np.sum(labels == class_id)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute frequencies
    total_points = class_counts.sum()
    class_frequencies = class_counts / total_points
    
    # Compute pos_weight = 1 / sqrt(freq)
    # Add small epsilon to avoid division by zero
    pos_weight = 1.0 / np.sqrt(class_frequencies + 1e-8)
    
    # Normalize pos_weight so average is 1.0
    pos_weight = pos_weight / pos_weight.mean()
    
    # Print results
    print("\n" + "="*80)
    print("Class Frequency Analysis")
    print("="*80)
    print(f"{'Class':<15} {'Count':<15} {'Frequency':<15} {'pos_weight':<15}")
    print("-"*80)
    
    for i, class_name in enumerate(Config.CONTACT_CLASSES):
        print(f"{class_name:<15} {class_counts[i]:<15} {class_frequencies[i]:<15.6f} {pos_weight[i]:<15.4f}")
    
    print("-"*80)
    print(f"{'Total':<15} {total_points:<15}")
    print("="*80)
    
    # Prepare output
    result = {
        'class_counts': class_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'pos_weight': pos_weight.tolist(),
        'total_points': int(total_points),
        'num_samples': len(dataset),
        'contact_threshold_mm': cfg['data']['contact_threshold'] * 1000,
        'split': split
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute class frequencies for pos_weight")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to OakInk dataset')
    parser.add_argument('--render_dir', type=str, default=None,
                        help='Path to rendered objects (default: data_path/rendered_objects)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split to compute frequencies for')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_path/cache)')
    
    args = parser.parse_args()
    
    # Compute frequencies
    result = compute_class_frequencies(args.data_path, args.render_dir, args.split)
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.data_path) / 'cache'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'class_frequencies_{args.split}.json'
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Saved class frequencies to: {output_file}")
    print(f"\nThe pos_weight values will be automatically loaded during training.")
    print(f"   pos_weight: {result['pos_weight']}")


if __name__ == '__main__':
    main()

