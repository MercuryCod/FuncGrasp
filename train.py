#!/usr/bin/env python3
"""
Main training script for Functional Grasp Prediction.

This script trains a multi-modal grasp prediction model that:
1. Takes object point clouds and multi-view images as input
2. Processes text instructions using Qwen2.5-VL semantic encoder
3. Predicts finger-specific contact points (7 classes)
4. Generates grasp poses using conditional flow matching

Usage:
    # Basic training with default config
    python train.py
    
    # Custom experiment name
    EXP_NAME=exp1 python train.py
    
    # Debug mode (frequent logging and checkpointing)
    DEBUG=true python train.py
    
    # Custom Qwen tuning mode
    QWEN_TUNING=frozen python train.py    # Freeze Qwen (fastest)
    QWEN_TUNING=lora python train.py      # LoRA fine-tuning (balanced)
    QWEN_TUNING=full python train.py      # Full fine-tuning (slowest)
"""
import os
import sys
import random
import numpy as np

# Suppress tokenizers parallelism warning (we use DataLoader workers)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

# Project imports
from functools import partial

from config import Config
from dataset.dataset import OakInkDataset
from dataset.collate import collate_oakink_batch
from models.functional_grasp_model import FunctionalGraspModel
from losses import ContactLoss, FlowMatchingLoss, compute_total_loss
from utils.training_utils import (
    setup_logging, save_checkpoint, load_checkpoint,
    compute_contact_metrics, visualize_predictions
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    contact_criterion: ContactLoss,
    flow_criterion: FlowMatchingLoss,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    epoch: int,
    device: torch.device,
    global_step: int
) -> tuple:
    """
    Train for one epoch.
    
    Args:
        model: FunctionalGraspModel instance
        dataloader: Training data loader
        optimizer: Optimizer (AdamW)
        scheduler: Learning rate scheduler
        contact_criterion: ContactLoss instance
        flow_criterion: FlowMatchingLoss instance
        config: Configuration dict
        epoch: Current epoch number
        device: torch device
        global_step: Current global step counter
    
    Returns:
        (epoch_metrics, updated_global_step)
    """
    model.train()
    epoch_metrics = {
        'loss': 0,
        'contact_loss': 0,
        'flow_loss': 0,
        'contact_accuracy': 0
    }
    
    # Get logging and checkpoint intervals from config
    log_interval = config['training'].get('log_interval', 20)
    checkpoint_interval = config['training'].get('checkpoint_interval', 500)
    total_batches = len(dataloader)
    
    # Simple progress tracking without tqdm
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        pts = batch['pts'].to(device)
        mano_params = batch['mano_params'].to(device)
        contact_soft_targets = batch['contact_soft_targets'].to(device)  # [B, N, 6] soft probabilities
        contact_hard_labels = batch['contact_hard_labels'].to(device)    # [B, N] for metrics
        images_list = batch['images_list']  # List of PIL images
        texts_list = batch['texts_list']    # List of strings
        
        # Forward pass: contact prediction + semantic/geometric encoding
        outputs = model.forward_train(images_list, texts_list, pts)
        
        # Sample flow matching pairs
        # Rectified flow: learn v(x_t, t, c) where x_t = (1-t)*x_0 + t*x_1
        B = pts.shape[0]
        t = torch.rand(B, device=device)  # Random timestep in [0, 1]
        x0 = torch.randn_like(mano_params)  # Noise (source distribution)
        x1 = mano_params  # Target pose (data distribution)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1  # Linear interpolation
        target_velocity = x1 - x0  # Constant velocity along straight path
        
        # Forward pass: flow matching network
        model_velocity = model.flow_step(x_t, t, outputs['cond'])
        outputs['model_velocity'] = model_velocity
        
        # Compute losses (use soft targets for training)
        losses = compute_total_loss(
            outputs,
            {
                'contact_soft_targets': contact_soft_targets,  # Soft probability distributions
                'target_velocity': target_velocity,
                't': t
            },
            contact_criterion,
            flow_criterion,
            lambda_contact=config['training']['lambda_contact'],
            lambda_flow=config['training']['lambda_flow']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping to prevent exploding gradients
        if config['training']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        # Compute metrics (use hard labels for interpretable metrics)
        with torch.no_grad():
            metrics = compute_contact_metrics(outputs['logits_c'], contact_hard_labels)
        
        # Accumulate epoch metrics
        epoch_metrics['loss'] += losses['total'].item()
        epoch_metrics['contact_loss'] += losses['contact'].item()
        epoch_metrics['flow_loss'] += losses['flow'].item()
        epoch_metrics['contact_accuracy'] += metrics['contact_accuracy']
        
        # Log metrics at intervals
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            avg_loss = epoch_metrics['loss'] / (batch_idx + 1)
            avg_c_loss = epoch_metrics['contact_loss'] / (batch_idx + 1)
            avg_f_loss = epoch_metrics['flow_loss'] / (batch_idx + 1)
            avg_c_acc = epoch_metrics['contact_accuracy'] / (batch_idx + 1)
            
            logging.info(
                f"Epoch {epoch+1} [{batch_idx+1}/{total_batches}] "
                f"Loss: {losses['total'].item():.4f} (avg: {avg_loss:.4f}) "
                f"C_Loss: {losses['contact'].item():.4f} (avg: {avg_c_loss:.4f}) "
                f"F_Loss: {losses['flow'].item():.4f} (avg: {avg_f_loss:.4f}) "
                f"C_Acc: {metrics['contact_accuracy']:.3f} (avg: {avg_c_acc:.3f})"
            )
        
        # Validate at step intervals
        val_interval = config['eval']['val_interval']
        if global_step > 0 and global_step % val_interval == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"Validation at step {global_step}")
            logging.info(f"{'='*80}")
            val_metrics = validate(
                model, val_loader, contact_criterion,
                config, device, epoch, global_step, test_loader
            )
            logging.info(f"Val - Total: {val_metrics['total_loss']:.4f} "
                        f"(C: {val_metrics['contact_loss']:.4f}, F: {val_metrics['flow_loss']:.4f}), "
                        f"Acc: {val_metrics['contact_accuracy']:.3f}, "
                        f"F1: {val_metrics['contact_f1']:.3f}")
            logging.info(
                f"  Per-Class Acc: "
                f"no_c={val_metrics['acc_no_contact']:.3f} "
                f"palm={val_metrics['acc_palm']:.3f} "
                f"thumb={val_metrics['acc_thumb']:.3f} "
                f"index={val_metrics['acc_index']:.3f} "
                f"mid={val_metrics['acc_middle']:.3f} "
                f"ring={val_metrics['acc_ring']:.3f} "
                f"pinky={val_metrics['acc_pinky']:.3f}"
            )
            logging.info(f"{'='*80}\n")
        
        # Save debug checkpoint at intervals (if debug mode)
        if config.get('debug_mode', False) and batch_idx > 0 and (batch_idx + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], f'debug_step_{global_step}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
            logging.info(f"Debug checkpoint saved at step {global_step}")
        

        global_step += 1
    
    # Average metrics over epoch
    for k in epoch_metrics:
        epoch_metrics[k] /= len(dataloader)
    
    return epoch_metrics, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    contact_criterion: ContactLoss,
    config: dict,
    device: torch.device,
    epoch: int,
    global_step: int = 0,
    test_loader: DataLoader = None
) -> dict:
    """
    Validate model on validation set.
    
    Args:
        model: FunctionalGraspModel instance
        dataloader: Validation data loader
        contact_criterion: ContactLoss instance
        config: Configuration dict
        device: torch device
        epoch: Current epoch number (for visualization naming)
        test_loader: Optional test data loader for qualitative visualization
    
    Returns:
        Dict of validation metrics
    """
    model.eval()
    val_metrics = {
        'contact_loss': 0,
        'flow_loss': 0,  # NEW: Track flow loss during validation
        'total_loss': 0,  # NEW: Track total weighted loss
        'contact_accuracy': 0,
        'contact_precision': 0,
        'contact_recall': 0,
        'contact_f1': 0,
        # Per-class accuracies
        'acc_no_contact': 0,
        'acc_palm': 0,
        'acc_thumb': 0,
        'acc_index': 0,
        'acc_middle': 0,
        'acc_ring': 0,
        'acc_pinky': 0
    }
    
    # Get flow criterion and loss weights from config
    flow_criterion = FlowMatchingLoss()
    lambda_contact = config['training']['lambda_contact']
    lambda_flow = config['training']['lambda_flow']
    
    total_batches = len(dataloader)
    log_interval = max(1, total_batches // 10)  # Log 10 times during validation
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        pts = batch['pts'].to(device)
        mano_params = batch['mano_params'].to(device)  # NEW: Need for flow loss
        contact_soft_targets = batch['contact_soft_targets'].to(device)  # [B, N, 6] for loss
        contact_hard_labels = batch['contact_hard_labels'].to(device)    # [B, N] for metrics
        images_list = batch['images_list']
        texts_list = batch['texts_list']
        
        # Forward pass for contact prediction
        outputs = model.forward_train(images_list, texts_list, pts)
        
        # Contact loss (use soft targets)
        contact_loss = contact_criterion(outputs['logits_c'], contact_soft_targets)
        
        # Flow matching loss computation 
        B = pts.shape[0]
        t = torch.rand(B, device=device)  # Random timestep
        x0 = torch.randn_like(mano_params)  # Noise
        x1 = mano_params  # Target pose
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        target_velocity = x1 - x0
        
        # Forward pass for flow matching
        model_velocity = model.flow_step(x_t, t, outputs['cond'])
        flow_loss = flow_criterion(model_velocity, target_velocity, t)
        
        # Total loss (weighted combination)
        total_loss = lambda_contact * contact_loss + lambda_flow * flow_loss
        
        # Contact metrics (use hard labels for interpretable accuracy)
        metrics = compute_contact_metrics(outputs['logits_c'], contact_hard_labels)
        
        # Accumulate metrics
        val_metrics['contact_loss'] += contact_loss.item()
        val_metrics['flow_loss'] += flow_loss.item()
        val_metrics['total_loss'] += total_loss.item()
        for k, v in metrics.items():
            val_metrics[k] += v
        
        # Log validation progress at intervals
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            # Compute running averages for per-class metrics
            avg_metrics = {k: val_metrics[k] / (batch_idx + 1) for k in val_metrics}
            
            logging.info(
                f"Validation [{batch_idx+1}/{total_batches}] "
                f"Loss: {total_loss.item():.4f} (C: {contact_loss.item():.4f}, F: {flow_loss.item():.4f}) "
                f"C_Acc: {metrics['contact_accuracy']:.3f} (avg: {avg_metrics['contact_accuracy']:.3f})"
            )
            logging.info(
                f"  Per-Class Acc: "
                f"no_c={avg_metrics['acc_no_contact']:.3f} "
                f"palm={avg_metrics['acc_palm']:.3f} "
                f"thumb={avg_metrics['acc_thumb']:.3f} "
                f"index={avg_metrics['acc_index']:.3f} "
                f"mid={avg_metrics['acc_middle']:.3f} "
                f"ring={avg_metrics['acc_ring']:.3f} "
                f"pinky={avg_metrics['acc_pinky']:.3f}"
            )
    
    # Average metrics over all batches
    for k in val_metrics:
        val_metrics[k] /= len(dataloader)
    
    # Qualitative visualization on test set (random samples)
    if test_loader is not None:
        import random
        # Prefer validation set for qualitative unless explicitly configured to use test
        qual_source = config['eval'].get('qual_source', 'val')  # 'val' or 'test'
        if qual_source == 'test':
            logging.info("\n  Generating qualitative visualizations on test set...")
            qual_loader = test_loader
        else:
            logging.info("\n  Generating qualitative visualizations on validation set...")
            qual_loader = dataloader
        
        # Randomly select samples from chosen set
        num_qual = config['eval']['num_qualitative']
        qual_dataset = qual_loader.dataset
        total_qual = len(qual_dataset)
        
        # Random indices
        random_indices = random.sample(range(total_qual), min(num_qual, total_qual))
        
        # Manually fetch samples (to avoid batch collation issues)
        test_samples = [qual_dataset[i] for i in random_indices]
        
        # Collate into a batch
        collate_fn = qual_loader.collate_fn
        test_batch = collate_fn(test_samples)
        
        # Move to device
        pts = test_batch['pts'].to(device)
        images_list = test_batch['images_list']
        texts_list = test_batch['texts_list']
        
        # Forward pass for contact prediction
        outputs = model.forward_train(images_list, texts_list, pts)
        
        # Generate predicted grasp poses using flow matching
        logging.info("  Sampling grasp poses using flow matching...")
        predicted_poses = model.sample(
            images_list, texts_list, pts, 
            num_steps=20  # Could be configurable
        )  # [B, 61]
        outputs['predicted_poses'] = predicted_poses
        
        # Visualize (uses hard_labels from batch for GT visualization)
        # Include global_step in directory name to differentiate step-based validations
        save_dir = os.path.join(config['paths']['qual_dir'], f'step_{global_step:06d}')
        visualize_predictions(test_batch, outputs, save_dir, num_samples=num_qual)
    
    model.train()
    return val_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main training loop"""
    
    # =================================================================
    # Setup
    # =================================================================
    config = Config.get_config('train')
    Config.create_dirs()
    
    # Reproducibility
    set_seed(42)

    # Setup logging
    log_file = setup_logging(config['paths']['log_dir'], 'train')
    run_log = os.path.join(config['paths']['log_dir'], 'run.log')
    print(f"\nLogs are being saved to:")
    print(f"  - Full path: {log_file}")
    print(f"  - Quick access: {run_log}")
    print(f"\nMonitor training with: tail -f {run_log}\n")
    
    # Check debug mode
    debug_mode = config.get('debug_mode', False)
    if debug_mode:
        logging.info("="*80)
        logging.info("DEBUG MODE ENABLED")
        logging.info(f"Log interval: {config['training']['log_interval']} steps")
        logging.info(f"Checkpoint interval: {config['training']['checkpoint_interval']} steps")
        logging.info(f"Validation interval: {config['eval']['val_interval']} steps")
        max_val = config['eval'].get('max_val_samples', None)
        if max_val:
            logging.info(f"Validation samples: limited to {max_val} (for faster debugging)")
        logging.info("="*80)
    
    logging.info("="*80)
    logging.info("FUNCTIONAL GRASP TRAINING")
    logging.info("="*80)
    logging.info(f"Experiment: {Config.EXP_NAME}")
    logging.info(f"Device: {config['device']}")
    logging.info(f"Debug mode: {config['debug_mode']}")
    logging.info(f"Qwen tuning: {config['model']['qwen_tuning']}")
    
    # Device setup
    device = torch.device(config['device'])
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # =================================================================
    # Data Loading
    # =================================================================
    logging.info("\n" + "="*80)
    logging.info("LOADING DATASETS")
    logging.info("="*80)
    
    # Training dataset
    logging.info("Creating training dataset...")
    train_dataset = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='train',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir']
    )
    logging.info(f"Train samples: {len(train_dataset)}")
    
    # Validation dataset
    logging.info("Creating validation dataset...")
    val_dataset_full = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='val',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir']
    )
    
    # Limit validation samples in debug mode for faster validation
    max_val_samples = config['eval'].get('max_val_samples', None)
    if max_val_samples is not None and len(val_dataset_full) > max_val_samples:
        from torch.utils.data import Subset
        import random
        random.seed(42)  # Reproducible subset
        indices = random.sample(range(len(val_dataset_full)), max_val_samples)
        val_dataset = Subset(val_dataset_full, indices)
        logging.info(f"Val samples: {len(val_dataset)} (subset of {len(val_dataset_full)} for debug mode)")
    else:
        val_dataset = val_dataset_full
        logging.info(f"Val samples: {len(val_dataset)}")
    
    # Test dataset (for qualitative visualization)
    logging.info("Creating test dataset...")
    test_dataset = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='test',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir']
    )
    logging.info(f"Test samples: {len(test_dataset)}")
    
    # Create collate function with tau from config
    collate_fn = partial(collate_oakink_batch, soft_target_tau=config['data']['soft_target_tau'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=3,  # Exactly 3 samples for visualization
        shuffle=False,
        num_workers=0,  # No multiprocessing for test
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    logging.info(f"Train batches per epoch: {len(train_loader)}")
    logging.info(f"Val batches: {len(val_loader)}")
    logging.info(f"Test batches: {len(test_loader)}")
    
    # =================================================================
    # Model Creation
    # =================================================================
    
    model = FunctionalGraspModel(
        CSEM=config['model']['CSEM'],
        CGEO=config['model']['CGEO'],
        DPOSE=config['model']['DPOSE'],
        K_CONTACT=config['model']['K_CONTACT'],
        qwen_tuning=config['model']['qwen_tuning'],
        lora_cfg=config['lora'],
        contact_regression=config['model']['contact_regression'],
        inference_threshold=config['model']['inference_threshold']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # =================================================================
    # Optimization Setup
    # =================================================================
    logging.info("\n" + "="*80)
    logging.info("OPTIMIZATION SETUP")
    logging.info("="*80)
    
    # Loss functions
    contact_criterion = ContactLoss(no_contact_weight=0.1)
    flow_criterion = FlowMatchingLoss()
    logging.info(f"Contact loss: BCE with no_contact_weight=0.1")
    logging.info(f"Flow loss: MSE for rectified flow")
    logging.info(f"Loss weights: λ_contact={config['training']['lambda_contact']}, λ_flow={config['training']['lambda_flow']}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    logging.info(f"Optimizer: AdamW (lr={config['training']['learning_rate']}, wd={config['training']['weight_decay']})")
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    logging.info(f"Scheduler: CosineAnnealingLR (T_max={config['training']['epochs']}, eta_min=1e-6)")
    
    # =================================================================
    # Training Loop
    # =================================================================
    logging.info("\n" + "="*80)
    logging.info("STARTING TRAINING")
    logging.info("="*80)
    logging.info(f"Epochs: {config['training']['epochs']}")
    logging.info(f"Batch size: {config['training']['batch_size']}")
    logging.info(f"Gradient clip: {config['training']['gradient_clip']}")
    
    best_val_f1 = 0.0
    global_step = 0
    
    for epoch in range(config['training']['epochs']):
        logging.info(f"\n{'='*80}")
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logging.info(f"{'='*80}")
        
        # Train (includes step-based validation)
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            contact_criterion, flow_criterion,
            val_loader, test_loader,
            config, epoch, device, global_step
        )
        
        # End-of-epoch validation
        logging.info(f"\n{'='*80}")
        logging.info(f"End-of-Epoch {epoch+1} Validation")
        logging.info(f"{'='*80}")
        val_metrics = validate(
            model, val_loader, contact_criterion,
            config, device, epoch, global_step, test_loader
        )
        
        # Log metrics
        logging.info(f"\nEpoch {epoch+1} Summary:")
        logging.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Contact: {train_metrics['contact_loss']:.4f}, "
                    f"Flow: {train_metrics['flow_loss']:.4f}, "
                    f"Acc: {train_metrics['contact_accuracy']:.3f}")
        logging.info(f"  Val   - Total: {val_metrics['total_loss']:.4f} "
                    f"(C: {val_metrics['contact_loss']:.4f}, F: {val_metrics['flow_loss']:.4f}), "
                    f"Acc: {val_metrics['contact_accuracy']:.3f}, "
                    f"Prec: {val_metrics['contact_precision']:.3f}, "
                    f"Rec: {val_metrics['contact_recall']:.3f}, "
                    f"F1: {val_metrics['contact_f1']:.3f}")
        
        # Log per-class accuracies
        logging.info(f"  Val Per-Class Accuracy:")
        logging.info(f"    no_contact: {val_metrics['acc_no_contact']:.3f}, "
                    f"palm: {val_metrics['acc_palm']:.3f}, "
                    f"thumb: {val_metrics['acc_thumb']:.3f}")
        logging.info(f"    index: {val_metrics['acc_index']:.3f}, "
                    f"middle: {val_metrics['acc_middle']:.3f}, "
                    f"ring: {val_metrics['acc_ring']:.3f}, "
                    f"pinky: {val_metrics['acc_pinky']:.3f}")

        
        # Save best model based on F1 score
        if val_metrics['contact_f1'] > best_val_f1:
            best_val_f1 = val_metrics['contact_f1']
            logging.info(f"  🎉 New best F1: {best_val_f1:.4f}")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(config['paths']['checkpoint_dir'], f'epoch_{epoch+1:03d}.pth'),
                is_best=True
            )
        
        # Save periodic checkpoints
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics,
                os.path.join(config['paths']['checkpoint_dir'], f'epoch_{epoch+1:03d}.pth')
            )
        
        # Step scheduler (cosine annealing)
        scheduler.step()
    
    # =================================================================
    # Training Complete
    # =================================================================
    logging.info("\n" + "="*80)
    logging.info("TRAINING COMPLETE")
    logging.info("="*80)
    logging.info(f"Best validation F1 score: {best_val_f1:.4f}")
    logging.info(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    logging.info(f"Training logs saved to: {log_file}")
    logging.info(f"Quick access: {run_log}")
    logging.info(f"Visualizations saved to: {config['paths']['qual_dir']}")
    logging.info("\n✅ Training finished successfully!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n\n❌ Training failed with error:")
        logging.error(f"{type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)

