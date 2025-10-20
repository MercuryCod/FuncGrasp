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
    QWEN_TUNING=lora python train.py      # LoRA fine-tuning (default, efficient)
    QWEN_TUNING=full python train.py      # Full fine-tuning (more parameters)
"""
import os
import sys
import random
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset.dataset import OakInkDataset
from dataset.collate import collate_oakink_batch
from models.functional_grasp_model import FunctionalGraspModel
from losses import ContactLoss, FlowMatchingLoss, compute_total_loss
from utils.training_utils import (
    setup_logging, save_checkpoint,
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
    global_step: int,
    best_val_total: float
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
        (epoch_metrics, updated_global_step, updated_best_val_total)
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
            metrics = compute_contact_metrics(
                outputs['logits_c'], 
                contact_hard_labels,
                tau=config['data']['soft_target_tau'],
                contact_threshold=config['data']['contact_threshold']
            )

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
                "Epoch %d [%d/%d] Loss: %.4f (avg: %.4f) "
                "C_Loss: %.4f (avg: %.4f) F_Loss: %.4f (avg: %.4f) "
                "C_Acc: %.3f (avg: %.3f)",
                epoch+1, batch_idx+1, total_batches,
                losses['total'].item(), avg_loss,
                losses['contact'].item(), avg_c_loss,
                losses['flow'].item(), avg_f_loss,
                metrics['contact_accuracy'], avg_c_acc
            )

        # Validate at step intervals
        val_interval = config['eval']['val_interval']
        if global_step > 0 and global_step % val_interval == 0:
            logging.info("\n%s", "="*80)
            logging.info("Validation at step %d", global_step)
            logging.info("%s", "="*80)
            val_metrics = validate(
                model, val_loader, contact_criterion,
                config, device, epoch, global_step, test_loader
            )
            logging.info(
                "Val - Total: %.4f (C: %.4f, F: %.4f), Acc: %.3f, F1: %.3f",
                val_metrics['total_loss'], val_metrics['contact_loss'],
                val_metrics['flow_loss'], val_metrics['contact_accuracy'],
                val_metrics['contact_f1']
            )
            logging.info(
                "  Per-Class Acc: no_c=%.3f palm=%.3f thumb=%.3f "
                "index=%.3f mid=%.3f ring=%.3f pinky=%.3f",
                val_metrics['acc_no_contact'], val_metrics['acc_palm'],
                val_metrics['acc_thumb'], val_metrics['acc_index'],
                val_metrics['acc_middle'], val_metrics['acc_ring'],
                val_metrics['acc_pinky']
            )

            # Save best model based on total validation loss
            if val_metrics['total_loss'] < best_val_total:
                best_val_total = val_metrics['total_loss']
                logging.info("  🎉 New best Total Loss: %.4f", best_val_total)
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    os.path.join(config['paths']['checkpoint_dir'], 'best.pth'),
                    is_best=True
                )

            logging.info("%s\n", "="*80)

        # Save debug checkpoint at intervals (if debug mode)
        if config.get('debug_mode', False) and batch_idx > 0 and (batch_idx + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], f'debug_step_{global_step}.pth')
            debug_metrics = {
                'global_step': global_step,
                'batch_idx': batch_idx,
                'epoch': epoch
            }
            save_checkpoint(model, optimizer, scheduler, epoch, debug_metrics, checkpoint_path)
            logging.info("Debug checkpoint saved at step %d", global_step)


        global_step += 1

    # Average metrics over epoch
    for k in epoch_metrics:
        epoch_metrics[k] /= len(dataloader)

    return epoch_metrics, global_step, best_val_total


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
        metrics = compute_contact_metrics(
            outputs['logits_c'], 
            contact_hard_labels,
            tau=config['data']['soft_target_tau'],
            contact_threshold=config['data']['contact_threshold']
        )

        # Accumulate metrics
        val_metrics['contact_loss'] += contact_loss.item()
        val_metrics['flow_loss'] += flow_loss.item()
        val_metrics['total_loss'] += total_loss.item()
        for k, v in metrics.items():
            val_metrics[k] += v

    # Average metrics over all batches
    for k in val_metrics:
        val_metrics[k] /= len(dataloader)

    # Qualitative visualization on test set (random samples)
    if test_loader is not None:
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
            num_steps=config['flow']['num_steps_inference']
        )  # [B, 61]
        outputs['predicted_poses'] = predicted_poses

        # Visualize (uses hard_labels from batch for GT visualization)
        # Include global_step in directory name to differentiate step-based validations
        save_dir = os.path.join(config['paths']['qual_dir'], f'step_{global_step:06d}')
        visualize_predictions(
            test_batch,
            outputs,
            save_dir,
            num_samples=num_qual,
            tau=config['data']['soft_target_tau'],
            contact_threshold=config['data']['contact_threshold']
        )

    model.train()
    return val_metrics


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
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

    # Suppress tokenizers parallelism warning (we use DataLoader workers)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Reproducibility
    set_seed(42)

    # Setup logging
    log_file = setup_logging(config['paths']['log_dir'], 'train')
    run_log = os.path.join(config['paths']['log_dir'], 'run.log')
    print(f"\nLogs are being saved to:")
    print(f"  - Full path: {log_file}")
    print(f"  - Quick access: {run_log}")
    print(f"\nMonitor training with: tail -f {run_log}\n")  # noqa: W1309

    # Check debug mode
    debug_mode = config.get('debug_mode', False)
    if debug_mode:
        logging.info("%s", "="*80)
        logging.info("DEBUG MODE ENABLED")
        logging.info("Log interval: %d steps", config['training']['log_interval'])
        logging.info("Checkpoint interval: %d steps", config['training']['checkpoint_interval'])
        logging.info("Validation interval: %d steps", config['eval']['val_interval'])
        max_val = config['eval'].get('max_val_samples', None)
        if max_val:
            logging.info("Validation samples: limited to %d (for faster debugging)", max_val)
        logging.info("%s", "="*80)

    logging.info("%s", "="*80)
    logging.info("FUNCTIONAL GRASP TRAINING")
    logging.info("%s", "="*80)
    logging.info("Experiment: %s", Config.EXP_NAME)
    logging.info("Device: %s", config['device'])
    logging.info("Debug mode: %s", config['debug_mode'])
    logging.info("Qwen tuning: %s", config['model']['qwen_tuning'])

    # Device setup
    device = torch.device(config['device'])
    logging.info("Using device: %s", device)
    if device.type == 'cuda':
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        logging.info("GPU memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # =================================================================
    # Data Loading
    # =================================================================
    logging.info("\n%s", "="*80)
    logging.info("LOADING DATASETS")
    logging.info("%s", "="*80)

    # Training dataset
    logging.info("Creating training dataset...")
    train_dataset = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='train',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir'],
        filter_zero_contact=config['data'].get('filter_zero_contact', False)
    )
    logging.info("Train samples: %d", len(train_dataset))

    # Validation dataset
    logging.info("Creating validation dataset...")
    val_dataset_full = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='val',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir'],
        # Evaluation should include zero-contact approach poses for fair metrics
        filter_zero_contact=False
    )

    # Limit validation samples in debug mode for faster validation
    max_val_samples = config['eval'].get('max_val_samples', None)
    if max_val_samples is not None and len(val_dataset_full) > max_val_samples:
        from torch.utils.data import Subset
        random.seed(42)  # Reproducible subset
        indices = random.sample(range(len(val_dataset_full)), max_val_samples)
        val_dataset = Subset(val_dataset_full, indices)
        logging.info("Val samples: %d (subset of %d for debug mode)",
                     len(val_dataset), len(val_dataset_full))
    else:
        val_dataset = val_dataset_full
        logging.info("Val samples: %d", len(val_dataset))

    # Test dataset (for qualitative visualization)
    logging.info("Creating test dataset...")
    test_dataset = OakInkDataset(
        data_root=config['data']['root_dir'],
        split='test',
        num_points=config['model']['n_points'],
        compute_contacts=True,
        contact_threshold=config['data']['contact_threshold'],
        load_object_images=True,
        rendered_dir=config['data']['render_dir'],
        filter_zero_contact=False  # Keep all test samples for comprehensive evaluation
    )
    logging.info("Test samples: %d", len(test_dataset))

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

    logging.info("Train batches per epoch: %d", len(train_loader))
    logging.info("Val batches: %d", len(val_loader))
    logging.info("Test batches: %d", len(test_loader))

    # =================================================================
    # Model Creation
    # =================================================================
    
    logging.info("\n%s", "="*80)
    logging.info("CREATING MODEL")
    logging.info("%s", "="*80)

    model = FunctionalGraspModel(
        CSEM=config['model']['CSEM'],
        CGEO=config['model']['CGEO'],
        DPOSE=config['model']['DPOSE'],
        K_CONTACT=config['model']['K_CONTACT'],
        qwen_tuning=config['model']['qwen_tuning'],
        lora_cfg=config['lora'],
        contact_regression=config['model']['contact_regression']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total parameters: %s", f"{total_params:,}")
    logging.info("Trainable parameters: %s (%.2f%%)",
                 f"{trainable_params:,}", 100*trainable_params/total_params)

    # =================================================================
    # Optimization Setup
    # =================================================================
    logging.info("\n%s", "="*80)
    logging.info("OPTIMIZATION SETUP")
    logging.info("%s", "="*80)

    # Loss functions
    contact_criterion = ContactLoss(no_contact_weight=config['data']['no_contact_weight'])
    flow_criterion = FlowMatchingLoss()
    logging.info("Contact loss: BCE with no_contact_weight=%.2f", config['data']['no_contact_weight'])
    logging.info("Flow loss: MSE for rectified flow")
    logging.info("Loss weights: λ_contact=%.1f, λ_flow=%.1f",
                 config['training']['lambda_contact'], config['training']['lambda_flow'])

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    logging.info("Optimizer: AdamW (lr=%.4f, wd=%.2f)",
                 config['training']['learning_rate'], config['training']['weight_decay'])

    # Learning rate scheduler with warmup
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of base LR
        end_factor=1.0,    # Reach 100% after warmup
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    logging.info("Scheduler: Warmup (%d epochs) + CosineAnnealing (T_max=%d, eta_min=1e-6)",
                 warmup_epochs, config['training']['epochs'] - warmup_epochs)

    # =================================================================
    # Training Loop
    # =================================================================
    logging.info("\n%s", "="*80)
    logging.info("STARTING TRAINING")
    logging.info("%s", "="*80)
    logging.info("Epochs: %d", config['training']['epochs'])
    logging.info("Batch size: %d", config['training']['batch_size'])
    logging.info("Gradient clip: %.1f", config['training']['gradient_clip'])

    best_val_total = float('inf')
    global_step = 0

    for epoch in range(config['training']['epochs']):
        logging.info("\n%s", "="*80)
        logging.info("Epoch %d/%d", epoch+1, config['training']['epochs'])
        logging.info("%s", "="*80)

        # Train (includes step-based validation)
        train_metrics, global_step, best_val_total = train_epoch(
            model, train_loader, optimizer, scheduler,
            contact_criterion, flow_criterion,
            val_loader, test_loader,
            config, epoch, device, global_step, best_val_total
        )

        # Log epoch summary with current LR
        current_lr = scheduler.get_last_lr()[0]
        logging.info("\nEpoch %d Summary:", epoch+1)
        logging.info("  Train - Loss: %.4f, Contact: %.4f, Flow: %.4f, Acc: %.3f",
                    train_metrics['loss'], train_metrics['contact_loss'],
                    train_metrics['flow_loss'], train_metrics['contact_accuracy'])
        logging.info("  Learning rate: %.6f", current_lr)

        # Save periodic checkpoints
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics,
                os.path.join(config['paths']['checkpoint_dir'], f'epoch_{epoch+1:03d}.pth')
            )

        # Step scheduler (warmup + cosine annealing)
        scheduler.step()

    # =================================================================
    # Training Complete
    # =================================================================
    logging.info("\n%s", "="*80)
    logging.info("TRAINING COMPLETE")
    logging.info("%s", "="*80)
    logging.info("Best validation Total Loss: %.4f", best_val_total)
    logging.info("Checkpoints saved to: %s", config['paths']['checkpoint_dir'])
    logging.info("Training logs saved to: %s", log_file)
    logging.info("Quick access: %s", run_log)
    logging.info("Visualizations saved to: %s", config['paths']['qual_dir'])
    logging.info("\n✅ Training finished successfully!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logging.error("\n\n❌ Training failed with error:")
        logging.error("%s: %s", type(e).__name__, str(e), exc_info=True)
        sys.exit(1)

