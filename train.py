"""
Main training script for functional grasp model.
Implements training loop with contact and flow matching losses.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not installed. Logging will be disabled.")

from models.functional_grasp_model import FunctionalGraspModel
from config import Config
from data.oakink_loader import create_oakink_loaders


def compute_flow_matching_loss(model, conditioning, target_pose):
    """
    Compute rectified flow matching loss.
    
    Args:
        model: FunctionalGraspModel
        conditioning: [B, CCOND] conditioning vector
        target_pose: [B, DPOSE] target grasp poses
    
    Returns:
        loss: Flow matching loss
    """
    B = target_pose.size(0)
    device = target_pose.device
    
    # Sample noise and time
    x0 = torch.randn_like(target_pose)  # Start: Gaussian noise
    t = torch.rand(B, device=device)    # Time: U[0,1]
    
    # Interpolate between noise and target
    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * target_pose
    
    # Predict velocity field
    v_pred = model.flow_step(x_t, t, conditioning)
    
    # Target velocity (constant for rectified flow)
    v_target = target_pose - x0
    
    return F.mse_loss(v_pred, v_target)


def train_one_epoch(model, loader, optimizer, cfg, device, writer, global_step):
    """
    Train for one epoch.
    
    Args:
        model: FunctionalGraspModel
        loader: DataLoader
        optimizer: Optimizer
        cfg: Configuration dict
        device: Device to train on
        writer: TensorBoard writer
        global_step: Global training step
    
    Returns:
        global_step: Updated global step
    """
    model.train()
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Progress bar
    pbar = tqdm(loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        pts = batch["points"].to(device)
        contact_labels = batch["contact_labels"].to(device)
        y_pose = batch["pose"].to(device)
        
        # Forward pass
        out = model.forward_train(
            images=batch["images"],
            texts=batch["texts"],
            pts=pts
        )
        
        # Contact loss
        logits_c = out["logits_c"]
        loss_contact = F.binary_cross_entropy_with_logits(
            logits_c.squeeze(-1),
            contact_labels.float()
        )
        
        # Flow matching loss
        loss_flow = compute_flow_matching_loss(
            model,
            out["cond"],
            y_pose
        )
        
        # Total loss
        loss = (cfg['training']['lambda_contact'] * loss_contact + 
                cfg['training']['lambda_flow'] * loss_flow)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            cfg['training']['gradient_clip']
        )
        
        optimizer.step()
        
        # Logging
        if batch_idx % cfg['training']['log_interval'] == 0:
            if writer is not None:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/contact', loss_contact.item(), global_step)
                writer.add_scalar('Loss/flow', loss_flow.item(), global_step)
            
            # Contact accuracy
            with torch.no_grad():
                pred_contacts = torch.sigmoid(logits_c.squeeze(-1)) > 0.5
                contact_acc = (pred_contacts == contact_labels).float().mean()
                if writer is not None:
                    writer.add_scalar('Metrics/contact_accuracy', contact_acc.item(), global_step)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'contact': f'{loss_contact.item():.4f}',
                'flow': f'{loss_flow.item():.4f}',
                'acc': f'{contact_acc.item():.3f}'
            })
        
        # Checkpoint saving
        if global_step % cfg['training']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, global_step, cfg)
        
        global_step += 1
    
    return global_step


def validate(model, loader, cfg, device, writer, global_step):
    """
    Validate the model.
    
    Args:
        model: FunctionalGraspModel
        loader: Validation DataLoader
        cfg: Configuration dict
        device: Device
        writer: TensorBoard writer
        global_step: Current training step
    
    Returns:
        metrics: Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0
    total_contact_loss = 0
    total_flow_loss = 0
    total_contact_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            # Move data to device
            pts = batch["points"].to(device)
            contact_labels = batch["contact_labels"].to(device)
            y_pose = batch["pose"].to(device)
            
            # Forward pass
            out = model.forward_train(
                images=batch["images"],
                texts=batch["texts"],
                pts=pts
            )
            
            # Losses
            logits_c = out["logits_c"]
            loss_contact = F.binary_cross_entropy_with_logits(
                logits_c.squeeze(-1),
                contact_labels.float()
            )
            
            loss_flow = compute_flow_matching_loss(
                model,
                out["cond"],
                y_pose
            )
            
            loss = (cfg['training']['lambda_contact'] * loss_contact + 
                   cfg['training']['lambda_flow'] * loss_flow)
            
            # Metrics
            pred_contacts = torch.sigmoid(logits_c.squeeze(-1)) > 0.5
            contact_acc = (pred_contacts == contact_labels).float().mean()
            
            # Accumulate
            total_loss += loss.item()
            total_contact_loss += loss_contact.item()
            total_flow_loss += loss_flow.item()
            total_contact_acc += contact_acc.item()
            num_batches += 1
    
    # Average metrics
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_contact_loss': total_contact_loss / num_batches,
        'val_flow_loss': total_flow_loss / num_batches,
        'val_contact_acc': total_contact_acc / num_batches,
    }
    
    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'Validation/{key}', value, global_step)
    
    return metrics


def save_checkpoint(model, optimizer, step, cfg):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': cfg,
    }
    
    path = os.path.join(
        cfg['paths']['checkpoint_dir'],
        f'checkpoint_step_{step}.pt'
    )
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train functional grasp model")
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to OakInk dataset')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Get configuration
    cfg = Config.get_config(mode='train')
    
    # Override with command line arguments
    if args.data_path:
        cfg['data']['root_dir'] = args.data_path
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.device:
        cfg['device'] = args.device
    
    # Create directories
    Config.create_dirs()
    
    # Device
    device = torch.device(cfg['device'])
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader = create_oakink_loaders(
        root_dir=cfg['data']['root_dir'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['num_workers'],
        split_mode=cfg['data']['split_mode'],
        n_points=cfg['model']['n_points'],
        contact_threshold=cfg['data']['contact_threshold'],
        single_view=cfg['data']['single_view'],
        use_cache=cfg['data']['use_cache']
    )
    
    # Model
    print("Initializing model...")
    model = FunctionalGraspModel(
        qwen_name=cfg['model']['qwen_name'],
        freeze_qwen=cfg['model'].get('freeze_qwen', True),
        CSEM=cfg['model']['CSEM'],
        CGEO=cfg['model']['CGEO'],
        DPOSE=cfg['model']['DPOSE'],
        K_CONTACT=cfg['model']['K_CONTACT']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with differential learning rates
    if not cfg['model'].get('freeze_qwen', True):
        # Separate parameter groups for backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'sem.backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        optimizer_groups = [
            {'params': backbone_params, 'lr': cfg['training'].get('backbone_lr', 1e-5)},
            {'params': head_params, 'lr': cfg['training']['learning_rate']}
        ]
        
        print(f"Backbone params: {sum(p.numel() for p in backbone_params):,}")
        print(f"Head params: {sum(p.numel() for p in head_params):,}")
        
        optimizer = AdamW(
            optimizer_groups,
            weight_decay=cfg['training']['weight_decay']
        )
    else:
        # Standard optimizer for frozen backbone
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params_list,
            lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay']
        )
    
    # Load checkpoint if provided
    global_step = 0
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        start_epoch = global_step // len(train_loader)
    
    # TensorBoard writer
    writer = SummaryWriter(cfg['paths']['log_dir']) if HAS_TENSORBOARD else None
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")
        
        # Train
        global_step = train_one_epoch(
            model, train_loader, optimizer, cfg, device, writer, global_step
        )
        
        # Validate
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            print("Running validation...")
            metrics = validate(model, val_loader, cfg, device, writer, global_step)
            print(f"Validation metrics: {metrics}")
        
        # Save checkpoint at end of epoch
        save_checkpoint(model, optimizer, global_step, cfg)
    
    if writer is not None:
        writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()