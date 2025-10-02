"""
Main training script for functional grasp model.
Implements training loop with contact and flow matching losses.
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter


from models.functional_grasp_model import FunctionalGraspModel
from config import Config
from dataset.oakink_loader import create_oakink_loaders


def compute_flow_matching_loss(model, conditioning, target_pose, bone_length_reg=0.0):
    """
    Compute rectified flow matching loss.
    
    Args:
        model: FunctionalGraspModel
        conditioning: [B, CCOND] conditioning vector
        target_pose: [B, DPOSE] target grasp poses (flattened 21×3 joints)
        bone_length_reg: Weight for bone length regularization (optional)
    
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
    
    # Main flow matching loss
    loss = F.mse_loss(v_pred, v_target)
    
    # Optional bone length regularization for joint representation
    if bone_length_reg > 0 and target_pose.shape[1] == 63:
        # Reshape to joints: [B, 21, 3]
        joints_pred = (x_t + t.unsqueeze(-1) * v_pred).reshape(B, 21, 3)
        
        # Define hand bone connections (parent -> child)
        # Based on standard hand skeleton
        bone_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Little finger
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Expected bone lengths (in meters, approximate for adult hand)
        # These should be computed from training data statistics
        expected_lengths = torch.tensor([
            # Thumb bones
            0.050, 0.035, 0.025, 0.020,
            # Index finger
            0.080, 0.045, 0.030, 0.020,
            # Middle finger
            0.085, 0.050, 0.035, 0.025,
            # Ring finger
            0.080, 0.045, 0.030, 0.025,
            # Little finger
            0.070, 0.035, 0.025, 0.020
        ], device=device)
        
        # Compute bone length regularization
        bone_loss = 0.0
        for i, (parent, child) in enumerate(bone_connections):
            bone_vec = joints_pred[:, child] - joints_pred[:, parent]
            bone_len = torch.norm(bone_vec, dim=1)
            bone_loss += F.mse_loss(bone_len, expected_lengths[i].expand(B))
        
        loss = loss + bone_length_reg * bone_loss / len(bone_connections)
    
    return loss


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
    
    # Iterator over loader
    iterator = iter(loader)
    
    grad_accum_steps = max(1, cfg['training'].get('gradient_accumulation', 1))
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(iterator):
        # Move data to device
        pts = batch["points"].to(device)
        contact_labels = batch["contact_labels"].to(device)
        y_pose = batch["pose"].to(device)
        
        # Forward pass
        out = model.forward_train(
            images_list=batch["images_list"],
            texts_list=batch["texts_list"],
            pts=pts
        )
        
        # Contact loss (7-way classification)
        logits_c = out["logits_c"]  # [B, N, 7]
        B, N = logits_c.shape[:2]
        loss_contact = F.cross_entropy(
            logits_c.view(B * N, 7),
            contact_labels.view(B * N)
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

        # Backward pass with gradient accumulation
        (loss / grad_accum_steps).backward()

        # Optimizer step on accumulation boundary or last batch
        if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == len(loader) - 1):
            # Gradient clipping just before stepping
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                cfg['training']['gradient_clip']
            )
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging (step-based across epochs)
        if global_step % cfg['training']['log_interval'] == 0:
            if writer is not None:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/contact', loss_contact.item(), global_step)
                writer.add_scalar('Loss/flow', loss_flow.item(), global_step)
            
            # Contact accuracy
            with torch.no_grad():
                pred_contacts = logits_c.argmax(-1)  # [B, N]
                contact_acc = (pred_contacts == contact_labels).float().mean()
                
                if writer is not None:
                    writer.add_scalar('Metrics/contact_accuracy', contact_acc.item(), global_step)
            
            print(f"Step {global_step:06d} | loss={loss.item():.4f} | contact={loss_contact.item():.4f} | flow={loss_flow.item():.4f} | acc={contact_acc.item():.3f}")
        
        # Checkpoint saving
        if global_step > 0 and global_step % cfg['training']['checkpoint_interval'] == 0:
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
        for batch in loader:
            # Move data to device
            pts = batch["points"].to(device)
            contact_labels = batch["contact_labels"].to(device)
            y_pose = batch["pose"].to(device)
            
            # Forward pass
            out = model.forward_train(
                images_list=batch["images_list"],
                texts_list=batch["texts_list"],
                pts=pts
            )
            
            # Losses
            logits_c = out["logits_c"]  # [B, N, 7]
            B, N = logits_c.shape[:2]
            loss_contact = F.cross_entropy(
                logits_c.view(B * N, 7),
                contact_labels.view(B * N)
            )
            
            loss_flow = compute_flow_matching_loss(
                model,
                out["cond"],
                y_pose
            )
            
            loss = (cfg['training']['lambda_contact'] * loss_contact + 
                   cfg['training']['lambda_flow'] * loss_flow)
            
            # Metrics
            pred_contacts = logits_c.argmax(-1)  # [B, N]
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
    logging.getLogger('funcgrasp').info("Saved checkpoint to %s", path)
    
    # Save LoRA adapters separately if using LoRA
    if cfg['model'].get('qwen_tuning') == 'lora':
        lora_path = os.path.join(
            cfg['paths']['checkpoint_dir'],
            f'lora_adapters_step_{step}'
        )
        model.sem.save_lora_weights(lora_path)
        logging.getLogger('funcgrasp').info("LoRA weights saved to %s", lora_path)


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
    parser.add_argument('--checkpoint_interval', type=int, default=None,
                        help='Batches between checkpoint saves during training')
    
    # LoRA-specific arguments
    parser.add_argument('--qwen_tuning', type=str, default=None,
                        choices=['frozen', 'full', 'lora'],
                        help='Qwen tuning mode (default: frozen)')
    parser.add_argument('--lora_r', type=int, default=None,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=None,
                        help='LoRA alpha (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=None,
                        help='LoRA dropout (default: 0.05)')
    parser.add_argument('--lora_targets', type=str, default=None,
                        help='Comma-separated LoRA target modules (default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)')
    parser.add_argument('--lora_bias', type=str, default=None,
                        choices=['none', 'all', 'lora_only'],
                        help='LoRA bias mode (default: none)')
    parser.add_argument('--lora_use_8bit', action='store_true',
                        help='Use 8-bit base model with LoRA (requires bitsandbytes)')
    
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Batches between console/TensorBoard logs')

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
    if args.log_interval:
        cfg['training']['log_interval'] = args.log_interval
    if args.checkpoint_interval:
        cfg['training']['checkpoint_interval'] = args.checkpoint_interval
    
    # Override LoRA settings
    if args.qwen_tuning:
        cfg['model']['qwen_tuning'] = args.qwen_tuning
    if args.lora_r is not None:
        cfg['lora']['r'] = args.lora_r
    if args.lora_alpha is not None:
        cfg['lora']['alpha'] = args.lora_alpha
    if args.lora_dropout is not None:
        cfg['lora']['dropout'] = args.lora_dropout
    if args.lora_targets:
        cfg['lora']['target_modules'] = args.lora_targets.split(',')
    if args.lora_bias:
        cfg['lora']['bias'] = args.lora_bias
    if args.lora_use_8bit:
        cfg['lora']['use_8bit'] = True
    
    # Initialize logging to file (run.log)
    os.makedirs(cfg['paths']['log_dir'], exist_ok=True)
    log_path = os.path.join(cfg['paths']['log_dir'], 'run.log')
    logger = logging.getLogger('funcgrasp')
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    logger.handlers = []
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Redirect stdout/stderr to logger so prints from dependencies also go to run.log
    class _LoggerWriter:
        def __init__(self, _logger, level):
            self._logger = _logger
            self._level = level
        def write(self, message):
            message = str(message)
            if message and not message.isspace():
                for line in message.rstrip().splitlines():
                    self._logger.log(self._level, line)
        def flush(self):
            pass
    sys.stdout = _LoggerWriter(logger, logging.INFO)
    sys.stderr = _LoggerWriter(logger, logging.WARNING)

    # Display tuning mode
    logger.info("Qwen tuning mode: %s", cfg['model']['qwen_tuning'])
    if cfg['model']['qwen_tuning'] == 'lora':
        logger.info("LoRA config: r=%s, alpha=%s, dropout=%s", cfg['lora']['r'], cfg['lora']['alpha'], cfg['lora']['dropout'])
        logger.info("LoRA targets: %s", cfg['lora']['target_modules'])
    
    # Create directories
    Config.create_dirs()
    
    # Device
    device = torch.device(cfg['device'])
    logger.info("Using device: %s", device)
    
    # Data loaders
    logger.info("Loading data...")
    # Use all 6 views
    num_views = 6
    train_loader, val_loader = create_oakink_loaders(
        root_dir=cfg['data']['root_dir'],
        render_dir=cfg['data'].get('render_dir', './rendered_objects'),
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['num_workers'],
        split_mode=cfg['data']['split_mode'],
        n_points=cfg['model']['n_points'],
        contact_threshold=cfg['data']['contact_threshold'],
        num_views=num_views,
        use_cache=cfg['data']['use_cache']
    )
    
    # Model
    logger.info("Initializing model...")
    model = FunctionalGraspModel(
        CSEM=cfg['model']['CSEM'],
        CGEO=cfg['model']['CGEO'],
        DPOSE=cfg['model']['DPOSE'],
        K_CONTACT=cfg['model']['K_CONTACT'],
        qwen_tuning=cfg['model']['qwen_tuning'],
        lora_cfg=cfg['lora'] if cfg['model']['qwen_tuning'] == 'lora' else None
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    
    # Optimizer
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
        logger.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle LoRA adapter loading if needed
        if cfg['model']['qwen_tuning'] == 'lora' and 'lora_state_dict' in checkpoint:
            # For LoRA, we need special handling
            # Load non-LoRA parameters normally
            model_state = checkpoint['model_state_dict']
            model.load_state_dict(model_state, strict=False)
            # LoRA adapters are loaded as part of the model state
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        start_epoch = global_step // len(train_loader)
        
        # Try to load LoRA adapters from separate file if exists
        if cfg['model']['qwen_tuning'] == 'lora':
            lora_path = os.path.join(
                os.path.dirname(args.checkpoint),
                f'lora_adapters_step_{global_step}'
            )
            if os.path.exists(lora_path):
                logger.info("Loading LoRA adapters from %s", lora_path)
                # Note: actual loading handled by PEFT through model state dict
    
    # TensorBoard writer
    writer = SummaryWriter(cfg['paths']['log_dir'])
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg['training']['epochs']):
        # print(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        # Train
        global_step = train_one_epoch(
            model, train_loader, optimizer, cfg, device, writer, global_step
        )
        
        # Validate
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            logger.info("Running validation...")
            metrics = validate(model, val_loader, cfg, device, writer, global_step)
            logger.info("Validation metrics: %s", metrics)
        
        # Note: Checkpoints are saved strictly by step via checkpoint_interval
    
    if writer is not None:
        writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
