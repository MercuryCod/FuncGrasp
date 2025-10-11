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

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def compute_contact_predictions(logits_c, contact_regression=True, inference_threshold=0.4):
    """
    Compute contact class predictions from logits.
    
    Args:
        logits_c: [B, N, 7] contact logits
        contact_regression: Whether using regression (BCE) or classification (CE)
        inference_threshold: Threshold for no_contact in regression mode
    
    Returns:
        predictions: [B, N] predicted class indices (0-6)
    """
    if contact_regression:
        # Regression mode: use threshold on max part probability
        part_probs = torch.sigmoid(logits_c[..., :6])  # [B, N, 6]
        max_part_probs, max_part_indices = torch.max(part_probs, dim=-1)  # [B, N]
        
        # Predict no_contact (class 6) if max part prob <= threshold
        predictions = torch.where(
            max_part_probs <= inference_threshold,
            torch.full_like(max_part_indices, 6),  # no_contact
            max_part_indices  # predicted part
        )
    else:
        # Classification mode: simple argmax
        predictions = torch.argmax(logits_c, dim=-1)  # [B, N]
    
    return predictions


def compute_contact_metrics(predictions, labels, num_classes=7):
    """
    Compute per-class accuracy, macro-F1, and confusion matrix.
    
    Args:
        predictions: [B, N] predicted class indices
        labels: [B, N] ground truth class indices
        num_classes: Number of classes (default 7)
    
    Returns:
        dict with metrics
    """
    # Flatten for sklearn
    pred_flat = predictions.cpu().numpy().flatten()
    label_flat = labels.cpu().numpy().flatten()
    
    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = label_flat == c
        if mask.sum() > 0:
            acc = (pred_flat[mask] == c).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    
    # Macro-F1
    macro_f1 = f1_score(label_flat, pred_flat, average='macro', labels=list(range(num_classes)), zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(label_flat, pred_flat, labels=list(range(num_classes)))
    
    return {
        'per_class_accuracy': per_class_acc,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_matrix
    }


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
        
        # Contact loss: branch based on loss type
        logits_c = out["logits_c"]  # [B, N, 7]
        B, N = logits_c.shape[:2]
        
        if cfg['training']['contact_loss_type'] == 'bce':
            # BCE with logits for regression
            contact_targets = batch["contact_targets"].to(device)  # [B, N, 7]
            
            # Compute pos_weight if available
            pos_weight = cfg['training'].get('pos_weight', None)
            if pos_weight is not None:
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
            
            loss_contact = F.binary_cross_entropy_with_logits(
                logits_c, 
                contact_targets,
                pos_weight=pos_weight
            )
        else:
            # CE for classification
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
                pred_contacts = compute_contact_predictions(
                    logits_c,
                    contact_regression=cfg['model']['contact_regression'],
                    inference_threshold=cfg['model']['inference_threshold']
                )
                contact_acc = (pred_contacts == contact_labels).float().mean()
                
                if writer is not None:
                    writer.add_scalar('Metrics/contact_accuracy', contact_acc.item(), global_step)
            
            print(f"Step {global_step:06d} | loss={loss.item():.4f} | contact={loss_contact.item():.4f} | flow={loss_flow.item():.4f} | acc={contact_acc.item():.3f}")
        
        # Checkpoint saving
        if global_step > 0 and global_step % cfg['training']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, global_step, cfg)
        
        global_step += 1
    
    return global_step


def generate_qualitative_visualizations(model, loader, cfg, device, global_step, logger):
    """
    Generate qualitative visualizations comparing predicted and ground-truth contact maps.
    
    Args:
        model: Trained model
        loader: Validation dataloader
        cfg: Config dict
        device: Device
        global_step: Current training step
        logger: Logger
    """
    from utils.visualization_3d import create_hand_object_figure
    
    model.eval()
    qual_dir = Path(cfg['paths']['qual_dir'])
    qual_dir.mkdir(parents=True, exist_ok=True)
    
    num_to_generate = min(cfg['eval']['num_qualitative'], len(loader.dataset))
    
    with torch.no_grad():
        # Get first batch
        batch = next(iter(loader))
        
        # Move to device
        pts = batch["points"].to(device)
        contact_labels = batch["contact_labels"]  # Keep on CPU for visualization
        
        # Forward pass
        out = model.forward_train(
            images_list=batch["images_list"],
            texts_list=batch["texts_list"],
            pts=pts
        )
        
        logits_c = out["logits_c"]  # [B, N, 7]
        
        # Get predictions
        pred_contacts = compute_contact_predictions(
            logits_c,
            contact_regression=cfg['model']['contact_regression'],
            inference_threshold=cfg['model']['inference_threshold']
        )  # [B, N]
        
        # Generate poses
        pred_poses = model.sample(
            batch["images_list"],
            batch["texts_list"],
            pts,
            num_steps=20,
            device=device
        )  # [B, 63]
        
        # Generate visualizations
        for i in range(min(num_to_generate, len(batch['points']))):
            # Get data for this sample
            points_i = batch['points'][i].cpu().numpy()  # [N, 3]
            pred_labels_i = pred_contacts[i].cpu().numpy()  # [N]
            gt_labels_i = contact_labels[i].cpu().numpy()  # [N]
            pred_pose_i = pred_poses[i].cpu().numpy().reshape(21, 3)  # [21, 3]
            gt_pose_i = batch['pose'][i].cpu().numpy().reshape(21, 3)  # [21, 3]
            
            # Get hand vertices if available
            hand_vertices_i = None
            if 'meta' in batch and i < len(batch['meta']):
                hand_vertices_i = batch['meta'][i].get('hand_vertices', None)
            
            # Generate prediction figure
            fig_pred = create_hand_object_figure(
                points_i,
                pred_labels_i,
                hand_vertices=None,  # Don't show GT vertices with prediction
                hand_joints=pred_pose_i,
                show_joints=True,
                figsize=(15, 5)
            )
            fig_pred.suptitle(f'Predicted Contact Map & Pose (Step {global_step}, Sample {i})', fontsize=14)
            pred_path = qual_dir / f'val_step_{global_step}_sample_{i}_pred.png'
            fig_pred.savefig(pred_path, dpi=100, bbox_inches='tight')
            plt.close(fig_pred)
            
            # Generate ground truth figure if enabled
            if cfg['eval']['save_gt_qualitative']:
                fig_gt = create_hand_object_figure(
                    points_i,
                    gt_labels_i,
                    hand_vertices=hand_vertices_i,
                    hand_joints=gt_pose_i,
                    show_joints=True,
                    figsize=(15, 5)
                )
                fig_gt.suptitle(f'Ground Truth Contact Map & Pose (Step {global_step}, Sample {i})', fontsize=14)
                gt_path = qual_dir / f'val_step_{global_step}_sample_{i}_gt.png'
                fig_gt.savefig(gt_path, dpi=100, bbox_inches='tight')
                plt.close(fig_gt)
                
                logger.info(f"Saved qualitative visualization pair: {pred_path.name}, {gt_path.name}")
            else:
                logger.info(f"Saved qualitative visualization: {pred_path.name}")


def validate(model, loader, cfg, device, writer, global_step):
    """
    Validate the model with enhanced metrics.
    
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
    logger = logging.getLogger('funcgrasp')
    
    total_loss = 0
    total_contact_loss = 0
    total_flow_loss = 0
    total_contact_acc = 0
    num_batches = 0
    
    # Accumulate predictions and labels for metrics
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
            
            if cfg['training']['contact_loss_type'] == 'bce':
                contact_targets = batch["contact_targets"].to(device)
                pos_weight = cfg['training'].get('pos_weight', None)
                if pos_weight is not None:
                    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
                loss_contact = F.binary_cross_entropy_with_logits(
                    logits_c, contact_targets, pos_weight=pos_weight
                )
            else:
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
            
            # Predictions
            pred_contacts = compute_contact_predictions(
                logits_c,
                contact_regression=cfg['model']['contact_regression'],
                inference_threshold=cfg['model']['inference_threshold']
            )
            contact_acc = (pred_contacts == contact_labels).float().mean()
            
            # Accumulate for metrics
            all_predictions.append(pred_contacts)
            all_labels.append(contact_labels)
            
            # Accumulate losses
            total_loss += loss.item()
            total_contact_loss += loss_contact.item()
            total_flow_loss += loss_flow.item()
            total_contact_acc += contact_acc.item()
            num_batches += 1
    
    # Compute enhanced metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    contact_metrics = compute_contact_metrics(all_predictions, all_labels)
    
    # Average metrics
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_contact_loss': total_contact_loss / num_batches,
        'val_flow_loss': total_flow_loss / num_batches,
        'val_contact_acc': total_contact_acc / num_batches,
        'val_macro_f1': contact_metrics['macro_f1'],
        'val_per_class_acc': contact_metrics['per_class_accuracy'],
        'val_confusion_matrix': contact_metrics['confusion_matrix']
    }
    
    # Log to run.log
    logger.info(f"Validation at step {global_step}:")
    logger.info(f"  Loss: {metrics['val_loss']:.4f} | Contact: {metrics['val_contact_loss']:.4f} | Flow: {metrics['val_flow_loss']:.4f}")
    logger.info(f"  Overall Acc: {metrics['val_contact_acc']:.4f} | Macro-F1: {metrics['val_macro_f1']:.4f}")
    logger.info(f"  Per-class Acc: {[f'{a:.3f}' for a in metrics['val_per_class_acc']]}")
    logger.info(f"  Confusion Matrix:\n{metrics['val_confusion_matrix']}")
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Validation/val_loss', metrics['val_loss'], global_step)
        writer.add_scalar('Validation/val_contact_loss', metrics['val_contact_loss'], global_step)
        writer.add_scalar('Validation/val_flow_loss', metrics['val_flow_loss'], global_step)
        writer.add_scalar('Validation/val_contact_acc', metrics['val_contact_acc'], global_step)
        writer.add_scalar('Validation/val_macro_f1', metrics['val_macro_f1'], global_step)
        
        # Log per-class accuracy
        for i, acc in enumerate(metrics['val_per_class_acc']):
            writer.add_scalar(f'Validation/per_class_acc_{Config.CONTACT_CLASSES[i]}', acc, global_step)
    
    # Generate qualitative visualizations if enabled
    if cfg['eval']['num_qualitative'] > 0:
        generate_qualitative_visualizations(
            model, loader, cfg, device, global_step, logger
        )
    
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
    
    # Contact regression arguments
    parser.add_argument('--contact_regression', action='store_true',
                        help='Use contact regression (BCE) instead of classification (CE)')
    parser.add_argument('--no_contact_regression', dest='contact_regression', action='store_false',
                        help='Use classification (CE) instead of regression (BCE)')
    parser.add_argument('--contact_loss_type', type=str, default=None,
                        choices=['bce', 'ce'],
                        help='Contact loss type (bce or ce)')
    parser.add_argument('--inference_threshold', type=float, default=None,
                        help='Threshold for no_contact prediction in regression mode')
    parser.add_argument('--tau_mm', type=float, default=None,
                        help='Gaussian decay parameter for soft targets (mm)')

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
    
    # Override contact regression settings
    if hasattr(args, 'contact_regression') and args.contact_regression is not None:
        cfg['model']['contact_regression'] = args.contact_regression
    if args.contact_loss_type:
        cfg['training']['contact_loss_type'] = args.contact_loss_type
    if args.inference_threshold is not None:
        cfg['model']['inference_threshold'] = args.inference_threshold
    if args.tau_mm is not None:
        cfg['regression_hparams']['tau_mm'] = args.tau_mm
    
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
    
    # Load class frequencies if available
    freq_file = Path(cfg['data']['root_dir']) / 'cache' / 'class_frequencies_train.json'
    if freq_file.exists() and cfg['training']['pos_weight'] is None:
        import json
        try:
            with open(freq_file, 'r') as f:
                freq_data = json.load(f)
            cfg['training']['pos_weight'] = freq_data['pos_weight']
            logger.info(f"Loaded pos_weight from cache: {cfg['training']['pos_weight']}")
        except Exception as e:
            logger.warning(f"Failed to load pos_weight from {freq_file}: {e}")
    
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
        use_cache=cfg['data']['use_cache'],
        use_soft_targets=cfg['model']['contact_regression'],
        regression_hparams=cfg['regression_hparams']
    )
    
    # Model
    logger.info("Initializing model...")
    model = FunctionalGraspModel(
        CSEM=cfg['model']['CSEM'],
        CGEO=cfg['model']['CGEO'],
        DPOSE=cfg['model']['DPOSE'],
        K_CONTACT=cfg['model']['K_CONTACT'],
        qwen_tuning=cfg['model']['qwen_tuning'],
        lora_cfg=cfg['lora'] if cfg['model']['qwen_tuning'] == 'lora' else None,
        contact_regression=cfg['model']['contact_regression'],
        inference_threshold=cfg['model']['inference_threshold']
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
