"""
DeepSpeed training script for functional grasp model.
Implements distributed training with ZeRO optimization.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import deepspeed
from deepspeed import comm as dist
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
HAS_TENSORBOARD = True

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
    v_pred = model.module.flow_step(x_t, t, conditioning)

    # Target velocity (constant for rectified flow)
    v_target = target_pose - x0

    # Main flow matching loss
    loss = F.mse_loss(v_pred, v_target)

    # Optional bone length regularization for joint representation
    if bone_length_reg > 0 and target_pose.shape[1] == 63:
        # Reshape to joints: [B, 21, 3]
        joints_pred = (x_t + t.unsqueeze(-1) * v_pred).reshape(B, 21, 3)

        # Define hand bone connections (parent -> child)
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


def train_one_epoch(model_engine, loader, cfg, device, writer, global_step):
    """
    Train for one epoch with DeepSpeed.

    Args:
        model_engine: DeepSpeed engine
        loader: DataLoader
        cfg: Configuration dict
        device: Device to train on
        writer: TensorBoard writer
        global_step: Global training step

    Returns:
        global_step: Updated global step
    """
    model_engine.train()

    # Progress bar (only on rank 0)
    if dist.get_rank() == 0:
        pbar = tqdm(loader, desc="Training")
    else:
        pbar = loader

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        pts = batch["points"].to(device)
        contact_labels = batch["contact_labels"].to(device)
        y_pose = batch["pose"].to(device)

        # Forward pass
        out = model_engine.module.forward_train(
            images_list=batch["images_list"],
            texts_list=batch["texts_list"],
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
            model_engine,
            out["cond"],
            y_pose
        )

        # Total loss
        loss = (cfg['training']['lambda_contact'] * loss_contact +
                cfg['training']['lambda_flow'] * loss_flow)

        # Backward pass with DeepSpeed
        model_engine.backward(loss)
        model_engine.step()

        # Logging (only on rank 0)
        if dist.get_rank() == 0 and batch_idx % cfg['training']['log_interval'] == 0:
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

        # Checkpoint saving (handled by DeepSpeed)
        if global_step > 0 and global_step % cfg['training']['checkpoint_interval'] == 0:
            save_checkpoint_deepspeed(model_engine, cfg, global_step)

        global_step += 1

    return global_step


def validate(model_engine, loader, cfg, device, writer, global_step):
    """
    Validate the model.

    Args:
        model_engine: DeepSpeed engine
        loader: Validation DataLoader
        cfg: Configuration dict
        device: Device
        writer: TensorBoard writer
        global_step: Current training step

    Returns:
        metrics: Dictionary of validation metrics
    """
    model_engine.eval()

    total_loss = 0
    total_contact_loss = 0
    total_flow_loss = 0
    total_contact_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", disable=(dist.get_rank() != 0)):
            # Move data to device
            pts = batch["points"].to(device)
            contact_labels = batch["contact_labels"].to(device)
            y_pose = batch["pose"].to(device)

            # Forward pass
            out = model_engine.module.forward_train(
                images_list=batch["images_list"],
                texts_list=batch["texts_list"],
                pts=pts
            )

            # Losses
            logits_c = out["logits_c"]
            loss_contact = F.binary_cross_entropy_with_logits(
                logits_c.squeeze(-1),
                contact_labels.float()
            )

            loss_flow = compute_flow_matching_loss(
                model_engine,
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

    # Log to tensorboard (only on rank 0)
    if dist.get_rank() == 0 and writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'Validation/{key}', value, global_step)

    return metrics


def save_checkpoint_deepspeed(model_engine, cfg, step):
    """Save DeepSpeed checkpoint."""
    checkpoint_dir = os.path.join(
        cfg['paths']['checkpoint_dir'],
        f'deepspeed_ckpt_step_{step}'
    )
    model_engine.save_checkpoint(checkpoint_dir)
    if dist.get_rank() == 0:
        print(f"Saved checkpoint to {checkpoint_dir}")


def load_checkpoint_deepspeed(model_engine, checkpoint_dir):
    """Load DeepSpeed checkpoint."""
    _, client_states = model_engine.load_checkpoint(checkpoint_dir)
    step = client_states.get('step', 0) if client_states else 0
    if dist.get_rank() == 0:
        print(f"Loaded checkpoint from {checkpoint_dir}, resuming from step {step}")
    return step


def main():
    parser = argparse.ArgumentParser(description="Train functional grasp model with DeepSpeed")

    # Standard training arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to OakInk dataset')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with checkpoint to resume from')

    # DeepSpeed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')

    # Include DeepSpeed's argument parser (this adds --deepspeed_config among other args)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Get configuration
    cfg = Config.get_config(mode='train')

    # Override with command line arguments
    if args.data_path:
        cfg['data']['root_dir'] = args.data_path
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['epochs'] = args.epochs

    # Create directories (only on rank 0)
    if dist.get_rank() == 0:
        Config.create_dirs()

    # Device
    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(args.local_rank)

    if dist.get_rank() == 0:
        print(f"Using DeepSpeed with {dist.get_world_size()} GPUs")
        print(f"DeepSpeed config: {args.deepspeed_config}")

    # Data loaders with DistributedSampler
    if dist.get_rank() == 0:
        print("Loading data...")

    num_views = 1 if cfg['data'].get('single_view', True) else 6
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
        distributed=True,  # Enable distributed sampling
        world_size=dist.get_world_size(),
        rank=dist.get_rank()
    )

    # Model
    if dist.get_rank() == 0:
        print("Initializing model...")

    model = FunctionalGraspModel(
        CSEM=cfg['model']['CSEM'],
        CGEO=cfg['model']['CGEO'],
        DPOSE=cfg['model']['DPOSE'],
        K_CONTACT=cfg['model']['K_CONTACT']
    )

    # Count parameters (before DeepSpeed wrapping)
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Load DeepSpeed configuration
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    # Update config with runtime values
    ds_config['train_batch_size'] = cfg['training']['batch_size'] * dist.get_world_size()
    ds_config['train_micro_batch_size_per_gpu'] = cfg['training']['batch_size']
    ds_config['gradient_accumulation_steps'] = cfg['training'].get('gradient_accumulation', 1)
    ds_config['gradient_clipping'] = cfg['training']['gradient_clip']

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Load checkpoint if provided
    global_step = 0
    start_epoch = 0
    if args.checkpoint_dir:
        global_step = load_checkpoint_deepspeed(model_engine, args.checkpoint_dir)
        start_epoch = global_step // len(train_loader)

    # TensorBoard writer (only on rank 0)
    writer = None
    if dist.get_rank() == 0 and HAS_TENSORBOARD:
        writer = SummaryWriter(cfg['paths']['log_dir'] + '_deepspeed')

    # Training loop
    if dist.get_rank() == 0:
        print("Starting DeepSpeed training...")

    for epoch in range(start_epoch, cfg['training']['epochs']):
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        # Set epoch for DistributedSampler
        train_loader.sampler.set_epoch(epoch)

        # Train
        global_step = train_one_epoch(
            model_engine, train_loader, cfg, device, writer, global_step
        )

        # Validate
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            if dist.get_rank() == 0:
                print("Running validation...")
            metrics = validate(model_engine, val_loader, cfg, device, writer, global_step)
            if dist.get_rank() == 0:
                print(f"Validation metrics: {metrics}")

        # Save checkpoint at end of epoch
        save_checkpoint_deepspeed(model_engine, cfg, global_step)

    if writer is not None:
        writer.close()

    if dist.get_rank() == 0:
        print("DeepSpeed training complete!")


if __name__ == "__main__":
    main()