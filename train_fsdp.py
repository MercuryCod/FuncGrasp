"""
FSDP training script for functional grasp model.
Implements distributed training with Fully Sharded Data Parallel.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    StateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
HAS_TENSORBOARD = True

from models.functional_grasp_model import FunctionalGraspModel
from config import Config
from dataset.oakink_loader import create_oakink_loaders
from torch.utils.data.distributed import DistributedSampler


def setup_fsdp():
    """Initialize FSDP process group."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_fsdp():
    """Clean up FSDP process group."""
    destroy_process_group()


def get_fsdp_config(cfg):
    """Get FSDP configuration based on config."""
    fsdp_cfg = cfg.get('fsdp', {})

    # Mixed precision settings
    if fsdp_cfg.get('mixed_precision', True):
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = None

    # Sharding strategy
    sharding_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
        'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_map.get(
        fsdp_cfg.get('sharding_strategy', 'FULL_SHARD'),
        ShardingStrategy.FULL_SHARD
    )

    # CPU offload
    cpu_offload = CPUOffload(offload_params=fsdp_cfg.get('cpu_offload', False))

    # Backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE if fsdp_cfg.get('backward_prefetch', True) else None

    return {
        'mixed_precision': mixed_precision,
        'sharding_strategy': sharding_strategy,
        'cpu_offload': cpu_offload,
        'backward_prefetch': backward_prefetch,
        'limit_all_gathers': fsdp_cfg.get('limit_all_gathers', True),
        'use_orig_params': True,  # Required for optimizer state dict
    }


def get_auto_wrap_policy(model, cfg):
    """Get auto-wrap policy for FSDP based on model architecture."""
    fsdp_cfg = cfg.get('fsdp', {})
    min_params = fsdp_cfg.get('min_params_to_wrap', 100_000_000)  # 100M params default

    # For Qwen-based model, wrap transformer blocks
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer

    # Define modules to wrap
    transformer_layer_cls = {
        Qwen2VLDecoderLayer,  # Qwen transformer layers
    }

    # Also add custom transformer layers if any
    try:
        from models.fusion_transformer import TransformerBlock
        transformer_layer_cls.add(TransformerBlock)
    except:
        pass

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    # Fallback to size-based wrapping if needed
    if fsdp_cfg.get('use_size_based_wrap', False):
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_params,
        )

    return auto_wrap_policy


def wrap_model_for_fsdp(model, cfg, device_id):
    """Wrap model with FSDP."""
    fsdp_config = get_fsdp_config(cfg)
    auto_wrap_policy = get_auto_wrap_policy(model, cfg)

    # Special handling for frozen Qwen backbone
    if hasattr(model, 'sem_enc') and hasattr(model.sem_enc, 'qwen'):
        # If Qwen is frozen, we can wrap it separately for memory efficiency
        if model.sem_enc.freeze_qwen:
            # Wrap Qwen backbone separately (no gradients)
            model.sem_enc.qwen = FSDP(
                model.sem_enc.qwen,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=fsdp_config['mixed_precision'],
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # Full shard for frozen params
                cpu_offload=fsdp_config['cpu_offload'],
                device_id=device_id,
                use_orig_params=True,
            )

    # Wrap entire model
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **fsdp_config,
        device_id=device_id,
    )

    return model


def compute_flow_matching_loss(model, conditioning, target_pose, bone_length_reg=0.0):
    """
    Compute rectified flow matching loss (same as single GPU version).
    """
    B = target_pose.size(0)
    device = target_pose.device

    x0 = torch.randn_like(target_pose)
    t = torch.rand(B, device=device)

    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * target_pose

    v_pred = model.flow_step(x_t, t, conditioning)

    v_target = target_pose - x0

    loss = F.mse_loss(v_pred, v_target)

    if bone_length_reg > 0 and target_pose.shape[1] == 63:
        joints_pred = (x_t + t.unsqueeze(-1) * v_pred).reshape(B, 21, 3)

        bone_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        expected_lengths = torch.tensor([
            0.050, 0.035, 0.025, 0.020,
            0.080, 0.045, 0.030, 0.020,
            0.085, 0.050, 0.035, 0.025,
            0.080, 0.045, 0.030, 0.025,
            0.070, 0.035, 0.025, 0.020
        ], device=device)

        bone_loss = 0.0
        for i, (parent, child) in enumerate(bone_connections):
            bone_vec = joints_pred[:, child] - joints_pred[:, parent]
            bone_len = torch.norm(bone_vec, dim=1)
            bone_loss += F.mse_loss(bone_len, expected_lengths[i].expand(B))

        loss = loss + bone_length_reg * bone_loss / len(bone_connections)

    return loss


def train_one_epoch_fsdp(model, loader, optimizer, cfg, rank, writer, global_step):
    """
    Train for one epoch with FSDP.
    """
    model.train()

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc="Training")
    else:
        pbar = loader

    grad_accum_steps = max(1, cfg['training'].get('gradient_accumulation', 1))
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Data already on correct device from DistributedSampler
        pts = batch["points"].cuda()
        contact_labels = batch["contact_labels"].cuda()
        y_pose = batch["pose"].cuda()

        # Forward pass
        out = model.forward_train(
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
            # Gradient clipping (FSDP compatible)
            if cfg['training']['gradient_clip'] > 0:
                model.clip_grad_norm_(cfg['training']['gradient_clip'])

            optimizer.step()
            optimizer.zero_grad()

        # Logging (only on rank 0)
        if rank == 0 and batch_idx % cfg['training']['log_interval'] == 0:
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

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'contact': f'{loss_contact.item():.4f}',
                    'flow': f'{loss_flow.item():.4f}',
                    'acc': f'{contact_acc.item():.3f}'
                })

        # Checkpoint saving (only on rank 0)
        if rank == 0 and global_step > 0 and global_step % cfg['training']['checkpoint_interval'] == 0:
            save_checkpoint_fsdp(model, optimizer, global_step, cfg, rank)

        global_step += 1

    return global_step


def validate_fsdp(model, loader, cfg, rank, writer, global_step):
    """
    Validate the model with FSDP (only on rank 0 to avoid duplication).
    """
    if rank != 0:
        return {}

    model.eval()

    total_loss = 0
    total_contact_loss = 0
    total_flow_loss = 0
    total_contact_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pts = batch["points"].cuda()
            contact_labels = batch["contact_labels"].cuda()
            y_pose = batch["pose"].cuda()

            out = model.forward_train(
                images_list=batch["images_list"],
                texts_list=batch["texts_list"],
                pts=pts
            )

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

            pred_contacts = torch.sigmoid(logits_c.squeeze(-1)) > 0.5
            contact_acc = (pred_contacts == contact_labels).float().mean()

            total_loss += loss.item()
            total_contact_loss += loss_contact.item()
            total_flow_loss += loss_flow.item()
            total_contact_acc += contact_acc.item()
            num_batches += 1

    metrics = {
        'val_loss': total_loss / num_batches,
        'val_contact_loss': total_contact_loss / num_batches,
        'val_flow_loss': total_flow_loss / num_batches,
        'val_contact_acc': total_contact_acc / num_batches,
    }

    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'Validation/{key}', value, global_step)

    return metrics


def save_checkpoint_fsdp(model, optimizer, step, cfg, rank):
    """Save FSDP checkpoint (only on rank 0)."""
    if rank != 0:
        return

    # Configure state dict settings
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy
    ):
        model_state = model.state_dict()
        optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if rank == 0:
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'step': step,
            'config': cfg,
        }

        path = os.path.join(
            cfg['paths']['checkpoint_dir'],
            f'checkpoint_fsdp_step_{step}.pt'
        )
        torch.save(checkpoint, path)
        print(f"Saved FSDP checkpoint to {path}")


def load_checkpoint_fsdp(model, optimizer, checkpoint_path, device_id):
    """Load FSDP checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{device_id}')

    # Configure state dict settings
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        load_policy
    ):
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, checkpoint['optimizer_state_dict']
        )
        optimizer.load_state_dict(optim_state)

    return checkpoint['step']


def main():
    parser = argparse.ArgumentParser(description="Train functional grasp model with FSDP")
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to OakInk dataset')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (uses config_fsdp.py if exists)')

    args = parser.parse_args()

    # Setup FSDP
    setup_fsdp()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Get configuration
    if args.config and os.path.exists(args.config):
        # Load custom config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = config_module.Config.get_config(mode='train')
    else:
        # Try to load FSDP config if exists
        if os.path.exists('config_fsdp.py'):
            from config_fsdp import Config
        else:
            from config import Config
        cfg = Config.get_config(mode='train')

    # Override with command line arguments
    if args.data_path:
        cfg['data']['root_dir'] = args.data_path
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['epochs'] = args.epochs

    # Adjust batch size for world size if needed
    if cfg.get('fsdp', {}).get('scale_batch_size', True):
        effective_batch_size = cfg['training']['batch_size'] * world_size
        if rank == 0:
            print(f"Effective batch size: {effective_batch_size} ({cfg['training']['batch_size']} per GPU x {world_size} GPUs)")

    # Create directories (only on rank 0)
    if rank == 0:
        Config.create_dirs()

    # Device
    device_id = local_rank
    torch.cuda.set_device(device_id)

    if rank == 0:
        print(f"Using {world_size} GPUs for training")

    # Data loaders with DistributedSampler
    if rank == 0:
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
        world_size=world_size,
        rank=rank
    )

    # Model
    if rank == 0:
        print("Initializing model...")

    model = FunctionalGraspModel(
        CSEM=cfg['model']['CSEM'],
        CGEO=cfg['model']['CGEO'],
        DPOSE=cfg['model']['DPOSE'],
        K_CONTACT=cfg['model']['K_CONTACT']
    ).cuda()

    # Wrap with FSDP
    model = wrap_model_for_fsdp(model, cfg, device_id)

    # Count parameters (only on rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # Load checkpoint if provided
    global_step = 0
    start_epoch = 0
    if args.checkpoint:
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint}")
        global_step = load_checkpoint_fsdp(model, optimizer, args.checkpoint, device_id)
        start_epoch = global_step // len(train_loader)

    # TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0 and HAS_TENSORBOARD:
        writer = SummaryWriter(cfg['paths']['log_dir'] + '_fsdp')

    # Training loop
    if rank == 0:
        print("Starting FSDP training...")

    for epoch in range(start_epoch, cfg['training']['epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        # Set epoch for DistributedSampler
        train_loader.sampler.set_epoch(epoch)

        # Train
        global_step = train_one_epoch_fsdp(
            model, train_loader, optimizer, cfg, rank, writer, global_step
        )

        # Validate (only on rank 0)
        if (epoch + 1) % 5 == 0:
            if rank == 0:
                print("Running validation...")
            metrics = validate_fsdp(model, val_loader, cfg, rank, writer, global_step)
            if rank == 0:
                print(f"Validation metrics: {metrics}")

        # Save checkpoint at end of epoch (only on rank 0)
        if rank == 0:
            save_checkpoint_fsdp(model, optimizer, global_step, cfg, rank)

    if writer is not None:
        writer.close()

    if rank == 0:
        print("FSDP training complete!")

    # Cleanup
    cleanup_fsdp()


if __name__ == "__main__":
    main()