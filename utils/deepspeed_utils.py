"""
DeepSpeed utility functions for distributed training.
"""

import os
import json
import torch
import deepspeed
from deepspeed import comm as dist
from typing import Dict, Optional, Any
import psutil
import GPUtil


def get_memory_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        gpu_memory = []
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_memory.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': f"{gpu.memoryUsed:.1f}MB",
                'memory_total': f"{gpu.memoryTotal:.1f}MB",
                'memory_percent': f"{gpu.memoryUtil*100:.1f}%"
            })
    else:
        gpu_memory = None

    cpu_memory = {
        'percent': psutil.virtual_memory().percent,
        'used': f"{psutil.virtual_memory().used / 1024**3:.1f}GB",
        'total': f"{psutil.virtual_memory().total / 1024**3:.1f}GB"
    }

    return {
        'gpu': gpu_memory,
        'cpu': cpu_memory
    }


def estimate_model_memory(model, batch_size, seq_len=1024):
    """
    Estimate memory requirements for model with DeepSpeed.

    Args:
        model: PyTorch model
        batch_size: Batch size per GPU
        seq_len: Sequence length for transformer models

    Returns:
        Dictionary with memory estimates
    """
    num_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = num_params * 4 / 1024**3  # FP32

    # Estimate based on ZeRO stage
    zero_stages = {
        0: {  # DDP
            'param_memory': param_memory_gb,
            'grad_memory': param_memory_gb,
            'optimizer_memory': param_memory_gb * 2,  # Adam has 2 states
        },
        1: {  # ZeRO-1: Shard optimizer states
            'param_memory': param_memory_gb,
            'grad_memory': param_memory_gb,
            'optimizer_memory': param_memory_gb * 2 / dist.get_world_size() if dist.is_initialized() else param_memory_gb * 2,
        },
        2: {  # ZeRO-2: Shard optimizer states + gradients
            'param_memory': param_memory_gb,
            'grad_memory': param_memory_gb / dist.get_world_size() if dist.is_initialized() else param_memory_gb,
            'optimizer_memory': param_memory_gb * 2 / dist.get_world_size() if dist.is_initialized() else param_memory_gb * 2,
        },
        3: {  # ZeRO-3: Shard everything
            'param_memory': param_memory_gb / dist.get_world_size() if dist.is_initialized() else param_memory_gb,
            'grad_memory': param_memory_gb / dist.get_world_size() if dist.is_initialized() else param_memory_gb,
            'optimizer_memory': param_memory_gb * 2 / dist.get_world_size() if dist.is_initialized() else param_memory_gb * 2,
        }
    }

    # Rough activation memory estimate
    activation_memory_gb = param_memory_gb * batch_size * 0.5

    return {
        'num_parameters': num_params,
        'param_memory_gb': param_memory_gb,
        'zero_stages': zero_stages,
        'activation_memory_gb': activation_memory_gb,
        'total_memory_gb': {
            stage: sum(stage_mem.values()) + activation_memory_gb
            for stage, stage_mem in zero_stages.items()
        }
    }


def create_deepspeed_config(
    base_config_path: str,
    output_path: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a DeepSpeed configuration with overrides.

    Args:
        base_config_path: Path to base configuration JSON
        output_path: Path to save modified configuration
        overrides: Dictionary of values to override

    Returns:
        Modified configuration dictionary
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    if overrides:
        # Apply overrides recursively
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config = update_dict(config, overrides)

    # Save modified config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config


def get_optimal_deepspeed_config(
    model_size_gb: float,
    gpu_memory_gb: float,
    num_gpus: int,
    batch_size: int
) -> str:
    """
    Recommend optimal DeepSpeed configuration based on hardware.

    Args:
        model_size_gb: Model size in GB
        gpu_memory_gb: GPU memory per device in GB
        num_gpus: Number of GPUs
        batch_size: Desired batch size per GPU

    Returns:
        Recommended config file name
    """
    total_gpu_memory = gpu_memory_gb * num_gpus

    # Estimate memory requirements
    param_memory = model_size_gb
    grad_memory = model_size_gb
    optimizer_memory = model_size_gb * 2  # Adam
    activation_memory = model_size_gb * batch_size * 0.5

    total_required = param_memory + grad_memory + optimizer_memory + activation_memory

    # Recommend based on memory constraints
    if total_required <= gpu_memory_gb * 0.8:
        # Can fit on single GPU with headroom
        return "ds_config_zero2.json"
    elif total_required <= total_gpu_memory * 0.8:
        # Need to shard across GPUs
        return "ds_config_zero3.json"
    else:
        # Need CPU offloading
        return "ds_config_zero3_offload.json"


def setup_deepspeed_logger(rank: int = 0):
    """Setup logging for DeepSpeed training."""
    import logging

    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger(__name__)


def print_model_summary(model, rank: int = 0):
    """Print model summary (only on rank 0)."""
    if rank != 0:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print("Model Summary")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
    print(f"Model size: {total_params * 2 / 1024**3:.2f} GB (FP16/BF16)")
    print("="*50 + "\n")


def convert_checkpoint_to_deepspeed(
    pytorch_checkpoint_path: str,
    output_dir: str,
    num_gpus: int = 1
):
    """
    Convert PyTorch checkpoint to DeepSpeed format.

    Args:
        pytorch_checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save DeepSpeed checkpoint
        num_gpus: Number of GPUs for sharding
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Create DeepSpeed checkpoint structure
    ds_checkpoint = {
        'module': state_dict,
        'optimizer': None,  # Will be initialized by DeepSpeed
        'lr_scheduler': None,  # Will be initialized by DeepSpeed
        'sparse_tensor_module_names': [],
        'version': 0.1
    }

    # Save in DeepSpeed format
    torch.save(ds_checkpoint, os.path.join(output_dir, 'mp_rank_00_model_states.pt'))

    print(f"Converted checkpoint saved to {output_dir}")


def convert_deepspeed_to_pytorch(
    deepspeed_checkpoint_dir: str,
    output_path: str
):
    """
    Convert DeepSpeed checkpoint back to PyTorch format.

    Args:
        deepspeed_checkpoint_dir: Directory with DeepSpeed checkpoint
        output_path: Path to save PyTorch checkpoint
    """
    # Find model state file
    model_file = os.path.join(deepspeed_checkpoint_dir, 'mp_rank_00_model_states.pt')

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model state file not found: {model_file}")

    # Load DeepSpeed checkpoint
    ds_checkpoint = torch.load(model_file, map_location='cpu')

    # Extract model state dict
    if 'module' in ds_checkpoint:
        state_dict = ds_checkpoint['module']
    else:
        state_dict = ds_checkpoint

    # Save as PyTorch checkpoint
    torch.save({
        'model_state_dict': state_dict,
        'checkpoint_type': 'pytorch',
        'converted_from': 'deepspeed'
    }, output_path)

    print(f"Converted to PyTorch checkpoint: {output_path}")


def profile_communication(model_engine, num_iterations: int = 10):
    """
    Profile communication overhead in distributed training.

    Args:
        model_engine: DeepSpeed model engine
        num_iterations: Number of iterations to profile

    Returns:
        Communication statistics
    """
    if not dist.is_initialized():
        return None

    import time

    comm_times = []
    compute_times = []

    for _ in range(num_iterations):
        # Dummy forward pass
        start_compute = time.time()
        dummy_input = torch.randn(1, 512).cuda()
        output = model_engine(dummy_input)
        loss = output.mean()

        # Backward pass (includes communication)
        start_comm = time.time()
        model_engine.backward(loss)
        end_comm = time.time()

        compute_times.append(start_comm - start_compute)
        comm_times.append(end_comm - start_comm)

        model_engine.step()

    avg_compute = sum(compute_times) / len(compute_times)
    avg_comm = sum(comm_times) / len(comm_times)

    return {
        'avg_compute_time': avg_compute,
        'avg_communication_time': avg_comm,
        'communication_overhead': avg_comm / (avg_compute + avg_comm) * 100
    }


def get_deepspeed_stats(model_engine):
    """
    Get DeepSpeed training statistics.

    Args:
        model_engine: DeepSpeed model engine

    Returns:
        Dictionary with training stats
    """
    stats = {}

    if hasattr(model_engine, 'optimizer'):
        stats['learning_rate'] = model_engine.get_lr()[0]

    if hasattr(model_engine, 'global_steps'):
        stats['global_steps'] = model_engine.global_steps

    if hasattr(model_engine, 'global_samples'):
        stats['global_samples'] = model_engine.global_samples

    if hasattr(model_engine, 'skipped_steps'):
        stats['skipped_steps'] = model_engine.skipped_steps

    if hasattr(model_engine, 'module'):
        stats['num_parameters'] = sum(p.numel() for p in model_engine.module.parameters())

    return stats


class DeepSpeedCheckpointManager:
    """Manage DeepSpeed checkpoints with automatic cleanup."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model_engine, step: int):
        """Save checkpoint and manage old checkpoints."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'deepspeed_ckpt_step_{step}'
        )

        model_engine.save_checkpoint(checkpoint_path)
        self.checkpoints.append((step, checkpoint_path))

        # Remove old checkpoints if exceeding limit
        if len(self.checkpoints) > self.max_checkpoints:
            old_step, old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                import shutil
                shutil.rmtree(old_path)
                if dist.get_rank() == 0:
                    print(f"Removed old checkpoint: {old_path}")

    def load_latest(self, model_engine):
        """Load the latest checkpoint."""
        if not self.checkpoints:
            # Search for existing checkpoints
            import glob
            pattern = os.path.join(self.checkpoint_dir, 'deepspeed_ckpt_step_*')
            existing = glob.glob(pattern)
            if existing:
                # Sort by step number
                existing.sort(key=lambda x: int(x.split('_')[-1]))
                self.checkpoints = [(int(p.split('_')[-1]), p) for p in existing]

        if self.checkpoints:
            latest_step, latest_path = self.checkpoints[-1]
            _, client_states = model_engine.load_checkpoint(latest_path)
            return latest_step
        return 0