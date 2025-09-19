"""
FSDP utility functions for distributed training.
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy,
    _or_policy,
)
from functools import partial
import logging


def setup_logger(rank):
    """Setup logger for distributed training."""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def is_distributed():
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """Get world size in distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if current process is the main process."""
    return get_rank() == 0


def print_model_size(model, name="Model"):
    """Print model size and parameter count."""
    if not is_main_process():
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

    print(f"\n{name} Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_size:.2f} GB")


def get_fsdp_auto_wrap_policy(model_type="qwen", min_params=100_000_000):
    """
    Get auto-wrap policy based on model type.

    Args:
        model_type: Type of model ('qwen', 'custom', 'size_based')
        min_params: Minimum parameters for size-based wrapping

    Returns:
        Auto-wrap policy function
    """
    if model_type == "qwen":
        # Import Qwen transformer layers
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
            transformer_cls = {Qwen2VLDecoderLayer}
        except ImportError:
            print("Warning: Could not import Qwen2VL layers, using size-based policy")
            return partial(size_based_auto_wrap_policy, min_num_params=min_params)

        # Add custom transformer layers
        try:
            from models.fusion_transformer import TransformerBlock
            transformer_cls.add(TransformerBlock)
        except ImportError:
            pass

        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls
        )

    elif model_type == "size_based":
        return partial(size_based_auto_wrap_policy, min_num_params=min_params)

    else:
        # Custom lambda policy for specific modules
        def lambda_policy_fn(module):
            # Wrap large modules
            if sum(p.numel() for p in module.parameters()) > min_params:
                return True
            # Wrap specific module types
            if module.__class__.__name__ in ['PointNet2Encoder', 'FlowMatching', 'FusionTransformer']:
                return True
            return False

        return partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)


def create_mixed_precision_policy(dtype="bfloat16"):
    """
    Create mixed precision policy for FSDP.

    Args:
        dtype: Data type for mixed precision ('bfloat16', 'float16', 'float32')

    Returns:
        MixedPrecision policy or None
    """
    if dtype == "float32":
        return None

    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return MixedPrecision(
        param_dtype=dtype_map[dtype],
        reduce_dtype=dtype_map[dtype],
        buffer_dtype=dtype_map[dtype],
    )


def save_fsdp_checkpoint(model, optimizer, epoch, step, save_dir, rank=0):
    """
    Save FSDP model and optimizer checkpoint.

    Args:
        model: FSDP wrapped model
        optimizer: Optimizer
        epoch: Current epoch
        step: Current step
        save_dir: Directory to save checkpoint
        rank: Process rank
    """
    if rank != 0:
        return

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Configure state dict for saving
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
        optim_state_dict = FSDP.full_optim_state_dict(model, optimizer)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': cpu_state_dict,
        'optimizer_state_dict': optim_state_dict,
    }

    save_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}_step{step}.pt')
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

    # Save latest checkpoint link
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(save_path), latest_path)


def load_fsdp_checkpoint(model, optimizer, checkpoint_path, device_id=0):
    """
    Load FSDP checkpoint.

    Args:
        model: FSDP wrapped model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        device_id: CUDA device id

    Returns:
        epoch, step from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{device_id}')

    # Configure state dict for loading
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, checkpoint['optimizer_state_dict']
        )
        optimizer.load_state_dict(optim_state)

    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)


def reduce_tensor(tensor, world_size):
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes

    Returns:
        Reduced tensor
    """
    if world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor, world_size):
    """
    Gather tensors from all processes.

    Args:
        tensor: Local tensor
        world_size: Number of processes

    Returns:
        List of tensors from all processes
    """
    if world_size == 1:
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


class FSDPMemoryTracker:
    """Track memory usage during FSDP training."""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.peak_memory = 0
        self.current_memory = 0

    def update(self):
        """Update memory statistics."""
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated(self.device_id) / 1024**3
            self.peak_memory = max(self.peak_memory, self.current_memory)

    def report(self, prefix=""):
        """Report memory usage."""
        if is_main_process():
            print(f"{prefix} Memory: {self.current_memory:.2f} GB (Peak: {self.peak_memory:.2f} GB)")

    def reset_peak(self):
        """Reset peak memory."""
        self.peak_memory = self.current_memory


def estimate_model_memory(model, batch_size, seq_len=None):
    """
    Estimate memory requirements for model.

    Args:
        model: Model to estimate
        batch_size: Batch size
        seq_len: Sequence length (for transformer models)

    Returns:
        Dictionary with memory estimates
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    grad_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024**3

    # Rough estimates for activations and optimizer states
    activation_memory = param_memory * batch_size * 0.5  # Very rough estimate
    optimizer_memory = grad_memory * 2  # Adam has 2 momentum terms

    total_memory = param_memory + grad_memory + activation_memory + optimizer_memory

    return {
        'param_memory_gb': param_memory,
        'grad_memory_gb': grad_memory,
        'activation_memory_gb': activation_memory,
        'optimizer_memory_gb': optimizer_memory,
        'total_memory_gb': total_memory,
    }


def get_fsdp_stats(model):
    """
    Get FSDP statistics for debugging.

    Args:
        model: FSDP wrapped model

    Returns:
        Dictionary with FSDP stats
    """
    stats = {}

    if isinstance(model, FSDP):
        # Get number of FSDP units
        num_fsdp_units = 0
        for module in model.modules():
            if isinstance(module, FSDP):
                num_fsdp_units += 1

        stats['num_fsdp_units'] = num_fsdp_units
        stats['sharding_strategy'] = str(model.sharding_strategy)

    return stats


def synchronize_processes():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()