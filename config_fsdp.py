"""
FSDP-specific configuration for distributed training.
Extends base config with FSDP parameters.
"""

import torch
import os
import copy
from config import Config as BaseConfig


class Config(BaseConfig):
    """FSDP configuration extending base config."""

    # FSDP-specific parameters
    FSDP = {
        # Sharding strategy
        'sharding_strategy': 'FULL_SHARD',  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD

        # Mixed precision
        'mixed_precision': True,  # Use bfloat16 for training
        'param_dtype': 'bfloat16',  # Parameter dtype (bfloat16/float16/float32)
        'reduce_dtype': 'bfloat16',  # Gradient reduction dtype
        'buffer_dtype': 'bfloat16',  # Buffer dtype

        # Memory optimization
        'cpu_offload': False,  # Offload parameters to CPU (for very large models)
        'backward_prefetch': True,  # Prefetch next layer's parameters during backward
        'limit_all_gathers': True,  # Limit concurrent all-gathers

        # Auto-wrap policy
        'min_params_to_wrap': 100_000_000,  # Min params for wrapping (100M default for Qwen blocks)
        'use_size_based_wrap': False,  # Use size-based wrapping instead of transformer-based

        # Checkpointing
        'activation_checkpointing': False,  # Gradient checkpointing for memory savings
        'checkpoint_modules': [],  # Specific modules to checkpoint

        # Performance
        'sync_module_states': True,  # Sync module states at init
        'forward_prefetch': True,  # Prefetch next FSDP unit's params

        # Scaling
        'scale_batch_size': False,  # Don't scale batch size with world size (user controls total batch)
        'gradient_accumulation_steps': 1,  # Gradient accumulation across ranks
    }

    # Override training parameters for multi-GPU
    TRAINING_FSDP = {
        'batch_size': 8,  # Per-GPU batch size
        'learning_rate': 2e-4,  # Scaled learning rate for larger effective batch
        'gradient_accumulation': 1,  # Less accumulation needed with multiple GPUs
        'warmup_steps': 500,  # Warmup for scaled learning rate
        'max_grad_norm': 1.0,  # Gradient clipping
    }

    # Distributed training settings
    DISTRIBUTED = {
        'backend': 'nccl',  # Communication backend
        'find_unused_parameters': False,  # For debugging unused params
        'broadcast_buffers': True,  # Broadcast buffers at init
        'bucket_cap_mb': 25,  # Gradient bucket size in MB
        'gradient_as_bucket_view': True,  # Memory optimization
    }

    # Multi-node settings
    MULTINODE = {
        'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
        'master_port': os.environ.get('MASTER_PORT', '29500'),
        'node_rank': int(os.environ.get('NODE_RANK', 0)),
        'num_nodes': int(os.environ.get('NUM_NODES', 1)),
    }

    @classmethod
    def get_config(cls, mode='train'):
        """
        Get FSDP configuration dictionary.

        Args:
            mode: 'train' or 'eval'

        Returns:
            config dict with FSDP settings
        """
        # Start with base config
        config = super().get_config(mode)

        # Add FSDP-specific settings
        config['fsdp'] = copy.deepcopy(cls.FSDP)
        config['distributed'] = copy.deepcopy(cls.DISTRIBUTED)
        config['multinode'] = copy.deepcopy(cls.MULTINODE)

        # Override training params for FSDP
        if torch.cuda.is_available():
            config['training'].update(cls.TRAINING_FSDP)

        # Adjust for world size if available
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size > 1:
            # Effective batch size = per_gpu_batch * world_size * gradient_accumulation
            effective_batch = config['training']['batch_size'] * world_size * config['training']['gradient_accumulation']

            # Scale learning rate with sqrt(batch_size) scaling
            # base_lr * sqrt(effective_batch / base_batch)
            base_batch = 4  # Base batch size from single GPU training
            lr_scale = (effective_batch / base_batch) ** 0.5
            config['training']['learning_rate'] *= lr_scale

            print(f"FSDP Config: World size={world_size}, "
                  f"Per-GPU batch={config['training']['batch_size']}, "
                  f"Effective batch={effective_batch}, "
                  f"LR={config['training']['learning_rate']:.2e}")

        return config

    @classmethod
    def get_sharding_strategy(cls, strategy_name):
        """
        Get PyTorch FSDP ShardingStrategy from string name.
        """
        from torch.distributed.fsdp import ShardingStrategy

        strategies = {
            'FULL_SHARD': ShardingStrategy.FULL_SHARD,
            'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
            'NO_SHARD': ShardingStrategy.NO_SHARD,
            'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
            '_HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
        }

        return strategies.get(strategy_name, ShardingStrategy.FULL_SHARD)

    @classmethod
    def get_mixed_precision_policy(cls, config):
        """
        Get PyTorch FSDP MixedPrecision policy from config.
        """
        from torch.distributed.fsdp import MixedPrecision

        if not config['fsdp']['mixed_precision']:
            return None

        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }

        return MixedPrecision(
            param_dtype=dtype_map[config['fsdp']['param_dtype']],
            reduce_dtype=dtype_map[config['fsdp']['reduce_dtype']],
            buffer_dtype=dtype_map[config['fsdp']['buffer_dtype']],
        )

    @classmethod
    def print_fsdp_config(cls, config):
        """Pretty print FSDP configuration."""
        print("\n" + "="*50)
        print("FSDP Configuration")
        print("="*50)

        print("\nSharding:")
        print(f"  Strategy: {config['fsdp']['sharding_strategy']}")
        print(f"  Min params to wrap: {config['fsdp']['min_params_to_wrap']:,}")

        print("\nMixed Precision:")
        print(f"  Enabled: {config['fsdp']['mixed_precision']}")
        if config['fsdp']['mixed_precision']:
            print(f"  Param dtype: {config['fsdp']['param_dtype']}")
            print(f"  Reduce dtype: {config['fsdp']['reduce_dtype']}")

        print("\nMemory Optimization:")
        print(f"  CPU offload: {config['fsdp']['cpu_offload']}")
        print(f"  Backward prefetch: {config['fsdp']['backward_prefetch']}")
        print(f"  Activation checkpointing: {config['fsdp']['activation_checkpointing']}")

        print("\nTraining:")
        print(f"  Per-GPU batch size: {config['training']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']:.2e}")
        print(f"  Gradient accumulation: {config['training']['gradient_accumulation']}")

        print("="*50 + "\n")