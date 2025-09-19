# FSDP Training Guide for FuncGrasp

## Overview

This guide covers the Fully Sharded Data Parallel (FSDP) implementation for training the FuncGrasp model across multiple GPUs and nodes. FSDP enables efficient training of large models by sharding model parameters, gradients, and optimizer states across GPUs.

## Key Features

- **Memory Efficiency**: Full parameter sharding reduces memory footprint per GPU
- **Mixed Precision**: BFloat16 training for faster computation and reduced memory
- **Automatic Wrapping**: Smart module wrapping for optimal performance
- **Multi-Node Support**: Scale across multiple machines
- **Checkpointing**: FSDP-aware checkpoint saving/loading

## Quick Start

### Single-Node Multi-GPU Training

```bash
# Using the launch script (recommended)
bash scripts/run_fsdp.sh --gpus 8 --data_path /path/to/OakInk

# Or using torchrun directly
torchrun --standalone --nproc_per_node=8 train_fsdp.py \
    --data_path /path/to/OakInk \
    --batch_size 8 \
    --epochs 100
```

### Multi-Node Training

On each node, run:

```bash
# Node 0 (master)
bash scripts/run_fsdp.sh \
    --num_nodes 2 \
    --node_rank 0 \
    --master_addr 192.168.1.100 \
    --gpus 8

# Node 1
bash scripts/run_fsdp.sh \
    --num_nodes 2 \
    --node_rank 1 \
    --master_addr 192.168.1.100 \
    --gpus 8
```

## Configuration

### FSDP-Specific Config (`config_fsdp.py`)

Key parameters:

```python
FSDP = {
    'sharding_strategy': 'FULL_SHARD',  # Maximum memory efficiency
    'mixed_precision': True,             # BFloat16 training
    'cpu_offload': False,               # Enable for very large models
    'min_params_to_wrap': 100_000_000,  # Wrap modules > 100M params
}
```

### Sharding Strategies

- **FULL_SHARD**: Shard parameters, gradients, and optimizer states (most memory efficient)
- **SHARD_GRAD_OP**: Only shard gradients and optimizer states
- **NO_SHARD**: Similar to DDP, no sharding
- **HYBRID_SHARD**: Shard within node, replicate across nodes

## Architecture Changes for FSDP

### Model Modifications

The `FunctionalGraspModel` includes FSDP-specific methods:

```python
# Configure model for FSDP
model.configure_for_fsdp()

# Get modules for separate wrapping
wrap_modules = model.get_fsdp_wrap_params()
```

### Auto-Wrapping Policy

The model uses transformer-based auto-wrapping for optimal performance:

1. **Qwen2.5-VL layers**: Wrapped individually (largest components)
2. **PointNet++ encoder**: Wrapped as a unit
3. **Fusion transformer**: Each transformer block wrapped
4. **Flow matching module**: Wrapped as a unit

### Data Loading

DistributedSampler ensures each GPU processes unique data:

```python
train_loader, val_loader = create_oakink_loaders(
    root_dir=data_path,
    distributed=True,
    world_size=world_size,
    rank=rank
)
```

## Memory Optimization Tips

### 1. Gradient Accumulation

For larger effective batch sizes with limited memory:

```python
# In config_fsdp.py
TRAINING_FSDP = {
    'batch_size': 4,  # Per-GPU batch
    'gradient_accumulation': 2,  # Effective batch = 4 * 2 * num_gpus
}
```

### 2. CPU Offloading

For extremely large models:

```python
FSDP = {
    'cpu_offload': True,  # Offload parameters to CPU
}
```

### 3. Activation Checkpointing

Save memory by recomputing activations:

```python
FSDP = {
    'activation_checkpointing': True,
    'checkpoint_modules': ['TransformerBlock'],  # Specific modules
}
```

## Performance Tuning

### 1. Optimal Batch Size

- Start with batch_size = 8 per GPU
- Scale learning rate: `lr = base_lr * sqrt(total_batch_size / base_batch)`
- Monitor GPU memory usage and adjust

### 2. Communication Optimization

```python
DISTRIBUTED = {
    'bucket_cap_mb': 25,  # Gradient bucket size
    'gradient_as_bucket_view': True,  # Memory optimization
}
```

### 3. Prefetching

```python
FSDP = {
    'backward_prefetch': True,  # Prefetch next layer during backward
    'forward_prefetch': True,   # Prefetch during forward
}
```

## Monitoring Training

### TensorBoard

```bash
# Training logs saved to logs_fsdp/
tensorboard --logdir logs_fsdp
```

### Key Metrics

- **Loss convergence**: Should match single-GPU training
- **GPU memory**: Monitor with `nvidia-smi`
- **Communication overhead**: Check with PyTorch profiler

## Checkpointing

### Saving Checkpoints

FSDP checkpoints are saved with full state dict on rank 0:

```python
# Automatically saved every N steps
checkpoint_interval: 500
```

### Resuming Training

```bash
# Resume from checkpoint
bash scripts/run_fsdp.sh \
    --checkpoint checkpoints/checkpoint_fsdp_step_1000.pt \
    --gpus 8
```

## Troubleshooting

### Common Issues

1. **OOM Errors**
   - Reduce batch size
   - Enable CPU offloading
   - Use SHARD_GRAD_OP strategy

2. **Slow Training**
   - Check network bandwidth between nodes
   - Ensure NCCL is properly configured
   - Use faster interconnect (InfiniBand)

3. **Convergence Issues**
   - Verify learning rate scaling
   - Check gradient clipping
   - Monitor gradient norms

### Debugging

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO

# Check FSDP wrapping
python test_fsdp.py  # Run on single machine
```

## Benchmarks

Expected training performance (approximate):

| Configuration | GPUs | Batch/GPU | Throughput | Memory/GPU |
|--------------|------|-----------|------------|------------|
| Single GPU   | 1    | 4         | 10 samples/s | 12 GB |
| Single Node  | 8    | 8         | 60 samples/s | 8 GB  |
| Multi Node   | 16   | 8         | 110 samples/s | 8 GB |

## Advanced Usage

### Custom Wrapping Policy

```python
# In utils/fsdp_utils.py
def custom_auto_wrap_policy(module):
    if isinstance(module, MyCustomLayer):
        return True
    if sum(p.numel() for p in module.parameters()) > threshold:
        return True
    return False
```

### Mixed Precision Options

```python
# Float16 (faster but less stable)
'param_dtype': 'float16'

# BFloat16 (recommended, better stability)
'param_dtype': 'bfloat16'

# Full precision (slower, more memory)
'mixed_precision': False
```

## Files Reference

- `train_fsdp.py`: Main FSDP training script
- `config_fsdp.py`: FSDP-specific configuration
- `utils/fsdp_utils.py`: Helper functions for FSDP
- `scripts/run_fsdp.sh`: Launch script for multi-GPU/node
- `test_fsdp.py`: Local testing without GPUs

## Next Steps

1. Test on single GPU first
2. Scale to single-node multi-GPU
3. Optimize batch size and learning rate
4. Scale to multi-node if needed

For production training, monitor metrics closely and adjust configuration based on your hardware capabilities.