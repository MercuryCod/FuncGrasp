# DeepSpeed Training Guide for FuncGrasp

## Overview

This guide covers distributed training of the FuncGrasp model using Microsoft DeepSpeed. DeepSpeed enables efficient training of large models through ZeRO (Zero Redundancy Optimizer) technology, which partitions model states across GPUs to reduce memory footprint.

## Key Features

- **ZeRO Optimization**: Three stages of memory optimization
- **Mixed Precision**: BF16 training for faster computation
- **CPU/NVMe Offloading**: Train models larger than GPU memory
- **Gradient Accumulation**: Simulate larger batch sizes
- **Distributed Training**: Scale across multiple GPUs and nodes

## Quick Start

### Installation

```bash
# Install DeepSpeed
pip install deepspeed

# Verify installation
python test_deepspeed.py
```

### Single-Node Multi-GPU Training

```bash
# Using the launch script (recommended)
bash scripts/run_deepspeed.sh --gpus 8 --data_path /path/to/OakInk

# Or using DeepSpeed directly
deepspeed --num_gpus=8 train_deepspeed.py \
    --data_path /path/to/OakInk \
    --batch_size 4 \
    --deepspeed_config ds_config_zero2.json
```

### Multi-Node Training

```bash
# Create hostfile
echo 'node1 slots=8' > hostfile
echo 'node2 slots=8' >> hostfile

# Launch on all nodes
bash scripts/run_deepspeed.sh \
    --num_nodes 2 \
    --hostfile hostfile \
    --master_addr node1 \
    --gpus 8
```

## Configuration Files

### ZeRO Stage Selection

Choose the appropriate configuration based on your hardware:

| Config File | ZeRO Stage | Memory Savings | Use Case |
|------------|------------|----------------|----------|
| `ds_config_zero2.json` | Stage 2 | 4x | Default, balanced performance |
| `ds_config_zero3.json` | Stage 3 | 8x | Large models, limited GPU memory |
| `ds_config_zero3_offload.json` | Stage 3 + Offload | 16x+ | Very large models, CPU offloading |

### Quick Selection Guide

```bash
# Standard training (GPU memory > 16GB)
bash scripts/run_deepspeed.sh --zero 2

# Memory-constrained (GPU memory 8-16GB)
bash scripts/run_deepspeed.sh --zero 3

# Very limited memory (GPU memory < 8GB)
bash scripts/run_deepspeed.sh --zero 3-offload
```

## Memory Optimization

### Memory Requirements by Configuration

For Qwen2.5-VL-3B model (3B parameters):

| Configuration | Memory/GPU (8 GPUs) | Notes |
|--------------|-------------------|-------|
| DDP (baseline) | ~14 GB | No optimization |
| ZeRO-2 | ~7 GB | Good balance |
| ZeRO-3 | ~3.5 GB | Maximum GPU efficiency |
| ZeRO-3 + Offload | ~1 GB | Extreme memory savings |

### Batch Size Recommendations

```python
# Recommended batch sizes per GPU
configs = {
    "16GB GPU": {
        "zero2": 4-8,
        "zero3": 8-16,
        "zero3_offload": 16-32
    },
    "24GB GPU": {
        "zero2": 8-16,
        "zero3": 16-32,
        "zero3_offload": 32-64
    },
    "40GB GPU": {
        "zero2": 16-32,
        "zero3": 32-64,
        "zero3_offload": 64-128
    }
}
```

## Training Commands

### Basic Training

```bash
# Train with default settings
bash scripts/run_deepspeed.sh

# Specify batch size and epochs
bash scripts/run_deepspeed.sh \
    --batch_size 8 \
    --epochs 100 \
    --gpus 4
```

### Resume from Checkpoint

```bash
# DeepSpeed checkpoints are saved in directories
bash scripts/run_deepspeed.sh \
    --checkpoint checkpoints/deepspeed_ckpt_step_5000 \
    --gpus 8
```

### Custom Configuration

```bash
# Use custom DeepSpeed config
deepspeed train_deepspeed.py \
    --data_path /path/to/data \
    --deepspeed_config my_config.json \
    --batch_size 16
```

## Performance Tuning

### 1. Communication Optimization

Optimize all-reduce operations:

```json
{
  "zero_optimization": {
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true
  }
}
```

### 2. Gradient Accumulation

Simulate larger batches:

```json
{
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2
}
```

Effective batch = 2 × 4 × num_gpus

### 3. Mixed Precision

Use BF16 for stability:

```json
{
  "bf16": {
    "enabled": true
  }
}
```

### 4. CPU Offloading

For very large models:

```json
{
  "zero_optimization": {
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

## Monitoring Training

### TensorBoard

```bash
# Logs are saved to ./logs/deepspeed
tensorboard --logdir ./logs/deepspeed
```

### Key Metrics

- **Loss convergence**: Should match single-GPU training
- **Throughput**: Samples/second across all GPUs
- **Memory usage**: Monitor with `nvidia-smi`
- **Communication time**: Check overlap efficiency

### DeepSpeed Profiling

Enable profiling in config:

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 100,
    "module_depth": -1,
    "top_modules": 3
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```bash
# Solution 1: Use higher ZeRO stage
bash scripts/run_deepspeed.sh --zero 3

# Solution 2: Reduce batch size
bash scripts/run_deepspeed.sh --batch_size 2

# Solution 3: Enable CPU offloading
bash scripts/run_deepspeed.sh --zero 3-offload
```

#### 2. Slow Training

```bash
# Check communication overhead
export NCCL_DEBUG=INFO

# Use faster interconnect
export NCCL_IB_DISABLE=0  # Enable InfiniBand

# Optimize bucket sizes
# Edit ds_config.json to increase bucket sizes
```

#### 3. Convergence Issues

```bash
# Adjust learning rate for distributed training
# LR should scale with effective batch size
# effective_batch = batch_size * gradient_accumulation * num_gpus
```

#### 4. Checkpoint Issues

```bash
# Convert PyTorch checkpoint to DeepSpeed format
python -c "
from utils.deepspeed_utils import convert_checkpoint_to_deepspeed
convert_checkpoint_to_deepspeed('model.pt', 'deepspeed_ckpt', num_gpus=8)
"
```

### Debugging

Enable verbose logging:

```bash
# Set environment variables
export DEEPSPEED_LOG_LEVEL=DEBUG
export TORCH_DISTRIBUTED_DEBUG=INFO

# Run with debugging
bash scripts/run_deepspeed.sh --gpus 1
```

## Advanced Usage

### Custom Model Modifications

```python
# In train_deepspeed.py, modify model initialization
model = FunctionalGraspModel(...)

# Freeze Qwen backbone to save memory
for param in model.sem.backbone.parameters():
    param.requires_grad = False

# Only optimizer will track trainable params
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
    config=ds_config
)
```

### Dynamic Batch Size

```python
# Automatically adjust batch size based on memory
from deepspeed.runtime.pipe.module import PipelineModule
model = PipelineModule(layers=model_layers,
                       num_stages=num_gpus,
                       partition_method='parameters')
```

### Multi-Task Training

```python
# Train multiple objectives
losses = {
    'contact': contact_loss * lambda_contact,
    'flow': flow_loss * lambda_flow,
    'regularization': reg_loss * lambda_reg
}
total_loss = sum(losses.values())
model_engine.backward(total_loss)
```

## Performance Benchmarks

Expected training performance:

| Configuration | GPUs | Batch/GPU | Throughput | Time/Epoch |
|--------------|------|-----------|------------|------------|
| Single GPU | 1 | 4 | 8 samples/s | 20 min |
| ZeRO-2 | 8 | 4 | 50 samples/s | 3.3 min |
| ZeRO-3 | 8 | 8 | 80 samples/s | 2.1 min |
| ZeRO-3 + Offload | 8 | 16 | 60 samples/s | 2.8 min |

## Best Practices

1. **Start Simple**: Begin with ZeRO-2 and standard batch size
2. **Profile First**: Use `test_deepspeed.py` to estimate memory
3. **Scale Gradually**: Increase GPUs/batch size incrementally
4. **Monitor Metrics**: Watch for communication bottlenecks
5. **Save Checkpoints**: Use DeepSpeed's checkpoint manager
6. **Version Control**: Track config files with experiments

## Migration from Single GPU

```bash
# Step 1: Test current single-GPU setup
python train.py --epochs 1

# Step 2: Test DeepSpeed with 1 GPU
deepspeed --num_gpus=1 train_deepspeed.py --epochs 1

# Step 3: Scale to multiple GPUs
bash scripts/run_deepspeed.sh --gpus 8 --epochs 1

# Step 4: Optimize configuration
# Adjust batch size, learning rate, and ZeRO stage
```

## Files Reference

- `train_deepspeed.py` - Main training script with DeepSpeed integration
- `ds_config_*.json` - DeepSpeed configuration files
- `utils/deepspeed_utils.py` - Helper utilities
- `scripts/run_deepspeed.sh` - Launch script
- `test_deepspeed.py` - Setup verification

## Next Steps

1. Run `python test_deepspeed.py` to verify setup
2. Start with single-GPU DeepSpeed training
3. Scale to multi-GPU with appropriate ZeRO stage
4. Optimize batch size and learning rate
5. Monitor and tune performance

For production training, always test configurations on a small subset first before running full training.