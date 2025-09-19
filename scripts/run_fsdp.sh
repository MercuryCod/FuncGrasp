#!/bin/bash
# FSDP training launch script for FuncGrasp
# Supports both single-node multi-GPU and multi-node training

set -e  # Exit on error

# Default values
DATA_PATH="${OAKINK_PATH:-/DATA/disk0/OakInk}"
RENDER_PATH="${OAKINK_RENDER_DIR:-/DATA/disk0/OakInk/rendered_objects}"
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-100}
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=${CONFIG_FILE:-config_fsdp.py}
CHECKPOINT=${CHECKPOINT:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "=================================================="
echo "FSDP Training Configuration"
echo "=================================================="
echo "Data path: $DATA_PATH"
echo "Render path: $RENDER_PATH"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "World size: $WORLD_SIZE"
echo "Node rank: $NODE_RANK"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"
if [ -n "$CHECKPOINT" ]; then
    echo "Resume from checkpoint: $CHECKPOINT"
fi
echo "=================================================="

# Set environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export NODE_RANK=$NODE_RANK
export NUM_NODES=$NUM_NODES

# Build training command
CMD="train_fsdp.py"
CMD="$CMD --data_path $DATA_PATH"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --config $CONFIG_FILE"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Launch training based on configuration
if [ $NUM_NODES -eq 1 ]; then
    # Single-node multi-GPU training
    echo "Launching single-node multi-GPU training..."

    if command -v torchrun &> /dev/null; then
        # Use torchrun (recommended for PyTorch >= 1.9)
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$GPUS_PER_NODE \
            $CMD
    else
        # Fallback to torch.distributed.launch
        python -m torch.distributed.launch \
            --nproc_per_node=$GPUS_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $CMD
    fi
else
    # Multi-node training
    echo "Launching multi-node training (node $NODE_RANK of $NUM_NODES)..."

    if command -v torchrun &> /dev/null; then
        # Use torchrun for multi-node
        torchrun \
            --nnodes=$NUM_NODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --rdzv_id=funcgrasp_fsdp \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --node_rank=$NODE_RANK \
            $CMD
    else
        # Fallback for older PyTorch versions
        python -m torch.distributed.launch \
            --nproc_per_node=$GPUS_PER_NODE \
            --nnodes=$NUM_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $CMD
    fi
fi

echo "=================================================="
echo "FSDP training completed!"
echo "==================================================