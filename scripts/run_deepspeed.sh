#!/bin/bash
# DeepSpeed training launch script for FuncGrasp
# Supports single-node and multi-node distributed training

set -e  # Exit on error

# Default values
DATA_PATH="${OAKINK_PATH:-/DATA/disk0/OakInk}"
RENDER_PATH="${OAKINK_RENDER_DIR:-/DATA/disk0/OakInk/rendered_objects}"
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-100}
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
DS_CONFIG=${DS_CONFIG:-ds_config_zero2.json}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-""}
HOSTFILE=${HOSTFILE:-""}

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
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --ds_config)
            DS_CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --zero)
            # Shortcut for selecting ZeRO stage
            case $2 in
                2)
                    DS_CONFIG="ds_config_zero2.json"
                    ;;
                3)
                    DS_CONFIG="ds_config_zero3.json"
                    ;;
                3-offload)
                    DS_CONFIG="ds_config_zero3_offload.json"
                    ;;
                *)
                    echo "Invalid ZeRO stage: $2"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect GPUs if not specified
if [ "$GPUS_PER_NODE" == "auto" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    echo "Auto-detected $GPUS_PER_NODE GPUs"
fi

# Calculate world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "=================================================="
echo "DeepSpeed Training Configuration"
echo "=================================================="
echo "Data path: $DATA_PATH"
echo "Render path: $RENDER_PATH"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "World size: $WORLD_SIZE"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "DeepSpeed config: $DS_CONFIG"
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Resume from checkpoint: $CHECKPOINT_DIR"
fi
if [ -n "$HOSTFILE" ]; then
    echo "Using hostfile: $HOSTFILE"
fi
echo "=================================================="

# Build training command
TRAIN_SCRIPT="train_deepspeed.py"
TRAIN_ARGS=""
TRAIN_ARGS="$TRAIN_ARGS --data_path $DATA_PATH"
TRAIN_ARGS="$TRAIN_ARGS --batch_size $BATCH_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --epochs $EPOCHS"
TRAIN_ARGS="$TRAIN_ARGS --deepspeed_config $DS_CONFIG"

if [ -n "$CHECKPOINT_DIR" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --checkpoint_dir $CHECKPOINT_DIR"
fi

# Set environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Launch training
if [ $NUM_NODES -eq 1 ]; then
    # Single-node training
    echo "Launching single-node DeepSpeed training..."

    deepspeed --num_gpus=$GPUS_PER_NODE \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              $TRAIN_SCRIPT \
              $TRAIN_ARGS \
              --deepspeed

else
    # Multi-node training
    echo "Launching multi-node DeepSpeed training..."

    if [ -z "$HOSTFILE" ]; then
        echo "ERROR: Hostfile required for multi-node training"
        echo "Create a hostfile with format:"
        echo "  hostname1 slots=8"
        echo "  hostname2 slots=8"
        exit 1
    fi

    deepspeed --hostfile=$HOSTFILE \
              --num_nodes=$NUM_NODES \
              --num_gpus=$GPUS_PER_NODE \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              --launcher=pdsh \
              $TRAIN_SCRIPT \
              $TRAIN_ARGS \
              --deepspeed
fi

echo "=================================================="
echo "DeepSpeed training completed!"
echo "=================================================="