#!/bin/bash
# Training script for Functional Grasp Prediction
# Usage: bash scripts/train.sh [experiment_name]
#
# Environment variables (optional):
#   DEBUG=true         Enable debug mode (default: false)
#   QWEN_TUNING=lora   Qwen tuning mode: frozen/lora/full (default: lora)
#   DATA_ROOT=/path    Data root directory (default: /workspace/data)

set -e  # Exit on error

# Get experiment name from argument or environment
EXP_NAME=lora

# Set defaults for optional environment variables
DEBUG=false
QWEN_TUNING=lora
DATA_ROOT=${DATA_ROOT:-/workspace/data}

echo "================================================================================"
echo "FUNCTIONAL GRASP TRAINING"
echo "================================================================================"
echo "Experiment: $EXP_NAME"
echo "Device: GPU (CUDA)"
echo ""

# Configuration
export EXP_NAME=$EXP_NAME
export DEBUG=$DEBUG
export QWEN_TUNING=$QWEN_TUNING
export DATA_ROOT=$DATA_ROOT

echo "Configuration:"
echo "  Debug mode: $DEBUG"
echo "  Qwen tuning: $QWEN_TUNING"
echo "  Data root: $DATA_ROOT"
echo ""

# Run training
python train.py

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
echo ""
echo "View logs at: ./exp/$EXP_NAME/logs/run.log"
echo "Monitor live: tail -f ./exp/$EXP_NAME/logs/run.log"
