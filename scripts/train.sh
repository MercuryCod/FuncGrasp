#!/bin/bash
# Training script for Functional Grasp Prediction
#
# Usage:
#   bash scripts/train.sh                           # Staged training (default)
#   TRAINING_MODE=joint bash scripts/train.sh       # Joint training (legacy)
#   DEBUG=true bash scripts/train.sh                # Debug mode (frequent logging)
#   QWEN_TUNING=full bash scripts/train.sh          # Full fine-tuning
#
# Environment variables:
#   TRAINING_MODE=staged/joint  Training strategy (default: staged)
#   DEBUG=true/false            Enable debug mode (default: false)
#   QWEN_TUNING=lora/full       Qwen tuning mode (default: lora)
#   DATA_ROOT=/path             Data root directory (default: /workspace/data)

set -e  # Exit on error

# Get experiment name from argument or environment
EXP_NAME=lora

# Set defaults for optional environment variables
TRAINING_MODE=${TRAINING_MODE:-staged}
DEBUG=${DEBUG:-false}
QWEN_TUNING=${QWEN_TUNING:-lora}
DATA_ROOT=${DATA_ROOT:-/workspace/data}

echo "================================================================================"
echo "FUNCTIONAL GRASP TRAINING"
echo "================================================================================"
echo "Experiment: $EXP_NAME"
echo "Device: GPU (CUDA)"
echo ""

# Configuration
export EXP_NAME=$EXP_NAME
export TRAINING_MODE=$TRAINING_MODE
export DEBUG=$DEBUG
export QWEN_TUNING=$QWEN_TUNING
export DATA_ROOT=$DATA_ROOT

echo "Configuration:"
echo "  Training mode: $TRAINING_MODE"
echo "  Qwen tuning: $QWEN_TUNING"
echo "  Debug mode: $DEBUG"
echo "  Data root: $DATA_ROOT"
echo ""

if [ "$TRAINING_MODE" = "staged" ]; then
    echo "Training Pipeline (STAGED):"
    echo "  Stage 1 (50 epochs): Contact prediction only"
    echo "  Stage 2 (50 epochs): Joint contact + flow matching"
    echo "  Benefits: Cleaner gradients, better contact learning"
    echo "  Checkpoints: stage1_final.pth and best.pth"
elif [ "$TRAINING_MODE" = "joint" ]; then
    echo "Training Pipeline (JOINT):"
    echo "  100 epochs: Contact + flow from start"
    echo "  Uses gradient isolation (.detach())"
    echo "  Backward compatible with original approach"
fi
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
