#!/bin/bash
# Training script for FuncGrasp with OakInk dataset

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate grasp

# Set environment variables (optional overrides)
# export OAKINK_PATH=/DATA/disk0/OakInk
# export OAKINK_RENDER_DIR=/DATA/disk0/OakInk/rendered_objects

# Training command
if [ "$1" == "test" ]; then
    echo "Running test mode - checking data loading..."
    python test_data_loading.py
elif [ "$1" == "cpu" ]; then
    echo "Running CPU training..."
    python train.py --device cpu --batch_size 2 --epochs 5
else
    echo "Running GPU training..."
    python train.py --epochs 100
fi
