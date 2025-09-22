#!/bin/bash
# Training script for FuncGrasp with OakInk dataset

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

data_path="/mnt/data/changma/OakInk"

# Automatically set environment variables based on data_path
export OAKINK_PATH=$data_path
export OAKINK_RENDER_DIR=$data_path/rendered_objects

echo "Using OakInk data path: $OAKINK_PATH"
echo "Using rendered objects path: $OAKINK_RENDER_DIR"

# Frozen mode (default)
# python train.py --data_path $data_path

# # Full fine-tuning
# python train.py --data_path $data_path --qwen_tuning full

# # LoRA fine-tuning
python train.py --data_path $data_path --qwen_tuning lora --lora_r 16 --lora_alpha 32

