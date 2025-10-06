#!/bin/bash
# Training script for FuncGrasp with OakInk dataset

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

data_path="/mnt/data/changma/OakInk"

# User-tunable params (with defaults). Override by exporting before calling or edit here.
BATCH_SIZE=64
LOG_INTERVAL=10
SAVE_INTERVAL=500

# Automatically set environment variables based on data_path
export TOKENIZERS_PARALLELISM=false
export OAKINK_PATH=$data_path
export OAKINK_RENDER_DIR=$data_path/rendered_objects

echo "Using OakInk data path: $OAKINK_PATH"
echo "Using rendered objects path: $OAKINK_RENDER_DIR"

echo "Batch size: $BATCH_SIZE | Log interval: $LOG_INTERVAL | Save interval: $SAVE_INTERVAL"

# Frozen mode (default)
# python train.py --data_path $data_path

# # Full fine-tuning
# python train.py --data_path $data_path --qwen_tuning full

# # LoRA fine-tuning
python train.py --data_path $data_path \
  --qwen_tuning lora --lora_r 16 --lora_alpha 32 \
  --batch_size "$BATCH_SIZE" \
  --log_interval "$LOG_INTERVAL" \
  --checkpoint_interval "$SAVE_INTERVAL"

