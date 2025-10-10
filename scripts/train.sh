#!/bin/bash
# Training script for FuncGrasp with OakInk dataset
# Includes contact accuracy improvements with class-based regression

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

data_path="/mnt/data/changma/OakInk"

# User-tunable params (with defaults). Override by exporting before calling or edit here.
BATCH_SIZE=${BATCH_SIZE:-64}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EPOCHS=${EPOCHS:-10}

# Contact regression params
CONTACT_REGRESSION=${CONTACT_REGRESSION:-true}  # Use regression (BCE) by default
CONTACT_LOSS_TYPE=${CONTACT_LOSS_TYPE:-bce}     # bce or ce
INFERENCE_THRESHOLD=${INFERENCE_THRESHOLD:-0.4}
TAU_MM=${TAU_MM:-10.0}

# Automatically set environment variables based on data_path
export TOKENIZERS_PARALLELISM=false
export OAKINK_PATH=$data_path
export OAKINK_RENDER_DIR=$data_path/rendered_objects

echo "=== FuncGrasp Training Configuration ==="
echo "Data path: $OAKINK_PATH"
echo "Rendered objects: $OAKINK_RENDER_DIR"
echo "Batch size: $BATCH_SIZE | Epochs: $EPOCHS"
echo "Log interval: $LOG_INTERVAL | Save interval: $SAVE_INTERVAL"
echo "Contact regression: $CONTACT_REGRESSION | Loss type: $CONTACT_LOSS_TYPE"
echo "Inference threshold: $INFERENCE_THRESHOLD | Tau: $TAU_MM mm"
echo "========================================"

# Build contact regression arguments
if [ "$CONTACT_REGRESSION" = "true" ]; then
  CONTACT_ARGS="--contact_regression --contact_loss_type $CONTACT_LOSS_TYPE --inference_threshold $INFERENCE_THRESHOLD --tau_mm $TAU_MM"
else
  CONTACT_ARGS="--no_contact_regression --contact_loss_type ce"
fi

# LoRA fine-tuning with contact regression (default)
python train.py --data_path $data_path \
  --qwen_tuning lora --lora_r 16 --lora_alpha 32 \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --log_interval "$LOG_INTERVAL" \
  --checkpoint_interval "$SAVE_INTERVAL" \
  $CONTACT_ARGS

# Alternative modes (comment above and uncomment one of these):

# Frozen Qwen with contact regression
# python train.py --data_path $data_path \
#   --qwen_tuning frozen \
#   --batch_size "$BATCH_SIZE" \
#   --epochs "$EPOCHS" \
#   $CONTACT_ARGS

# Classification mode (CE) for comparison
# python train.py --data_path $data_path \
#   --qwen_tuning lora --lora_r 16 --lora_alpha 32 \
#   --batch_size "$BATCH_SIZE" \
#   --epochs "$EPOCHS" \
#   --no_contact_regression --contact_loss_type ce

