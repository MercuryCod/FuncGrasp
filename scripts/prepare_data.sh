#!/bin/bash
# Data preparation script - renders object meshes to multi-view images
# This is a ONE-TIME setup step. After running this, use setup_training.sh.

OAKINK_PATH="/mnt/data/changma/OakInk"
OAKINK_RENDER_DIR="/mnt/data/changma/OakInk/rendered_objects"

echo "=== FuncGrasp Data Preparation ==="
echo "This script renders object meshes to multi-view images."
echo "Run this ONCE before training."
echo "===================================="

# Check if test mode is requested
if [ "$1" = "test" ]; then
    echo "Running in test mode - rendering small sample"
    python -m dataset.prepare_data --oakink_root "$OAKINK_PATH" --output_dir ./sample_rendered_objects --sample --num_samples 3
else
    echo "Running full data preparation"
    # Render all objects
    python -m dataset.prepare_data --oakink_root "$OAKINK_PATH" --output_dir "$OAKINK_RENDER_DIR"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Run: bash scripts/setup_training.sh  (computes class frequencies)"
echo "2. Run: bash scripts/train.sh          (starts training)"
echo "==================="