#!/bin/bash
# Setup script to prepare training statistics before running train.sh
# Run this ONCE after data preparation to compute class frequencies

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

data_path="/mnt/data/changma/OakInk"
render_dir="$data_path/rendered_objects"

echo "=== FuncGrasp Training Setup ==="
echo "Data path: $data_path"
echo "Render dir: $render_dir"
echo "================================"

# Step 1: Compute class frequencies for pos_weight
echo ""
echo "Step 1: Computing class frequencies from training data..."
echo "This will iterate through the training split once to count contact labels."
python compute_class_frequencies.py --data_path $data_path --render_dir $render_dir

# Check if it succeeded
if [ -f "$data_path/cache/class_frequencies_train.json" ]; then
    echo ""
    echo "✅ Class frequencies computed successfully!"
    echo "   File: $data_path/cache/class_frequencies_train.json"
    echo ""
    echo "The pos_weight values will be automatically loaded during training."
else
    echo ""
    echo "❌ Failed to compute class frequencies. Check the error above."
    echo "   Training will continue with uniform weights (pos_weight=None)."
fi

echo ""
echo "=== Setup Complete ==="
echo "You can now run: bash scripts/train.sh"
echo "======================"

