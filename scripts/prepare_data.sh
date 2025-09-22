#!/bin/bash

OAKINK_PATH="/mnt/data/changma/OakInk"
OAKINK_RENDER_DIR="/mnt/data/changma/OakInk/rendered_objects"

# Check if test mode is requested
if [ "$1" = "test" ]; then
    echo "Running in test mode - rendering small sample"
    python -m dataset.prepare_data --oakink_root "$OAKINK_PATH" --output_dir ./sample_rendered_objects --sample --num_samples 3
else
    echo "Running full data preparation"
    # Render all objects
    python -m dataset.prepare_data --oakink_root "$OAKINK_PATH" --output_dir "$OAKINK_RENDER_DIR"
fi