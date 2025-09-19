#!/bin/bash

# Check if test mode is requested
if [ "$1" = "test" ]; then
    echo "Running in test mode - rendering small sample"
    python -m dataset.prepare_data --oakink_root /DATA/disk0/OakInk --output_dir ./sample_rendered_objects --sample --num_samples 3
else
    echo "Running full data preparation"
    # Render all objects
    python -m dataset.prepare_data --oakink_root /DATA/disk0/OakInk --output_dir /DATA/disk0/OakInk/rendered_objects
fi