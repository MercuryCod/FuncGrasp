#!/bin/bash
# Pre-render all OakInk objects for faster training
# This is a one-time preprocessing step that takes ~2 minutes

set -e

OAKINK_ROOT="${OAKINK_ROOT:-/workspace/data/OakInk}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
ZOOM="${ZOOM:-1.6}"

echo "Pre-rendering OakInk objects..."
echo "  OakInk root: $OAKINK_ROOT"
echo "  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Zoom: $ZOOM"
echo ""

python3 "$(dirname "$0")/prerender_objects.py" \
    --oakink_root "$OAKINK_ROOT" \
    --image_size "$IMAGE_SIZE" \
    --zoom "$ZOOM"

echo ""
echo "Done! Rendered images saved to: ${OAKINK_ROOT}/rendered_objects/"
echo ""
echo "To use in training, load dataset with:"
echo "  dataset = OakInkDataset(data_root='$OAKINK_ROOT', load_object_images=True)"

