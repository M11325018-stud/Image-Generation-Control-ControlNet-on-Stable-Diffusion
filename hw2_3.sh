#!/bin/bash

# HW2 Problem 3 - ControlNet Inference Script
# This script must be executable within 15 minutes as per PDF requirements

# Check if correct number of arguments provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash hw2_3.sh <json_path> <source_dir> <output_dir> <controlnet_ckpt>"
    echo "Example: bash hw2_3.sh prompt.json ./source ./output model.ckpt"
    exit 1
fi

# Parse arguments (as specified in PDF page 35)
JSON_PATH=$1       # Path to prompt.json
SOURCE_DIR=$2      # Path to source images folder
OUTPUT_DIR=$3      # Path to output folder
CONTROLNET_CKPT=$4 # Path to trained ControlNet checkpoint

echo "======================================================================"
echo "HW2 Problem 3 - ControlNet Inference"
echo "======================================================================"
echo "JSON file:        $JSON_PATH"
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "ControlNet model: $CONTROLNET_CKPT"
echo "======================================================================"

# Validate inputs
if [ ! -f "$JSON_PATH" ]; then
    echo "[ERROR] JSON file not found: $JSON_PATH"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] Source directory not found: $SOURCE_DIR"
    exit 1
fi

if [ ! -f "$CONTROLNET_CKPT" ]; then
    echo "[ERROR] ControlNet checkpoint not found: $CONTROLNET_CKPT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Path to Stable Diffusion (as per PDF, TAs will install with pip install -e)
SD_CONFIG="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
SD_CKPT="../models/ldm/stable-diffusion-v1/model.ckpt"

# Check if SD files exist (try alternative paths)
if [ ! -f "$SD_CONFIG" ]; then
    if [ -f "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml" ]; then
        SD_CONFIG="./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    elif [ -f "../stable-diffusion/configs/stable-diffusion/v1-inference.yaml" ]; then
        SD_CONFIG="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    fi
fi

if [ ! -f "$SD_CKPT" ]; then
    if [ -f "./models/ldm/stable-diffusion-v1/model.ckpt" ]; then
        SD_CKPT="./models/ldm/stable-diffusion-v1/model.ckpt"
    elif [ -f "../models/ldm/stable-diffusion-v1/model.ckpt" ]; then
        SD_CKPT="../models/ldm/stable-diffusion-v1/model.ckpt"
    fi
fi

# Run inference
# Parameters:
# - image_size: 512 (as per your code)
# - ddim_steps: 50 (standard DDIM)
# - scale: 7.5 (standard classifier-free guidance)
# - eta: 0.0 (deterministic sampling)
# - seed: 42 (fixed for reproducibility as per PDF page 18)
# - control_strength: 1.0 (full control)

echo ""
echo "[INFO] Starting ControlNet inference..."
echo "[INFO] This should complete within 15 minutes (PDF requirement)"
echo ""

python3 inference_hw2.3.py \
    --json_path "$JSON_PATH" \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --controlnet_path "$CONTROLNET_CKPT" \
    --sd_config "$SD_CONFIG" \
    --sd_ckpt "$SD_CKPT" \
    --ddim_steps 50 \
    --scale 7.5 \
    --eta 0.0 \
    --seed 42 \
    --image_size 512 \
    --control_strength 1.0

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "[SUCCESS] ControlNet inference completed successfully!"
    echo "[INFO] Output images saved to: $OUTPUT_DIR"
    echo "======================================================================"
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "[ERROR] Inference failed! Please check the error messages above."
    echo "======================================================================"
    exit 1
fi