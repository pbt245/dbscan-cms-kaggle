#!/bin/bash

echo "=========================================="
echo "CMS + ArcFace Setup Script"
echo "=========================================="

# Make all scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh
chmod +x bash_scripts/*.sh

echo "Scripts are now executable"

# Run environment setup
echo ""
bash scripts/setup_environment.sh

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download models
echo ""
bash scripts/download_pretrain.sh

# Prepare datasets
echo ""
bash scripts/prepare_datasets.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo ""
echo "  1. Train: bash scripts/contrastive_meanshift_arcface_training.sh"
echo "  2. Tune:  bash bash_scripts/hdbscan_tune_hyperparams.sh"
echo "  3. Test:  bash bash_scripts/hdbscan_clustering.sh"
