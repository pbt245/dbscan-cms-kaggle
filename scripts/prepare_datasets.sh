#!/bin/bash

echo "=========================================="
echo "Dataset Preparation"
echo "=========================================="

# Create datasets directory
mkdir -p datasets

echo ""
echo "Downloading CIFAR-10 and CIFAR-100..."
python get_dataset.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download CIFAR datasets"
    exit 1
fi

echo ""
echo "CIFAR datasets ready"