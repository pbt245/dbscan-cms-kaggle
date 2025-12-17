#!/bin/bash

echo "Downloading pretrained models..."

# Create models directory
mkdir -p models

# Download DINO ViT-B/16
if [ ! -f "models/dino_vitbase16_pretrain.pth" ]; then
    echo "Downloading DINO ViT-B/16 pretrained model..."
    wget -O models/dino_vitbase16_pretrain.pth \
        https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
    
    # Verify file size (should be around 343 MB)
    FILE_SIZE=$(stat -f%z "models/dino_vitbase16_pretrain.pth" 2>/dev/null || stat -c%s "models/dino_vitbase16_pretrain.pth" 2>/dev/null)
    
    if [ "$FILE_SIZE" -lt 100000000 ]; then
        echo "ERROR: Downloaded file is too small. Please check your internet connection."
        rm models/dino_vitbase16_pretrain.pth
        exit 1
    fi
    
    echo "DINO model downloaded successfully"
else
    echo "DINO model already exists"
fi

echo "All models ready!"