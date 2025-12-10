#!/bin/bash

# Move into the project root
cd /kaggle/working/dbscan-for-cms
export PYTHONPATH=$(pwd):$PYTHONPATH

# Single test
python methods/dbscan_clustering.py \
    --dataset_name cifar100 \
    --model_name cifar100_best \
    --eps 0.15 \
    --min_samples 5 \
    --pretrain_path /kaggle/working/dbscan-for-cms/models

# Full hyperparameter search
chmod +x dbscan_tune_hyperparams.sh
./dbscan_tune_hyperparams.sh
