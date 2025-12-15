#!/bin/bash

# Move into the project root
cd /kaggle/working/dbscan-for-cms
export PYTHONPATH=$(pwd):$PYTHONPATH

# Single test
python methods/hdbscan_clustering.py \
    --dataset_name cifar100 \
    --model_name cifar100_best \
    --min_cluster_size 10 \
    --min_samples 5 \
    --pretrain_path /kaggle/working/dwbscan-for-cms/models

# Full hyperparameter search
chmod +x hdbscan_tune_hyperparams.sh
./hdbscan_tune_hyperparams.sh
