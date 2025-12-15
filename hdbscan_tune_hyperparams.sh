#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Move into project root (important for correct paths)
cd /kaggle/working/dbscan-for-cms
export PYTHONPATH=$(pwd):$PYTHONPATH

DATASET="cifar100"
MODEL="cifar100_best"

# Grid search for min_cluster_size and min_samples
mkdir -p log

echo "=== HDBSCAN Hyperparameter Search ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo ""

# HDBSCAN uses min_cluster_size instead of eps
# min_cluster_size: minimum size of clusters (typically 5-50)
# min_samples: how conservative the clustering is (typically 1-10)

for MIN_CLUSTER_SIZE in 5 10 15 20 30
do
    for MIN_SAMPLES in 1 3 5 8 10
    do
        echo "==========================================="
        echo "Testing min_cluster_size=$MIN_CLUSTER_SIZE, min_samples=$MIN_SAMPLES"
        echo "==========================================="
        
        python methods/hdbscan_clustering.py \
            --dataset_name $DATASET \
            --model_name $MODEL \
            --min_cluster_size $MIN_CLUSTER_SIZE \
            --min_samples $MIN_SAMPLES \
            --epochs 10 \
            2>&1 | tee log/hdbscan_${DATASET}_mcs${MIN_CLUSTER_SIZE}_min${MIN_SAMPLES}.log
        
        echo ""
        echo "Completed: min_cluster_size=$MIN_CLUSTER_SIZE, min_samples=$MIN_SAMPLES"
        echo ""
    done
done

echo "=== All experiments completed ==="
echo "Check log/ directory for results"

echo ""
echo "=== Summary of Results ==="
grep -h "Final Results" log/hdbscan_*.log
