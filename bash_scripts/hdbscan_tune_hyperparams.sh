#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Configuration
DATASET="cifar100"
MODEL="cifar100_best"

# Create log directory
mkdir -p log/hdbscan_tuning

echo "=== HDBSCAN Hyperparameter Search ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo ""

# Grid search for min_cluster_size and min_samples
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
        
        python -m methods.hdbscan_clustering \
            --dataset_name $DATASET \
            --model_name $MODEL \
            --batch_size 128 \
            --num_workers 8 \
            --epochs 10 \
            --k 8 \
            --alpha 0.5 \
            --min_cluster_size $MIN_CLUSTER_SIZE \
            --min_samples $MIN_SAMPLES \
            --eval_funcs v2 \
            --use_ssb_splits True \
            --wandb \
            2>&1 | tee log/hdbscan_tuning/hdbscan_${DATASET}_mcs${MIN_CLUSTER_SIZE}_ms${MIN_SAMPLES}.log
        
        echo ""
        echo "Completed: min_cluster_size=$MIN_CLUSTER_SIZE, min_samples=$MIN_SAMPLES"
        echo ""
    done
done

echo "=== All experiments completed ==="
echo "Check log/hdbscan_tuning/ directory for results"

echo ""
echo "=== Summary of Results ==="
grep -h "FINAL RESULTS" log/hdbscan_tuning/hdbscan_*.log