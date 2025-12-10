#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Move into project root (important for correct paths)
cd /kaggle/working/dbscan-for-cms
export PYTHONPATH=$(pwd):$PYTHONPATH

DATASET="cifar100"
MODEL="cifar100_best"

# Grid search for eps and min_samples
mkdir -p log

echo "=== DBSCAN Hyperparameter Search ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo ""

for EPS in 0.10 0.15 0.20 0.25 0.30
do
    for MIN_SAMPLES in 3 5 8 10
    do
        echo "==========================================="
        echo "Testing eps=$EPS, min_samples=$MIN_SAMPLES"
        echo "==========================================="
        
        python methods/dbscan_clustering.py \
            --dataset_name $DATASET \
            --model_name $MODEL \
            --eps $EPS \
            --min_samples $MIN_SAMPLES \
            --epochs 10 \
            2>&1 | tee log/dbscan_${DATASET}_eps${EPS}_min${MIN_SAMPLES}.log
        
        echo ""
        echo "Completed: eps=$EPS, min_samples=$MIN_SAMPLES"
        echo ""
    done
done

echo "=== All experiments completed ==="
echo "Check log/ directory for results"

echo ""
echo "=== Summary of Results ==="
grep -h "Final Results" log/dbscan_*.log
