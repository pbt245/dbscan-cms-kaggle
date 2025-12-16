#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

###################################################
# HDBSCAN Clustering - GCD
###################################################

# CIFAR-100
# python -m methods.hdbscan_clustering \
#     --dataset_name 'cifar100' \
#     --model_name 'cifar100_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --eval_funcs v2 \
#     --use_ssb_splits True \
#     --wandb

# # ImageNet-100
# python -m methods.hdbscan_clustering \
#     --dataset_name 'imagenet_100' \
#     --model_name 'imagenet_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --wandb

# # CUB
# python -m methods.hdbscan_clustering \
#     --dataset_name 'cub' \
#     --model_name 'cub_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --wandb

# # Stanford Cars
# python -m methods.hdbscan_clustering \
#     --dataset_name 'scars' \
#     --model_name 'scars_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --wandb

# # FGVC-Aircraft
# python -m methods.hdbscan_clustering \
#     --dataset_name 'aircraft' \
#     --model_name 'aircraft_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --wandb

# # Herbarium-19
# python -m methods.hdbscan_clustering \
#     --dataset_name 'herbarium_19' \
#     --model_name 'herbarium_19_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --wandb


###################################################
# HDBSCAN Clustering - INDUCTIVE GCD
###################################################

# CIFAR-100 (Inductive)
python -m methods.hdbscan_clustering \
    --dataset_name 'cifar100' \
    --model_name 'cifar100_best' \
    --batch_size 128 \
    --num_workers 8 \
    --epochs 20 \
    --k 8 \
    --alpha 0.5 \
    --min_cluster_size 10 \
    --min_samples 5 \
    --inductive \
    --wandb

# # ImageNet-100 (Inductive)
# python -m methods.hdbscan_clustering \
#     --dataset_name 'imagenet_100' \
#     --model_name 'imagenet_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --inductive \
#     --wandb

# # CUB (Inductive)
# python -m methods.hdbscan_clustering \
#     --dataset_name 'cub' \
#     --model_name 'cub_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --inductive \
#     --wandb

# # Stanford Cars (Inductive)
# python -m methods.hdbscan_clustering \
#     --dataset_name 'scars' \
#     --model_name 'scars_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --inductive \
#     --wandb

# # FGVC-Aircraft (Inductive)
# python -m methods.hdbscan_clustering \
#     --dataset_name 'aircraft' \
#     --model_name 'aircraft_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --inductive \
#     --wandb

# # Herbarium-19 (Inductive)
# python -m methods.hdbscan_clustering \
#     --dataset_name 'herbarium_19' \
#     --model_name 'herbarium_19_best' \
#     --batch_size 128 \
#     --num_workers 8 \
#     --epochs 20 \
#     --k 8 \
#     --alpha 0.5 \
#     --min_cluster_size 10 \
#     --min_samples 5 \
#     --inductive \
#     --wandb