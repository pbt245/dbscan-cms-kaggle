#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

###################################################
# CMS + ArcFace - GCD
###################################################

# CIFAR-100
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'cifar100' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.01 \
#     --eta_min 1e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.3 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --transform 'imagenet' \
#     --seed 1 \
#     --n_views 2 \
#     --contrast_unlabel_only False \
#     --wandb

# # ImageNet-100
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'imagenet_100' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.01 \
#     --eta_min 1e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.3 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --wandb

# # CUB
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'cub' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --wandb

# # Stanford Cars
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'scars' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --wandb

# # FGVC-Aircraft
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'aircraft' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --wandb

# # Herbarium-19
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'herbarium_19' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --momentum 0.9 \
#     --weight_decay 5e-5 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --alpha 0.5 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --wandb


###################################################
# CMS + ArcFace - INDUCTIVE GCD
###################################################

# CIFAR-100 (Inductive)
python -m methods.contrastive_meanshift_arcface_training \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --num_workers 16 \
    --lr 0.01 \
    --eta_min 1e-3 \
    --epochs 200 \
    --temperature 0.3 \
    --sup_con_weight 0.35 \
    --k 8 \
    --arcface_weight 0.3 \
    --arcface_s 64.0 \
    --arcface_m 0.5 \
    --inductive \
    --wandb

# # ImageNet-100 (Inductive)
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'imagenet_100' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.01 \
#     --eta_min 1e-3 \
#     --epochs 200 \
#     --temperature 0.3 \
#     --sup_con_weight 0.35 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --inductive \
#     --wandb

# # CUB (Inductive)
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'cub' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --inductive \
#     --wandb

# # Stanford Cars (Inductive)
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'scars' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --inductive \
#     --wandb

# # FGVC-Aircraft (Inductive)
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'aircraft' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --inductive \
#     --wandb

# # Herbarium-19 (Inductive)
# python -m methods.contrastive_meanshift_arcface_training \
#     --dataset_name 'herbarium_19' \
#     --batch_size 128 \
#     --num_workers 16 \
#     --lr 0.05 \
#     --eta_min 5e-3 \
#     --epochs 200 \
#     --temperature 0.25 \
#     --sup_con_weight 0.35 \
#     --k 8 \
#     --arcface_weight 0.3 \
#     --arcface_s 64.0 \
#     --arcface_m 0.5 \
#     --inductive \
#     --wandb