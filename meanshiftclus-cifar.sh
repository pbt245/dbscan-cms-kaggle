#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python -m methods.meanshift_clustering \
        --dataset_name cifar100 \
        --warmup_model_dir 'cifar100_best'