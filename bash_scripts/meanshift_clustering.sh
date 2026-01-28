export CUDA_VISIBLE_DEVICES=0

###################################################
# GCD
###################################################

python -m methods.meanshift_clustering \
        --dataset_name cifar100 \
        --model_name 'cifar100_best'

python -m methods.meanshift_clustering \
        --dataset_name imagenet_100 \
        --model_name 'imagenet_best'

python -m methods.meanshift_clustering \
        --dataset_name cub \
        --model_name 'cub_best' 

python -m methods.meanshift_clustering \
        --dataset_name scars \
        --model_name 'scars_best' 

python -m methods.meanshift_clustering \
        --dataset_name aircraft \
        --model_name 'aircraft_best' 

python -m methods.meanshift_clustering \
        --dataset_name herbarium_19 \
        --model_name 'herbarium_19_best' 

###################################################
# INDUCTIVE GCD
###################################################

python -m methods.meanshift_clustering \
        --dataset_name cifar100 \
        --model_name 'cifar100_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name imagenet_100 \
        --model_name 'imagenet_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name cub \
        --model_name 'cub_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name scars \
        --model_name 'scars_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name aircraft \
        --model_name 'aircraft_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name herbarium_19 \
        --model_name 'herbarium_19_best' \
        --inductive