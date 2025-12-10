import os

# Root directory of this project
CMS_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'

# Dataset directories
DATASET_ROOT = CMS_ROOT + 'datasets/'
cifar_10_root = DATASET_ROOT + 'cifar10'
cifar_100_root = DATASET_ROOT + 'cifar100'
cub_root = DATASET_ROOT
aircraft_root = DATASET_ROOT + 'fgvc-aircraft-2013b'
herbarium_dataroot = DATASET_ROOT + 'herbarium_19'
imagenet_root = DATASET_ROOT + 'imagenet'

# OSR splits
osr_split_dir = CMS_ROOT + 'data/ssb_splits'

# Pretrained models
dino_pretrain_path = CMS_ROOT + 'models/dino_vitbase16_pretrain.pth'

# Log directory
exp_root = CMS_ROOT + 'log'
