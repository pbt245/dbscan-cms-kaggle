import torch
import torchvision
from torchvision import datasets
from config import DATASET_ROOT

print(" Downloading CIFAR-100 Dataset...")
print("="*60)

# CIFAR-100 will auto-download on first use
# But we can pre-download it here
try:
    # Download train set
    train_dataset = datasets.CIFAR100(
        root=DATASET_ROOT+'cifar100',
        train=True,
        download=True
    )
    print(f" CIFAR-100 Train: {len(train_dataset)} images")
    
    # Download test set
    test_dataset = datasets.CIFAR100(
        root=DATASET_ROOT+'cifar100',
        train=False,
        download=True
    )
    print(f" CIFAR-100 Test: {len(test_dataset)} images")
    
except Exception as e:
    print(f"Error downloading CIFAR-100: {e}")

print("\n Downloading CIFAR-10 Dataset...")
try:
    # Download CIFAR-10 as well
    train_dataset = datasets.CIFAR10(
        root=DATASET_ROOT+'cifar10',
        train=True,
        download=True
    )
    print(f" CIFAR-10 Train: {len(train_dataset)} images")
    
    test_dataset = datasets.CIFAR10(
        root=DATASET_ROOT+'cifar10',
        train=False,
        download=True
    )
    print(f" CIFAR-10 Test: {len(test_dataset)} images")
    
except Exception as e:
    print(f"Error downloading CIFAR-10: {e}")

print("\nDataset download complete!")