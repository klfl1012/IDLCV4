"""
Example script demonstrating how to use the PotholeProposalDataset.

This script shows how to:
1. Create a dataset instance with different hyperparameters
2. Create a DataLoader for batch training
3. Iterate through the data
4. Visualize samples
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from dataloader import (
    PotholeProposalDataset,
    DEFAULT_DATA_ROOTS
)


def visualize_batch(images, labels, num_samples=8):
    """Visualize a batch of images with their labels."""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        label = labels[i].item()
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {'Pothole' if label == 1 else 'No Pothole'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch.png')
    print("Saved visualization to 'sample_batch.png'")
    plt.close()


def main():
    # ============================================================================
    # Example 1: Basic usage with Selective Search proposals
    # ============================================================================
    print("=" * 80)
    print("Example 1: Basic Selective Search Dataset")
    print("=" * 80)
    
    dataset_ss = PotholeProposalDataset(
        data_root=DEFAULT_DATA_ROOTS["pothole"],
        proposal_type='selective_search',
        proposal_json='selective_search_proposals.json',
        image_size=(256, 256),
        iou_threshold=0.7,
        positive_ratio=0.25,
        seed=42
    )
    
    print(f"Total samples: {len(dataset_ss)}")
    
    # Test getting a single sample
    image, label = dataset_ss[0]
    print(f"Image shape: {image.shape}, Label: {label}")
    
    # ============================================================================
    # Example 2: Using EdgeBox proposals with different hyperparameters
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 2: EdgeBox Dataset with Custom Hyperparameters")
    print("=" * 80)
    
    dataset_eb = PotholeProposalDataset(
        data_root=DEFAULT_DATA_ROOTS["pothole"],
        proposal_type='edge_box',
        proposal_json='edge_box_proposals.json',
        image_size=(224, 224),  # Different image size
        iou_threshold=0.5,       # Lower IoU threshold
        positive_ratio=0.30,     # 30% positive samples
        seed=42
    )
    
    print(f"Total samples: {len(dataset_eb)}")
    
    # ============================================================================
    # Example 3: Creating a DataLoader for batch training
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 3: DataLoader for Batch Training")
    print("=" * 80)
    
    # Create DataLoader with batching and shuffling
    dataloader = DataLoader(
        dataset_ss,
        batch_size=16,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster GPU transfer
    )
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through a few batches
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 3:  # Just show first 3 batches
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Positive samples in batch: {labels.sum().item()}/{len(labels)}")
        
        # Visualize first batch
        if batch_idx == 0:
            visualize_batch(images, labels)
    
    # ============================================================================
    # Example 4: Using with data augmentation transforms
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 4: Dataset with Data Augmentation")
    print("=" * 80)
    
    # Define transforms for data augmentation
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_augmented = PotholeProposalDataset(
        data_root=DEFAULT_DATA_ROOTS["pothole"],
        proposal_type='selective_search',
        proposal_json='selective_search_proposals.json',
        image_size=(256, 256),
        iou_threshold=0.7,
        positive_ratio=0.25,
        transform=augmentation_transforms,
        seed=42
    )
    
    print(f"Dataset with augmentation created: {len(dataset_augmented)} samples")
    
    # ============================================================================
    # Example 5: Train/Val/Test split
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 5: Creating Train/Val/Test Splits")
    print("=" * 80)
    
    from torch.utils.data import random_split
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    total_size = len(dataset_ss)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_ss,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ============================================================================
    # Example 6: Computing class statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("Example 6: Dataset Statistics")
    print("=" * 80)
    
    # Count positive and negative samples
    positive_count = sum(1 for i in range(len(dataset_ss)) if dataset_ss.samples[i]['label'] == 1)
    negative_count = len(dataset_ss) - positive_count
    
    print(f"Total samples: {len(dataset_ss)}")
    print(f"Positive (pothole) samples: {positive_count} ({positive_count/len(dataset_ss)*100:.2f}%)")
    print(f"Negative (no pothole) samples: {negative_count} ({negative_count/len(dataset_ss)*100:.2f}%)")
    
    # IoU statistics
    ious = [sample['iou'] for sample in dataset_ss.samples]
    print(f"\nIoU statistics:")
    print(f"  Mean IoU: {np.mean(ious):.4f}")
    print(f"  Median IoU: {np.median(ious):.4f}")
    print(f"  Max IoU: {np.max(ious):.4f}")
    print(f"  Min IoU: {np.min(ious):.4f}")


if __name__ == "__main__":
    main()
