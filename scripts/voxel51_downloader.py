import fiftyone as fo
import fiftyone.zoo as foz
import random
import os
from pathlib import Path

# Set random seed for reproducible results
seed = random.randint(1000, 9999)
print(f"Using random seed: {seed}")

# Set up download directory structure
base_dir = Path(__file__).parent.parent / 'data' / 'oid_urban'
train_dir = base_dir / 'train'
val_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

# Create directories if they don't exist
for dir_path in [train_dir, val_dir, test_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Download directories set up:")
print(f"  Train: {train_dir}")
print(f"  Validation: {val_dir}")
print(f"  Test: {test_dir}")

# Load classes from my_classes.txt
def load_target_classes():
    """Load target classes from my_classes.txt."""
    classes_path = Path(__file__).parent.parent / 'my_classes.txt'
    
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_path}")
    
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(classes)} target classes from my_classes.txt")
    return classes

# Load target classes
target_classes = load_target_classes()

#
# Download 1000 images from Open Images V7 dataset
# Filtered by classes from my_classes.txt
# Label type: segmentations
# Distributed across train, validation, test splits
# With shuffling and random seed
#

# Calculate distribution across splits (70% train, 20% validation, 10% test)
train_samples = int(1000 * 0.7)  # 700
val_samples = int(1000 * 0.2)    # 200  
test_samples = 1000 - train_samples - val_samples  # 100

print(f"Downloading {1000} images distributed as:")
print(f"  Train: {train_samples} samples")
print(f"  Validation: {val_samples} samples") 
print(f"  Test: {test_samples} samples")

# Download train split
print("\n=== Downloading TRAIN split ===")
train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["segmentations"],
    classes=target_classes,
    max_samples=train_samples,
    only_matching=True,
    shuffle=True,
    seed=seed,
    dataset_dir=str(train_dir),
)

# Download validation split  
print("\n=== Downloading VALIDATION split ===")
val_dataset = foz.load_zoo_dataset(
    "open-images-v7", 
    split="validation",
    label_types=["segmentations"],
    classes=target_classes,
    max_samples=val_samples,
    only_matching=True,
    shuffle=True,
    seed=seed + 1,  # Different seed for validation
    dataset_dir=str(val_dir),
)

# Download test split
print("\n=== Downloading TEST split ===")
test_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test", 
    label_types=["segmentations"],
    classes=target_classes,
    max_samples=test_samples,
    only_matching=True,
    shuffle=True,  
    seed=seed + 2,  # Different seed for test
    dataset_dir=str(test_dir),
)

# Merge all datasets
print("\n=== Merging datasets ===")
dataset = train_dataset.clone()
dataset.merge_samples(val_dataset)
dataset.merge_samples(test_dataset)

# Set dataset name
dataset.name = "open_images_v7_urban_segmentation"

print(f"\nFinal dataset statistics:")
print(f"  Total samples: {len(dataset)}")
print(f"  Dataset name: {dataset.name}")

# Launch FiftyOne app to visualize the dataset
print(f"\nLaunching FiftyOne App...")
session = fo.launch_app(dataset)

# Print summary
print(f"\nDataset download completed!")
print(f"Classes used: {len(target_classes)} urban/architectural categories")
print(f"Seed used: {seed}")
print(f"Dataset saved as: {dataset.name}")
print(f"\nFiles downloaded to:")
print(f"  Train split: {train_dir}")
print(f"  Validation split: {val_dir}")
print(f"  Test split: {test_dir}")
print(f"\nAccess the dataset via FiftyOne App or programmatically using:")
print(f"  dataset = fo.load_dataset('{dataset.name}')")