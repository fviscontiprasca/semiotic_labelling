import fiftyone as fo
import fiftyone.zoo as foz
import random
import os
from pathlib import Path

# Set random seed for reproducible results
seed = random.randint(1000, 9999)
print(f"Using random seed: {seed}")


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



train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    classes=target_classes,
    max_samples=7,
    seed=seed,
    shuffle=True,
    include_id=True,
    only_matching=True,
    load_hierarchy=True,
)

test_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test",
    classes=target_classes,
    max_samples=1,
    seed=seed+1,
    shuffle=True,
    include_id=True,
    only_matching=True,
    load_hierarchy=True,
)

validation_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    classes=target_classes,
    max_samples=2,
    seed=seed+2,
    shuffle=True,
    include_id=True,
    only_matching=True,
    load_hierarchy=True,
)

# Merge all datasets
dataset = train_dataset.merge(validation_dataset).merge(test_dataset)

# Set dataset name
dataset.name = "open_images_v7_urban_segmentation"

session = fo.launch_app(dataset.view(), port=5151)