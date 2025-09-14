#!/usr/bin/env python3
"""
Load processed dataset into FiftyOne as 'semiocity_urban'
"""

import os
import json
import fiftyone as fo
import fiftyone.core.labels as fol
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_semiocity_dataset():
    """Load the processed dataset into FiftyOne as 'semiocity_urban'."""
    
    # Define new paths for imaginary synthetic data
    images_base_path = Path("data/imaginary_synthetic/images")
    annotations_base_path = Path("data/imaginary_synthetic/annotations")

    if not images_base_path.exists():
        logger.error(f"Images base path not found: {images_base_path}")
        return None
    if not annotations_base_path.exists():
        logger.error(f"Annotations base path not found: {annotations_base_path}")
        return None

    # Create dataset
    dataset_name = "semiocity_urban"
    if dataset_name in fo.list_datasets():
        logger.info(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)
    logger.info(f"Creating FiftyOne dataset: {dataset_name}")
    dataset = fo.Dataset(dataset_name)

    # Gather all image files
    image_files = list(images_base_path.glob("*.png")) + list(images_base_path.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} image files in {images_base_path}")

    samples = []
    for image_file in image_files:
        image_name = image_file.stem
        annotation_file = annotations_base_path / (image_name + ".json")
        metadata = {}
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load annotation for {image_file.name}: {e}")
        else:
            logger.warning(f"No annotation found for {image_file.name}")

        sample = fo.Sample(filepath=str(image_file))
        sample["image_id"] = image_name
        sample["filename"] = image_file.name
        sample["source"] = "imaginary_synthetic"

        # Add all metadata fields from annotation JSON
        for k, v in metadata.items():
            sample[k] = v

        samples.append(sample)

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples to dataset '{dataset_name}'")
        dataset.info = {
            "description": "Semiotic urban architectural dataset (imaginary synthetic)",
            "version": "1.1",
            "created_by": "semiotic_labelling_pipeline"
        }
        dataset.persistent = True
        logger.info(f"Dataset '{dataset_name}' created successfully!")
        logger.info(f"Total samples: {len(dataset)}")
        return dataset
    else:
        logger.error("No samples found to add to dataset")
        return None

def main():
    """Main function to load the dataset."""
    try:
        dataset = load_semiocity_dataset()
        if dataset:
            print(f"\n✅ Successfully loaded dataset 'semiocity_urban' with {len(dataset)} samples")
            print(f"   Dataset available in FiftyOne as: {dataset.name}")
            print(f"   Launch FiftyOne App: fiftyone app launch")
        else:
            print("❌ Failed to load dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

if __name__ == "__main__":
    main()