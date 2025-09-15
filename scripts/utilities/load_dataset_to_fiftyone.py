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
    """Load the processed dataset from pipeline outputs into FiftyOne as 'semiocity_urban'."""
    
    # Define paths to pipeline outputs
    base_path = Path("data/outputs")
    images_path = base_path / "01_data_pipeline" / "images"
    captions_file = base_path / "02_blip2_captioner" / "captions.json"
    sam_analysis_path = base_path / "03_sam_segmentation" / "analysis"

    if not images_path.exists():
        logger.error(f"Images path not found: {images_path}")
        return None
    if not captions_file.exists():
        logger.error(f"Captions file not found: {captions_file}")
        return None
    if not sam_analysis_path.exists():
        logger.error(f"SAM analysis path not found: {sam_analysis_path}")
        return None

    # Create dataset
    dataset_name = "semiocity_urban"
    if dataset_name in fo.list_datasets():
        logger.info(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)
    logger.info(f"Creating FiftyOne dataset: {dataset_name}")
    dataset = fo.Dataset(dataset_name)

    # Load captions data
    captions_data = {}
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        logger.info(f"Loaded captions for {len(captions_data)} images")
    except Exception as e:
        logger.warning(f"Could not load captions file: {e}")

    samples = []
    
    # Process train and validation splits
    for split_name in ["train", "val"]:
        split_path = images_path / split_name
        if not split_path.exists():
            logger.warning(f"Split path {split_path} does not exist")
            continue
            
        # Get all image files in this split
        image_files = list(split_path.glob("*.png")) + list(split_path.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} images in {split_name} split")
        
        for image_file in image_files:
            try:
                # Create sample
                sample = fo.Sample(filepath=str(image_file))
                
                # Add basic metadata
                sample["image_id"] = image_file.stem
                sample["filename"] = image_file.name
                sample["split"] = split_name
                sample["source"] = "semiotic_pipeline"
                
                # Add caption data if available
                # Captions are keyed by relative path like "train/train_000000.png"
                caption_key = f"{split_name}/{image_file.name}"
                if caption_key in captions_data:
                    caption_info = captions_data[caption_key]
                    
                    # Add individual caption types
                    sample['architectural_analysis'] = caption_info.get('architectural_analysis', '')
                    sample['mood_atmosphere'] = caption_info.get('mood_atmosphere', '')
                    sample['spatial_relations'] = caption_info.get('spatial_relations', '')
                    sample['materials_textures'] = caption_info.get('materials_textures', '')
                    sample['lighting_time'] = caption_info.get('lighting_time', '')
                    sample['cultural_context'] = caption_info.get('cultural_context', '')
                    sample['semiotic_elements'] = caption_info.get('semiotic_elements', '')
                    sample['unified_caption'] = caption_info.get('unified_caption', '')
                else:
                    logger.warning(f"No captions found for {caption_key}")
                
                # Add SAM segmentation analysis if available
                sam_file = sam_analysis_path / f"{split_name}_{image_file.stem}_sam_analysis.json"
                if sam_file.exists():
                    try:
                        with open(sam_file, 'r', encoding='utf-8') as f:
                            sam_data = json.load(f)
                        
                        # Add SAM analysis metadata
                        sample['total_segments'] = sam_data.get('total_segments', 0)
                        sample['architectural_segments'] = sam_data.get('architectural_segments', 0)
                        
                        # Add segment statistics if available
                        if 'segment_analysis' in sam_data:
                            segment_analysis = sam_data['segment_analysis']
                            sample['segment_count'] = segment_analysis.get('total_segments', 0)
                            sample['architectural_segment_count'] = segment_analysis.get('architectural_count', 0)
                            
                    except Exception as e:
                        logger.warning(f"Could not load SAM analysis for {image_file}: {e}")
                else:
                    logger.warning(f"No SAM analysis found for {sam_file}")
                
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples to dataset '{dataset_name}'")
        dataset.info = {
            "description": "Semiotic urban architectural dataset from processing pipeline",
            "version": "2.0",
            "created_by": "semiotic_labelling_pipeline",
            "total_images": len(samples),
            "splits": list(set([s['split'] for s in samples]))
        }
        dataset.persistent = True
        logger.info(f"Dataset '{dataset_name}' created successfully!")
        logger.info(f"Total samples: {dataset.count()}")
        
        # Print split statistics
        split_counts = {}
        for sample in samples:
            split = sample['split']
            split_counts[split] = split_counts.get(split, 0) + 1
        logger.info(f"Split distribution: {split_counts}")
        
        return dataset
    else:
        logger.error("No samples found to add to dataset")
        return None

def main():
    """Main function to load the dataset."""
    try:
        dataset = load_semiocity_dataset()
        if dataset:
            print(f"\n✅ Successfully loaded dataset 'semiocity_urban' with {dataset.count()} samples")
            print(f"   Dataset available in FiftyOne as: {dataset.name}")
            print(f"   Launch FiftyOne App: fiftyone app launch")
        else:
            print("❌ Failed to load dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

if __name__ == "__main__":
    main()