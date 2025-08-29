#!/usr/bin/env python3
"""
Open Images V7 Dataset Subset Creator - Buildings and Cities
============================================================

This script extracts a subset of the Open Images V7 dataset containing only images
that depict buildings and cities based on the dataset's own categories.

Required files (download from Open Images V7):
- class-descriptions-boxable.csv
- train-annotations-bbox.csv (or val/test annotations)
- train-images-boxable.csv (or val/test image lists)

Usage:
    python subset_creation.py --data_dir /path/to/openimages --output_dir /path/to/output
"""

import os
import pandas as pd
import requests
import argparse
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import time
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenImagesSubsetCreator:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Building and city related categories (Open Images V7 label names)
        self.target_categories = [
            "Building",
            "Skyscraper", 
            "House",
            "Apartment building",
            "Office building",
            "Commercial building",
            "Residential building",
            "Castle",
            "Church",
            "Mosque",
            "Temple",
            "Synagogue",
            "Cathedral",
            "Palace",
            "Warehouse",
            "Factory",
            "Stadium",
            "Hospital",
            "School",
            "University",
            "Hotel",
            "Shopping mall",
            "Store",
            "Restaurant",
            "Cafe",
            "Gas station",
            "Fire station",
            "Police station",
            "City",
            "Downtown",
            "Urban area",
            "Street",
            "Road",
            "Sidewalk",
            "Plaza",
            "Town square",
            "Neighbourhood",
            "Suburb",
            "Metropolitan area"
        ]
        
    def load_class_descriptions(self):
        """Load and process class descriptions to find target category IDs."""
        class_desc_file = self.data_dir / "class-descriptions-boxable.csv"
        
        if not class_desc_file.exists():
            logger.error(f"Class descriptions file not found: {class_desc_file}")
            logger.info("Please download class-descriptions-boxable.csv from Open Images V7")
            return None
            
        df_classes = pd.read_csv(class_desc_file, header=None, names=['LabelName', 'DisplayName'])
        
        # Find matching categories (case-insensitive)
        target_label_ids = []
        found_categories = []
        
        for category in self.target_categories:
            matches = df_classes[df_classes['DisplayName'].str.contains(category, case=False, na=False)]
            if not matches.empty:
                target_label_ids.extend(matches['LabelName'].tolist())
                found_categories.extend(matches['DisplayName'].tolist())
                
        logger.info(f"Found {len(target_label_ids)} matching categories:")
        for cat in found_categories:
            logger.info(f"  - {cat}")
            
        return target_label_ids
    
    def extract_image_ids(self, target_label_ids, split='train'):
        """Extract image IDs that contain building/city annotations."""
        annotations_file = self.data_dir / f"{split}-annotations-bbox.csv"
        
        if not annotations_file.exists():
            logger.error(f"Annotations file not found: {annotations_file}")
            logger.info(f"Please download {split}-annotations-bbox.csv from Open Images V7")
            return set()
            
        logger.info(f"Loading {split} annotations...")
        df_annotations = pd.read_csv(annotations_file)
        
        # Filter annotations for target categories
        filtered_annotations = df_annotations[df_annotations['LabelName'].isin(target_label_ids)]
        
        # Get unique image IDs
        image_ids = set(filtered_annotations['ImageID'].unique())
        
        logger.info(f"Found {len(image_ids)} images with building/city annotations in {split} set")
        
        # Save filtered annotations
        output_annotations_file = self.output_dir / f"{split}-annotations-bbox-filtered.csv"
        filtered_annotations.to_csv(output_annotations_file, index=False)
        logger.info(f"Saved filtered annotations to: {output_annotations_file}")
        
        return image_ids
    
    def get_image_urls(self, image_ids, split='train'):
        """Get download URLs for the filtered images."""
        images_file = self.data_dir / f"{split}-images-boxable.csv"
        
        if not images_file.exists():
            logger.error(f"Images file not found: {images_file}")
            logger.info(f"Please download {split}-images-boxable.csv from Open Images V7")
            return {}
            
        logger.info(f"Loading {split} image URLs...")
        df_images = pd.read_csv(images_file)
        
        # Filter for our target images
        filtered_images = df_images[df_images['ImageID'].isin(image_ids)]
        
        # Create URL mapping
        image_url_map = dict(zip(filtered_images['ImageID'], filtered_images['OriginalURL']))
        
        logger.info(f"Found URLs for {len(image_url_map)} images")
        
        return image_url_map
    
    def download_images(self, image_url_map, split='train', max_images=None):
        """Download the filtered images."""
        images_dir = self.output_dir / 'images' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        
        if max_images:
            items = list(image_url_map.items())[:max_images]
            logger.info(f"Downloading first {max_images} images...")
        else:
            items = image_url_map.items()
            logger.info(f"Downloading {len(image_url_map)} images...")
        
        downloaded = 0
        failed = 0
        
        for image_id, url in tqdm(items, desc=f"Downloading {split} images"):
            try:
                # Get file extension from URL
                parsed_url = urlparse(url)
                ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
                
                output_path = images_dir / f"{image_id}{ext}"
                
                # Skip if already downloaded
                if output_path.exists():
                    continue
                
                # Download with timeout and user agent
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                req = urllib.request.Request(url, headers=headers)
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(output_path, 'wb') as f:
                        f.write(response.read())
                
                downloaded += 1
                time.sleep(0.1)  # Be nice to servers
                
            except Exception as e:
                logger.warning(f"Failed to download {image_id}: {str(e)}")
                failed += 1
                continue
        
        logger.info(f"Download complete: {downloaded} successful, {failed} failed")
        
        # Save image list
        image_list_file = self.output_dir / f"{split}_image_list.txt"
        with open(image_list_file, 'w') as f:
            for image_id in image_url_map.keys():
                f.write(f"{image_id}\n")
        
        return downloaded, failed
    
    def create_dataset_info(self, splits_processed):
        """Create a dataset info file."""
        info_file = self.output_dir / "dataset_info.txt"
        
        with open(info_file, 'w') as f:
            f.write("Open Images V7 - Buildings and Cities Subset\n")
            f.write("=" * 45 + "\n\n")
            f.write("Target categories:\n")
            for cat in self.target_categories:
                f.write(f"  - {cat}\n")
            f.write(f"\nSplits processed: {', '.join(splits_processed)}\n")
            f.write(f"Created on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Dataset info saved to: {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract Open Images V7 subset for buildings and cities')
    parser.add_argument('--data_dir', required=True, help='Directory containing Open Images V7 CSV files')
    parser.add_argument('--output_dir', required=True, help='Output directory for subset')
    parser.add_argument('--splits', nargs='+', default=['train'], choices=['train', 'validation', 'test'],
                       help='Dataset splits to process (default: train)')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to download per split')
    parser.add_argument('--download_images', action='store_true', help='Download images (default: only extract IDs)')
    
    args = parser.parse_args()
    
    # Create subset creator
    creator = OpenImagesSubsetCreator(args.data_dir, args.output_dir)
    
    # Load class descriptions and find target categories
    target_label_ids = creator.load_class_descriptions()
    if not target_label_ids:
        return
    
    splits_processed = []
    
    for split in args.splits:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {split} split")
        logger.info(f"{'='*50}")
        
        # Extract image IDs
        image_ids = creator.extract_image_ids(target_label_ids, split)
        
        if not image_ids:
            logger.warning(f"No images found for {split} split")
            continue
            
        if args.download_images:
            # Get image URLs and download
            image_url_map = creator.get_image_urls(image_ids, split)
            if image_url_map:
                creator.download_images(image_url_map, split, args.max_images)
        
        splits_processed.append(split)
    
    # Create dataset info
    if splits_processed:
        creator.create_dataset_info(splits_processed)
        logger.info(f"\nSubset creation complete! Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()