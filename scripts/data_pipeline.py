"""
Unified data pipeline for combining OpenImages v7 urban classes with imaginary synthetic dataset
for semiotic-aware Flux.1d fine-tuning.
"""

import os
import json
import pandas as pd
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemioticDataPipeline:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.imaginary_path = self.base_path / "data" / "imaginary_synthetic"
        self.oid_path = self.base_path / "data" / "oid_urban"
        self.export_path = self.base_path / "data" / "export"
        
        # Load urban classes for OID
        with open(self.base_path / "my_classes_final.txt", "r") as f:
            self.urban_classes = [line.strip() for line in f.readlines()]
        
        logger.info(f"Initialized pipeline with {len(self.urban_classes)} urban classes")
        
    def create_unified_dataset(self, dataset_name: str = "semiotic_urban_combined", 
                             max_oid_samples: int = 1) -> fo.Dataset:
        """Create a unified FiftyOne dataset combining OID and synthetic data."""
        
        # Check if dataset exists
        if fo.dataset_exists(dataset_name):
            logger.info(f"Loading existing dataset: {dataset_name}")
            return fo.load_dataset(dataset_name)
        
        logger.info(f"Creating new dataset: {dataset_name}")
        dataset = fo.Dataset(dataset_name)
        
        # Add OID urban images
        oid_samples = self._load_oid_data(max_samples=max_oid_samples)
        if oid_samples:
            dataset.add_samples(oid_samples)
            logger.info(f"Added {len(oid_samples)} OID samples")
        
        # Add imaginary synthetic images
        synthetic_samples = self._load_synthetic_data()
        if synthetic_samples:
            dataset.add_samples(synthetic_samples)
            logger.info(f"Added {len(synthetic_samples)} synthetic samples")
        
        # Add metadata and tags
        dataset.compute_metadata()
        
        logger.info(f"Created unified dataset with {len(dataset)} total samples")
        return dataset
    
    def _load_oid_data(self, max_samples: int = 1) -> List[fo.Sample]:
        """Load OpenImages v7 data for urban classes."""
        try:
            # Ensure destination directory exists
            oid_images_dir = self.oid_path / "images"
            oid_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Load OID dataset through FiftyOne zoo with specific destination
            oid_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["segmentations", "detections"],
                classes=self.urban_classes,
                max_samples=max_samples,
                shuffle=True,
                dataset_name="temp_oid_urban",
                dataset_dir=str(oid_images_dir)
            )
            
            samples = []
            for sample in oid_dataset:
                # Add metadata for semiotic analysis
                sample["source"] = "openimages_v7"
                sample["semiotic_type"] = "real_urban"
                sample["architectural_style"] = None  # To be filled by analysis
                sample["urban_mood"] = None  # To be filled by analysis
                
                # Convert to list for return
                samples.append(sample)
            
            # Clean up temporary dataset
            oid_dataset.delete()
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading OID data: {e}")
            return []
    
    def _load_synthetic_data(self) -> List[fo.Sample]:
        """Load imaginary synthetic dataset."""
        samples = []
        
        try:
            # Load the CSV mapping file
            csv_path = self.imaginary_path / "02 Imaginary Cities - Mapping.csv"
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                return samples
            
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                filename = row['filename']
                prompt = row['prompt']
                
                # Check if image exists
                image_path = self.imaginary_path / "images" / filename
                if not image_path.exists():
                    continue
                
                # Load annotation if exists
                annotation_path = self.imaginary_path / "annotations" / f"{filename.split('.')[0]}.json"
                annotation_data = {}
                if annotation_path.exists():
                    with open(annotation_path, 'r') as f:
                        annotation_data = json.load(f)
                
                # Create FiftyOne sample
                sample = fo.Sample(filepath=str(image_path))
                
                # Add metadata
                sample["source"] = "imaginary_synthetic"
                sample["semiotic_type"] = "synthetic_urban"
                sample["original_prompt"] = prompt
                sample["generation_seed"] = annotation_data.get("seed")
                sample["generation_steps"] = annotation_data.get("steps")
                
                # Extract semiotic features from prompt
                semiotic_features = self._extract_semiotic_features(prompt)
                for key, value in semiotic_features.items():
                    sample[key] = value
                
                samples.append(sample)
        
        except Exception as e:
            logger.error(f"Error loading synthetic data: {e}")
        
        logger.info(f"Loaded {len(samples)} synthetic samples")
        return samples
    
    def _extract_semiotic_features(self, prompt: str) -> Dict[str, str]:
        """Extract semiotic features from text prompts."""
        features = {}
        
        # Architectural styles
        architectural_styles = [
            "brutalist", "modernist", "postmodern", "baroque", "gothic", 
            "renaissance", "art deco", "minimalist", "futuristic", "industrial"
        ]
        
        # Urban moods/atmospheres
        urban_moods = [
            "contemplative", "vibrant", "tense", "bustling", "serene",
            "dynamic", "melancholic", "energetic", "peaceful", "chaotic"
        ]
        
        # Time periods
        time_periods = [
            "dawn", "morning", "afternoon", "dusk", "evening", "night",
            "late afternoon", "early morning", "sunset", "sunrise"
        ]
        
        # Seasons
        seasons = ["spring", "summer", "autumn", "winter", "fall"]
        
        # Materials
        materials = [
            "concrete", "glass", "steel", "stone", "brick", "wood",
            "limestone", "marble", "metal", "stucco"
        ]
        
        prompt_lower = prompt.lower()
        
        # Extract features
        features["architectural_style"] = next(
            (style for style in architectural_styles if style in prompt_lower), None
        )
        features["urban_mood"] = next(
            (mood for mood in urban_moods if mood in prompt_lower), None
        )
        features["time_period"] = next(
            (time for time in time_periods if time in prompt_lower), None
        )
        features["season"] = next(
            (season for season in seasons if season in prompt_lower), None
        )
        features["primary_material"] = next(
            (material for material in materials if material in prompt_lower), None
        )
        
        # Extract city if mentioned
        if any(city in prompt_lower for city in ["paris", "venice", "buenos aires", "mumbai", "bristol", "new orleans"]):
            features["referenced_city"] = next(
                (city for city in ["paris", "venice", "buenos aires", "mumbai", "bristol", "new orleans"] 
                 if city in prompt_lower), None
            )
        else:
            features["referenced_city"] = None
        
        return features
    
    def export_for_training(self, dataset: fo.Dataset, split_ratio: Tuple[float, float] = (0.8, 0.2)):
        """Export dataset in format suitable for Flux.1d training."""
        
        # Create export directories
        train_dir = self.export_path / "images" / "train"
        val_dir = self.export_path / "images" / "val"
        train_labels = self.export_path / "labels" / "train"
        val_labels = self.export_path / "labels" / "val"
        
        for dir_path in [train_dir, val_dir, train_labels, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split dataset
        train_samples, val_samples = dataset.random_split([split_ratio[0], split_ratio[1]])
        
        # Export training samples
        self._export_samples(train_samples, train_dir, train_labels, "train")
        self._export_samples(val_samples, val_dir, val_labels, "val")
        
        logger.info(f"Exported {len(train_samples)} training and {len(val_samples)} validation samples")
    
    def _export_samples(self, samples, image_dir: Path, label_dir: Path, split: str):
        """Export samples to training format."""
        metadata = []
        
        for i, sample in enumerate(samples):
            # Copy image
            src_path = Path(sample.filepath)
            dst_path = image_dir / f"{split}_{i:06d}{src_path.suffix}"
            
            # Copy image file
            import shutil
            shutil.copy2(src_path, dst_path)
            
            # Create enhanced caption for Flux training
            caption = self._create_enhanced_caption(sample)
            
            # Save caption as text file
            caption_path = label_dir / f"{split}_{i:06d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Store metadata
            metadata.append({
                "filename": dst_path.name,
                "caption": caption,
                "source": sample.get("source", "unknown"),
                "semiotic_type": sample.get("semiotic_type", "unknown"),
                "architectural_style": sample.get("architectural_style"),
                "urban_mood": sample.get("urban_mood"),
                "time_period": sample.get("time_period"),
                "season": sample.get("season"),
                "primary_material": sample.get("primary_material")
            })
        
        # Save metadata
        metadata_path = label_dir.parent / f"{split}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _create_enhanced_caption(self, sample) -> str:
        """Create enhanced caption for semiotic-aware training."""
        
        if sample.get("source") == "imaginary_synthetic":
            # Use original prompt as base
            base_caption = sample.get("original_prompt", "")
        else:
            # For OID images, create caption from detected objects
            base_caption = f"Urban scene with {', '.join(self.urban_classes)}"
        
        # Add semiotic descriptors
        semiotic_parts = []
        
        if sample.get("architectural_style"):
            semiotic_parts.append(f"{sample['architectural_style']} architecture")
        
        if sample.get("urban_mood"):
            semiotic_parts.append(f"{sample['urban_mood']} atmosphere")
        
        if sample.get("time_period"):
            semiotic_parts.append(f"photographed at {sample['time_period']}")
        
        if sample.get("primary_material"):
            semiotic_parts.append(f"featuring {sample['primary_material']} materials")
        
        # Combine base caption with semiotic features
        if semiotic_parts:
            enhanced_caption = f"{base_caption}, {', '.join(semiotic_parts)}"
        else:
            enhanced_caption = base_caption
        
        return enhanced_caption

def main():
    """Main execution function."""
    base_path = Path(__file__).parent.parent
    pipeline = SemioticDataPipeline(str(base_path))
    
    # Create unified dataset
    dataset = pipeline.create_unified_dataset(max_oid_samples=500)
    
    # Export for training
    pipeline.export_for_training(dataset)
    
    # Launch FiftyOne app for inspection
    session = fo.launch_app(dataset, port=5151)
    print(f"Dataset ready with {len(dataset)} samples")
    print("FiftyOne app launched at http://localhost:5151")
    print("Press Ctrl+C to stop")
    
    try:
        session.wait()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()