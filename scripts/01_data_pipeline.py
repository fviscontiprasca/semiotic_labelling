"""
Unified data pipeline for combining OpenImages v7 urban classes with imaginary synthetic dataset
for semiotic-aware Flux.1d fine-tuning.
"""

import os
import json
import pandas as pd
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.labels as fol
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
        self.export_path = self.base_path / "data" / "outputs" / "01_data_pipeline"
        
        # Load urban classes for OID
        with open(self.base_path / "my_classes_final.txt", "r") as f:
            self.urban_classes = [line.strip() for line in f.readlines()]
        
        logger.info(f"Initialized pipeline with {len(self.urban_classes)} urban classes")
        
    def create_unified_dataset(self, dataset_name: str = "semiotic_urban_combined", 
                             max_oid_samples: int = 1000) -> fo.Dataset:
        """Create a unified FiftyOne dataset combining OID and synthetic data."""
        
        # Check if dataset exists and delete it to ensure fresh creation
        if fo.dataset_exists(dataset_name):
            logger.info(f"Deleting existing dataset: {dataset_name}")
            existing_dataset = fo.load_dataset(dataset_name)
            existing_dataset.delete()
        
        logger.info(f"Creating new dataset: {dataset_name}")
        dataset = fo.Dataset(dataset_name)
        
        # Add imaginary synthetic images first (faster)
        synthetic_samples = self._load_synthetic_data()
        if synthetic_samples:
            dataset.add_samples(synthetic_samples)
            logger.info(f"Added {len(synthetic_samples)} synthetic samples")
        
        # Add OID urban images - be patient and persistent
        logger.info("Starting OID data loading - this process requires patience...")
        try:
            oid_samples = self._load_oid_data(max_samples=max_oid_samples)
            if oid_samples:
                dataset.add_samples(oid_samples)
                logger.info(f"✅ Successfully added {len(oid_samples)} OID samples")
            else:
                logger.warning("No OID samples were loaded")
        except KeyboardInterrupt:
            logger.warning("OID loading interrupted by user (Ctrl+C)")
            logger.info("Continuing with synthetic data only")
        except Exception as e:
            logger.error(f"OID loading failed with error: {e}")
            logger.info("Continuing with synthetic data only")
            import traceback
            logger.debug(f"Full error traceback: {traceback.format_exc()}")
        
        # Add metadata and tags
        dataset.compute_metadata()
        
        logger.info(f"Created unified dataset with {dataset.count()} total samples")
        return dataset
    
    def _load_oid_data(self, max_samples: int = 1000) -> List[fo.Sample]:
        """Load OpenImages v7 data for urban classes - be patient and use existing data if available."""
        import time
        
        try:
            # Check for existing FiftyOne OID data
            fiftyone_oid_path = self.oid_path / "fiftyone"
            
            logger.info(f"Loading {max_samples} samples for classes: {self.urban_classes}")
            
            if fiftyone_oid_path.exists():
                logger.info(f"Found existing FiftyOne OID data at: {fiftyone_oid_path}")
                logger.info("Using existing data to avoid long download times...")
                
                # Load without dataset_dir to avoid parameter conflict
                oid_dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split="train",
                    label_types=["detections"],
                    classes=self.urban_classes,
                    max_samples=max_samples,
                    shuffle=True,
                    dataset_name="temp_oid_urban"
                )
            else:
                logger.info("No existing FiftyOne data found. Starting fresh download...")
                logger.info("This may take 10-30 minutes depending on network speed. Please be patient...")
                
                # Ensure destination directory exists
                oid_images_dir = self.oid_path / "images"
                oid_images_dir.mkdir(parents=True, exist_ok=True)
                
                start_time = time.time()
                
                # Load OID dataset through FiftyOne zoo - be patient
                oid_dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split="train",
                    label_types=["detections"],
                    classes=self.urban_classes,
                    max_samples=max_samples,
                    shuffle=True,
                    dataset_name="temp_oid_urban"
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"OID dataset loaded successfully in {elapsed_time:.1f} seconds")
            
            samples = []
            for sample in oid_dataset:
                # Add metadata for semiotic analysis - avoid None values
                sample["source"] = "openimages_v7"
                sample["semiotic_type"] = "real_urban"
                sample["architectural_style"] = "unknown"  # To be filled by analysis
                sample["urban_mood"] = "neutral"  # To be filled by analysis
                
                # Convert to list for return
                samples.append(sample)
            
            # Clean up temporary dataset
            oid_dataset.delete()
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading OID data: {e}")
            return []
    
    def _load_synthetic_data(self) -> List[fo.Sample]:
        """Load imaginary synthetic dataset - use all available samples."""
        samples = []
        
        try:
            # Load the CSV mapping file
            csv_path = self.imaginary_path / "02 Imaginary Cities - Mapping.csv"
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                return samples
            
            df = pd.read_csv(csv_path)
            logger.info(f"Loading all {len(df)} synthetic samples from CSV")
            
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
        
        # Split dataset using FiftyOne's take/skip method
        try:
            # Pylance can misinfer the return type; explicit int + ignore
            total_samples = int(dataset.count())  # type: ignore[arg-type]
        except Exception:
            total_samples = 0
        train_count = int(total_samples * float(split_ratio[0]))
        
        # Create train and validation views
        train_samples = dataset.take(train_count)
        val_samples = dataset.skip(train_count)
        
        # Export training samples
        self._export_samples(train_samples, train_dir, train_labels, "train")
        self._export_samples(val_samples, val_dir, val_labels, "val")
        
        logger.info(f"Exported {train_samples.count()} training and {val_samples.count()} validation samples")
    
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
                "source": getattr(sample, "source", "unknown"),
                "semiotic_type": getattr(sample, "semiotic_type", "unknown"),
                "architectural_style": getattr(sample, "architectural_style", None),
                "urban_mood": getattr(sample, "urban_mood", None),
                "time_period": getattr(sample, "time_period", None),
                "season": getattr(sample, "season", None),
                "primary_material": getattr(sample, "primary_material", None)
            })
        
        # Save metadata
        metadata_path = label_dir.parent / f"{split}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _create_enhanced_caption(self, sample) -> str:
        """Create enhanced caption for semiotic-aware training."""
        
        # Check if sample has source field
        if hasattr(sample, 'source') and sample.source == "imaginary_synthetic":
            # Use original prompt as base
            base_caption = getattr(sample, 'original_prompt', "")
        else:
            # For OID images, build caption from detected objects on this image
            obj_labels = self._extract_oid_objects(sample, top_k=4)
            if obj_labels:
                base_caption = f"Urban scene with {', '.join(obj_labels)}"
            else:
                base_caption = "Urban scene with architectural elements"
        
        # Add semiotic descriptors
        semiotic_parts = []
        
        arch_style = getattr(sample, "architectural_style", None)
        if arch_style:
            semiotic_parts.append(f"{arch_style} architecture")
        
        mood = getattr(sample, "urban_mood", None)
        if mood:
            semiotic_parts.append(f"{mood} atmosphere")
        
        time_period = getattr(sample, "time_period", None)
        if time_period:
            semiotic_parts.append(f"photographed at {time_period}")
        
        material = getattr(sample, "primary_material", None)
        if material:
            semiotic_parts.append(f"featuring {material} materials")

        season = getattr(sample, "season", None)
        if season:
            semiotic_parts.append(f"in {season}")

        city = getattr(sample, "referenced_city", None)
        if city:
            semiotic_parts.append(f"inspired by {city}")
        
        # Combine base caption with semiotic features
        if semiotic_parts:
            enhanced_caption = f"{base_caption}, {', '.join(semiotic_parts)}"
        else:
            enhanced_caption = base_caption
        
        return enhanced_caption

    def _extract_oid_objects(self, sample: fo.Sample, top_k: int = 4) -> List[str]:
        """Extract top object labels from a sample's detections for OID images.

        Attempts common label fields ('detections', 'ground_truth'), and falls
        back to scanning all fields for a fol.Detections container. Returns a
        list of unique labels, prioritized by highest confidence and frequency.
        """
        labels: List[Tuple[str, float]] = []  # (label, confidence)

        def collect_from(field_name: str) -> None:
            try:
                val = sample[field_name]
            except Exception:
                return
            if isinstance(val, fol.Detections):
                dets_obj = getattr(val, 'detections', None)
                try:
                    dets_list = list(dets_obj) if dets_obj is not None else []  # normalize to list
                except Exception:
                    dets_list = []
                for det in dets_list:
                    if det is None:
                        continue
                    lbl = getattr(det, 'label', None)
                    if not lbl:
                        continue
                    conf = getattr(det, 'confidence', None)
                    labels.append((str(lbl), float(conf) if conf is not None else 0.0))

        # Try common fields first
        for fname in ("detections", "ground_truth", "objects"):
            collect_from(fname)

        # If still empty, scan all fields for a Detections container
        if not labels:
            field_names = list(getattr(sample, 'field_names', []) or [])
            for fname in field_names:
                try:
                    val = sample[fname]
                except Exception:
                    continue
                if isinstance(val, fol.Detections):
                    dets_obj = getattr(val, 'detections', None)
                    try:
                        dets_list = list(dets_obj) if dets_obj is not None else []
                    except Exception:
                        dets_list = []
                    for det in dets_list:
                        if det is None:
                            continue
                        lbl = getattr(det, 'label', None)
                        if not lbl:
                            continue
                        conf = getattr(det, 'confidence', None)
                        labels.append((str(lbl), float(conf) if conf is not None else 0.0))

        if not labels:
            return []

        # Aggregate by label: keep max confidence and count occurrences
        from collections import defaultdict
        max_conf: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for lbl, conf in labels:
            counts[lbl] += 1
            if conf > max_conf[lbl]:
                max_conf[lbl] = conf

        # Rank by (max_conf, count), then alphabetically for stability
        ranked = sorted(max_conf.keys(), key=lambda k: (max_conf[k], counts[k], k), reverse=True)
        # Deduplicate and take top_k
        unique_top = []
        for lbl in ranked:
            if lbl not in unique_top:
                unique_top.append(lbl)
            if len(unique_top) >= top_k:
                break
        return unique_top

def main():
    """Main execution function."""
    base_path = Path(__file__).parent.parent
    pipeline = SemioticDataPipeline(str(base_path))
    
    # Create unified dataset with full synthetic data and more OID samples
    dataset = pipeline.create_unified_dataset(max_oid_samples=1000)
    
    # Export for training
    pipeline.export_for_training(dataset)
    
    print(f"Dataset ready with {dataset.count()} samples")
    print("Data pipeline completed successfully!")
    print(f"Training data exported to: {pipeline.export_path}")
    
    # Optional: Launch FiftyOne app for inspection (uncomment if needed)
    # session = fo.launch_app(dataset, port=5151)
    # print("FiftyOne app launched at http://localhost:5151")
    # session.wait()

if __name__ == "__main__":
    main()