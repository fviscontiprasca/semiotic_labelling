"""
Specialized data preparation for full Flux.1d fine-tuning.
Optimized for full model training with enhanced semiotic conditioning,
quality filtering, and memory-efficient data loading.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import fiftyone as fo
from datasets import Dataset
import hashlib
import re
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FullTrainingDataConfig:
    """Configuration for full fine-tuning data preparation."""
    
    # Quality thresholds - stricter for full fine-tuning
    min_semiotic_score: float = 0.5
    min_image_size: Tuple[int, int] = (768, 768)
    max_image_size: Tuple[int, int] = (2048, 2048)
    target_resolution: int = 1024
    min_caption_length: int = 20
    max_caption_length: int = 300
    
    # Data augmentation for full fine-tuning
    use_caption_augmentation: bool = True
    use_semiotic_token_augmentation: bool = True
    augmentation_probability: float = 0.3
    
    # Memory optimization
    use_webp_compression: bool = True
    webp_quality: int = 90
    precompute_latents: bool = False  # For very large datasets
    
    # Enhanced semiotic conditioning
    require_architectural_style: bool = True
    require_urban_mood: bool = True
    enhanced_prompt_templates: bool = True

class FullFluxTrainingDataPreparator:
    """Enhanced data preparation for full Flux.1d fine-tuning."""
    
    def __init__(self, base_path: str, output_path: str = None, config: FullTrainingDataConfig = None):
        """Initialize enhanced data preparation system."""
        
        self.base_path = Path(base_path)
        # Align with full fine-tuning outputs structure
        default_output = self.base_path / "data" / "outputs" / "05_flux_full_training_data"
        self.output_path = Path(output_path) if output_path else default_output
        
        self.config = config or FullTrainingDataConfig()
        
        # Create output directory structure
        self.images_dir = self.output_path / "images"
        self.captions_dir = self.output_path / "captions"
        self.metadata_dir = self.output_path / "metadata"
        
        for dir_path in [self.images_dir, self.captions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced semiotic prompt templates for full fine-tuning
        self.enhanced_prompt_templates = {
            "comprehensive_architectural": (
                "A {style} architectural scene featuring {elements}, "
                "constructed with {materials}, conveying {mood} atmosphere, "
                "photographed during {time_period}, emphasizing {symbolic_meaning} "
                "through {spatial_qualities} and {lighting_qualities}"
            ),
            "style_focused": (
                "{style} architecture, {mood} urban environment, "
                "showcasing {materials} construction, {lighting_qualities} lighting"
            ),
            "contextual": (
                "Urban architectural photography of {style} buildings in {cultural_context}, "
                "creating {mood} atmosphere through {design_approach} and {material_palette}"
            ),
            "semiotic_rich": (
                "Semiotic architectural analysis: {style} design conveying {symbolic_meaning}, "
                "{mood} spatial experience, {materials} material expression, "
                "{time_period} environmental context"
            ),
            "professional": (
                "Professional architectural photography: {style} {typology}, "
                "{mood} lighting conditions, {materials} facade treatment, "
                "award-winning composition"
            )
        }
        
        # Semiotic token vocabulary for augmentation
        self.semiotic_vocabulary = {
            "architectural_styles": [
                "modernist", "brutalist", "postmodern", "minimalist", "baroque", 
                "gothic", "industrial", "art_deco", "neoclassical", "bauhaus",
                "deconstructivist", "high_tech", "organic", "vernacular"
            ],
            "urban_moods": [
                "contemplative", "vibrant", "tense", "serene", "dramatic", 
                "peaceful", "energetic", "melancholic", "imposing", "intimate",
                "mysterious", "welcoming", "austere", "luxurious"
            ],
            "materials": [
                "concrete", "glass", "steel", "stone", "brick", "wood",
                "limestone", "marble", "aluminum", "copper", "granite",
                "sandstone", "terra_cotta", "ceramic"
            ],
            "lighting_qualities": [
                "dramatic", "soft", "harsh", "warm", "cool", "diffused",
                "directional", "atmospheric", "golden", "ethereal"
            ],
            "spatial_qualities": [
                "monumental", "intimate", "expansive", "compressed", "hierarchical",
                "flowing", "rigid", "organic", "geometric", "layered"
            ]
        }
        
        # Quality filters - stricter for full fine-tuning
        self.quality_thresholds = {
            "min_semiotic_score": self.config.min_semiotic_score,
            "min_image_size": self.config.min_image_size,
            "max_image_size": self.config.max_image_size,
            "min_caption_length": self.config.min_caption_length,
            "max_caption_length": self.config.max_caption_length
        }
        
        logger.info(f"Enhanced Flux full fine-tuning data preparation initialized at {self.output_path}")
    
    def prepare_full_training_data(self, dataset: fo.Dataset, 
                                  split_ratios: Tuple[float, float, float] = (0.85, 0.1, 0.05),
                                  max_samples: int = None) -> Dict[str, Any]:
        """Prepare enhanced training dataset for full Flux.1d fine-tuning."""
        
        logger.info(f"Preparing full fine-tuning data from {len(dataset)} samples")
        
        # Enhanced quality filtering
        filtered_dataset = self._enhanced_quality_filter(dataset)
        logger.info(f"Filtered to {len(filtered_dataset)} high-quality samples")
        
        # Limit samples if specified
        if max_samples and len(filtered_dataset) > max_samples:
            filtered_dataset = filtered_dataset.take(max_samples, seed=42)
            logger.info(f"Limited to {max_samples} samples")
        
        # Enhanced split for full fine-tuning (more training data)
        train_samples, val_samples, test_samples = filtered_dataset.random_split(
            split_ratios, seed=42
        )
        
        # Prepare each split with enhanced processing
        splits_info = {}
        
        for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            if len(samples) > 0:
                split_info = self._prepare_enhanced_split(samples, split_name)
                splits_info[split_name] = split_info
        
        # Create enhanced training configuration
        training_config = self._create_full_training_config(splits_info)
        
        # Save enhanced dataset metadata
        self._save_enhanced_metadata(splits_info, training_config)
        
        logger.info("Full fine-tuning data preparation completed successfully")
        return training_config
    
    def _enhanced_quality_filter(self, dataset: fo.Dataset) -> fo.Dataset:
        """Enhanced quality filtering for full fine-tuning."""
        
        filtered_samples = []
        stats = {
            "total": 0,
            "passed_image_quality": 0,
            "passed_semiotic_quality": 0,
            "passed_caption_quality": 0,
            "final_count": 0
        }
        
        for sample in dataset.iter_samples(progress=True):
            stats["total"] += 1
            
            try:
                # Enhanced image quality check
                if not self._enhanced_image_quality_check(sample.filepath):
                    continue
                stats["passed_image_quality"] += 1
                
                # Stricter semiotic quality check
                if not self._enhanced_semiotic_quality_check(sample):
                    continue
                stats["passed_semiotic_quality"] += 1
                
                # Enhanced caption quality check
                if not self._enhanced_caption_quality_check(sample):
                    continue
                stats["passed_caption_quality"] += 1
                
                # Additional checks for full fine-tuning
                if self.config.require_architectural_style and not self._has_architectural_style(sample):
                    continue
                
                if self.config.require_urban_mood and not self._has_urban_mood(sample):
                    continue
                
                filtered_samples.append(sample)
                stats["final_count"] += 1
                
            except Exception as e:
                logger.warning(f"Error checking sample {sample.filepath}: {e}")
                continue
        
        # Log filtering statistics
        logger.info(f"Quality filtering results: {stats}")
        
        # Create new dataset with filtered samples
        filtered_dataset = fo.Dataset()
        filtered_dataset.add_samples(filtered_samples)
        
        return filtered_dataset
    
    def _enhanced_image_quality_check(self, image_path: str) -> bool:
        """Enhanced image quality check for full fine-tuning."""
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Stricter size requirements
                if width < self.config.min_image_size[0] or height < self.config.min_image_size[1]:
                    return False
                if width > self.config.max_image_size[0] or height > self.config.max_image_size[1]:
                    return False
                
                # Check aspect ratio (prefer square-ish for architectural images)
                aspect_ratio = width / height
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    return False
                
                # Check for corruption
                img.verify()
                
                # Reopen for additional checks
                with Image.open(image_path) as img_check:
                    # Check if image is not too dark or too bright
                    if img_check.mode == "RGB":
                        img_array = np.array(img_check)
                        mean_brightness = np.mean(img_array)
                        if mean_brightness < 30 or mean_brightness > 225:
                            return False
                
                return True
                
        except Exception as e:
            logger.debug(f"Image quality check failed for {image_path}: {e}")
            return False
    
    def _enhanced_semiotic_quality_check(self, sample: fo.Sample) -> bool:
        """Enhanced semiotic quality check for full fine-tuning."""
        
        # Check for semiotic features
        if not hasattr(sample, "semiotic_features"):
            return False
        
        semiotic_features = sample.semiotic_features
        if not isinstance(semiotic_features, dict):
            return False
        
        # Stricter semiotic score threshold
        semiotic_score = semiotic_features.get("semiotic_score", 0)
        if semiotic_score < self.config.min_semiotic_score:
            return False
        
        # Check for multiple semiotic elements
        required_elements = ["architectural_style", "urban_mood", "materials"]
        present_elements = sum(1 for element in required_elements if semiotic_features.get(element))
        
        if present_elements < 2:  # Require at least 2 out of 3
            return False
        
        return True
    
    def _enhanced_caption_quality_check(self, sample: fo.Sample) -> bool:
        """Enhanced caption quality check for full fine-tuning."""
        
        caption_sources = ["original_prompt", "semiotic_captions", "enhanced_semiotic_caption"]
        
        best_caption = ""
        best_score = 0
        
        for source in caption_sources:
            if hasattr(sample, source) and sample[source]:
                caption = sample[source]
                
                if isinstance(caption, str):
                    score = self._score_caption_quality(caption)
                    if score > best_score:
                        best_caption = caption
                        best_score = score
                        
                elif isinstance(caption, dict):
                    for cap in caption.values():
                        if isinstance(cap, str):
                            score = self._score_caption_quality(cap)
                            if score > best_score:
                                best_caption = cap
                                best_score = score
        
        # Stricter requirements for full fine-tuning
        return (len(best_caption) >= self.config.min_caption_length and 
                len(best_caption) <= self.config.max_caption_length and
                best_score >= 0.7)
    
    def _score_caption_quality(self, caption: str) -> float:
        """Score caption quality for full fine-tuning."""
        
        score = 0.0
        caption_lower = caption.lower()
        
        # Architectural vocabulary
        arch_terms = ["building", "architecture", "design", "structure", "facade", "interior", "urban"]
        arch_score = sum(1 for term in arch_terms if term in caption_lower) / len(arch_terms)
        score += arch_score * 0.3
        
        # Semiotic richness
        semiotic_terms = ["style", "mood", "atmosphere", "material", "lighting", "context", "meaning"]
        semiotic_score = sum(1 for term in semiotic_terms if term in caption_lower) / len(semiotic_terms)
        score += semiotic_score * 0.4
        
        # Descriptive richness
        descriptive_terms = ["featuring", "showcasing", "characterized", "emphasizing", "conveying"]
        desc_score = sum(1 for term in descriptive_terms if term in caption_lower) / len(descriptive_terms)
        score += desc_score * 0.3
        
        return min(score, 1.0)
    
    def _has_architectural_style(self, sample: fo.Sample) -> bool:
        """Check if sample has identifiable architectural style."""
        
        if hasattr(sample, "semiotic_features") and sample.semiotic_features:
            style = sample.semiotic_features.get("architectural_style")
            return style and style in self.semiotic_vocabulary["architectural_styles"]
        return False
    
    def _has_urban_mood(self, sample: fo.Sample) -> bool:
        """Check if sample has identifiable urban mood."""
        
        if hasattr(sample, "semiotic_features") and sample.semiotic_features:
            mood = sample.semiotic_features.get("urban_mood")
            return mood and mood in self.semiotic_vocabulary["urban_moods"]
        return False
    
    def _prepare_enhanced_split(self, samples: fo.Dataset, split_name: str) -> Dict[str, Any]:
        """Prepare enhanced data split for full fine-tuning."""
        
        logger.info(f"Preparing enhanced {split_name} split with {len(samples)} samples")
        
        split_dir = self.images_dir / split_name
        split_captions_dir = self.captions_dir / split_name
        
        split_dir.mkdir(exist_ok=True)
        split_captions_dir.mkdir(exist_ok=True)
        
        processed_samples = []
        sample_metadata = []
        
        for i, sample in enumerate(samples.iter_samples(progress=True)):
            try:
                # Generate unique filename
                file_hash = hashlib.md5(sample.filepath.encode()).hexdigest()[:8]
                filename = f"{split_name}_{i:06d}_{file_hash}"
                
                # Enhanced image processing
                image_info = self._process_enhanced_image(sample, split_dir, filename)
                
                # Generate multiple enhanced captions for full fine-tuning
                captions = self._generate_multiple_enhanced_captions(sample)
                
                # Save enhanced captions
                caption_info = self._save_enhanced_captions(captions, split_captions_dir, filename)
                
                # Collect enhanced metadata
                metadata = self._collect_enhanced_metadata(sample, image_info, caption_info)
                sample_metadata.append(metadata)
                
                processed_samples.append({
                    "image_path": str(image_info["output_path"]),
                    "caption_path": str(caption_info["primary_caption_path"]),
                    "enhanced_captions": caption_info["all_captions"],
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Save enhanced split metadata
        split_metadata_path = self.metadata_dir / f"{split_name}_enhanced_metadata.json"
        with open(split_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
        
        split_info = {
            "split_name": split_name,
            "sample_count": len(processed_samples),
            "samples": processed_samples,
            "metadata_path": str(split_metadata_path),
            "enhanced_features": True
        }
        
        logger.info(f"Completed enhanced {split_name} split preparation: {len(processed_samples)} samples")
        return split_info
    
    def _process_enhanced_image(self, sample: fo.Sample, output_dir: Path, filename: str) -> Dict[str, Any]:
        """Enhanced image processing for full fine-tuning."""
        
        original_path = Path(sample.filepath)
        
        # Choose output format based on config
        if self.config.use_webp_compression:
            output_path = output_dir / f"{filename}.webp"
            output_format = "WEBP"
        else:
            output_path = output_dir / f"{filename}.jpg"
            output_format = "JPEG"
        
        # Load and enhance image
        with Image.open(original_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize to target resolution maintaining aspect ratio
            target_size = self.config.target_resolution
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Center crop to square if needed for full fine-tuning
            if img.size[0] != img.size[1]:
                min_dim = min(img.size)
                left = (img.size[0] - min_dim) // 2
                top = (img.size[1] - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Save with high quality
            save_kwargs = {"optimize": True}
            if output_format == "WEBP":
                save_kwargs["quality"] = self.config.webp_quality
                save_kwargs["method"] = 6  # Best compression
            else:
                save_kwargs["quality"] = 95
            
            img.save(output_path, output_format, **save_kwargs)
        
        return {
            "original_path": str(original_path),
            "output_path": output_path,
            "size": (target_size, target_size),
            "format": output_format,
            "enhanced": True
        }
    
    def _generate_multiple_enhanced_captions(self, sample: fo.Sample) -> Dict[str, str]:
        """Generate multiple enhanced captions for full fine-tuning diversity."""
        
        captions = {}
        
        # Extract enhanced semiotic features
        semiotic_features = sample.semiotic_features if hasattr(sample, "semiotic_features") else {}
        
        # Base information extraction
        base_info = self._extract_enhanced_base_info(sample, semiotic_features)
        
        # Generate captions using different templates
        if self.config.enhanced_prompt_templates:
            for template_name, template in self.enhanced_prompt_templates.items():
                try:
                    caption = template.format(**base_info)
                    captions[template_name] = self._clean_and_enhance_caption(caption)
                except KeyError as e:
                    # Fallback for missing keys
                    simplified_caption = self._generate_fallback_caption(sample, semiotic_features)
                    captions[template_name] = simplified_caption
        
        # Generate augmented versions for training diversity
        if self.config.use_caption_augmentation:
            primary_caption = captions.get("comprehensive_architectural", list(captions.values())[0])
            augmented_captions = self._generate_caption_augmentations(primary_caption, semiotic_features)
            captions.update(augmented_captions)
        
        return captions
    
    def _extract_enhanced_base_info(self, sample: fo.Sample, semiotic_features: Dict[str, Any]) -> Dict[str, str]:
        """Extract enhanced base information for caption templates."""
        
        # Default values
        info = {
            "style": "contemporary",
            "elements": "architectural elements",
            "materials": "mixed materials",
            "mood": "neutral",
            "time_period": "daytime",
            "symbolic_meaning": "functional design",
            "spatial_qualities": "structured space",
            "lighting_qualities": "natural lighting",
            "cultural_context": "urban environment",  
            "design_approach": "modern design",
            "material_palette": "diverse materials",
            "typology": "building"
        }
        
        # Fill from semiotic features
        if "architectural_style" in semiotic_features and semiotic_features["architectural_style"]:
            info["style"] = semiotic_features["architectural_style"]
        
        if "urban_mood" in semiotic_features and semiotic_features["urban_mood"]:
            info["mood"] = semiotic_features["urban_mood"]
        
        if "materials" in semiotic_features and semiotic_features["materials"]:
            materials = semiotic_features["materials"]
            if isinstance(materials, list) and materials:
                info["materials"] = " and ".join(materials[:3])
                info["material_palette"] = ", ".join(materials[:2])
        
        if "time_period" in semiotic_features and semiotic_features["time_period"]:
            info["time_period"] = semiotic_features["time_period"]
        
        if "cultural_context" in semiotic_features and semiotic_features["cultural_context"]:
            info["cultural_context"] = semiotic_features["cultural_context"]
        
        if "symbolic_meaning" in semiotic_features and semiotic_features["symbolic_meaning"]:
            info["symbolic_meaning"] = semiotic_features["symbolic_meaning"]
        
        if "spatial_hierarchy" in semiotic_features and semiotic_features["spatial_hierarchy"]:
            info["spatial_qualities"] = semiotic_features["spatial_hierarchy"]
        
        # Enhance with architectural vocabulary
        info["elements"] = f"{info['style']} architectural elements"
        info["design_approach"] = f"{info['style']} design methodology"
        
        return info
    
    def _generate_caption_augmentations(self, base_caption: str, semiotic_features: Dict[str, Any]) -> Dict[str, str]:
        """Generate caption augmentations for training diversity."""
        
        augmentations = {}
        
        if not self.config.use_semiotic_token_augmentation:
            return augmentations
        
        import random
        
        # Token-augmented version
        if random.random() < self.config.augmentation_probability:
            tokens = self._generate_semiotic_tokens(semiotic_features)
            if tokens:
                augmentations["token_augmented"] = f"{' '.join(tokens)} {base_caption}"
        
        # Style-emphasized version
        if "architectural_style" in semiotic_features:
            style = semiotic_features["architectural_style"]
            augmentations["style_emphasized"] = f"<{style}> architectural style, {base_caption}"
        
        # Mood-emphasized version
        if "urban_mood" in semiotic_features:
            mood = semiotic_features["urban_mood"]
            augmentations["mood_emphasized"] = f"<{mood}> atmosphere, {base_caption}"
        
        return augmentations
    
    def _generate_semiotic_tokens(self, semiotic_features: Dict[str, Any]) -> List[str]:
        """Generate semiotic conditioning tokens."""
        
        tokens = []
        
        if "architectural_style" in semiotic_features and semiotic_features["architectural_style"]:
            tokens.append(f"<{semiotic_features['architectural_style']}>")
        
        if "urban_mood" in semiotic_features and semiotic_features["urban_mood"]:
            tokens.append(f"<{semiotic_features['urban_mood']}>")
        
        if "materials" in semiotic_features and semiotic_features["materials"]:
            materials = semiotic_features["materials"]
            if isinstance(materials, list):
                for material in materials[:2]:  # Limit tokens
                    tokens.append(f"<{material}>")
        
        return tokens
    
    def _generate_fallback_caption(self, sample: fo.Sample, semiotic_features: Dict[str, Any]) -> str:
        """Generate fallback caption when template formatting fails."""
        
        # Try to get original caption first
        if hasattr(sample, "original_prompt") and sample.original_prompt:
            base = sample.original_prompt
        elif hasattr(sample, "semiotic_captions") and sample.semiotic_captions:
            if isinstance(sample.semiotic_captions, dict):
                base = list(sample.semiotic_captions.values())[0]
            else:
                base = str(sample.semiotic_captions)
        else:
            base = "architectural scene"
        
        # Enhance with available semiotic info
        enhancements = []
        
        if "architectural_style" in semiotic_features:
            enhancements.append(f"{semiotic_features['architectural_style']} style")
        
        if "urban_mood" in semiotic_features:
            enhancements.append(f"{semiotic_features['urban_mood']} atmosphere")
        
        if enhancements:
            return f"{base}, featuring {', '.join(enhancements)}"
        else:
            return f"Architectural photography: {base}"
    
    def _clean_and_enhance_caption(self, caption: str) -> str:
        """Clean and enhance caption for full fine-tuning."""
        
        # Basic cleaning
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Ensure proper capitalization
        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        # Add quality enhancer for full fine-tuning
        if not any(term in caption.lower() for term in ["professional", "award", "high quality"]):
            caption = f"Professional architectural photography: {caption}"
        
        # Ensure proper ending
        if not caption.endswith('.'):
            caption += '.'
        
        return caption
    
    def _save_enhanced_captions(self, captions: Dict[str, str], output_dir: Path, filename: str) -> Dict[str, Any]:
        """Save enhanced captions for full fine-tuning."""
        
        # Save primary caption (comprehensive_architectural or first available)
        primary_key = "comprehensive_architectural" if "comprehensive_architectural" in captions else list(captions.keys())[0]
        primary_caption = captions[primary_key]
        
        primary_path = output_dir / f"{filename}.txt"
        with open(primary_path, 'w', encoding='utf-8') as f:
            f.write(primary_caption)
        
        # Save all captions as JSON for training variety
        all_captions_path = output_dir / f"{filename}_all.json"
        with open(all_captions_path, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        
        return {
            "primary_caption_path": primary_path,
            "all_captions_path": all_captions_path,
            "all_captions": captions,
            "primary_caption": primary_caption
        }
    
    def _collect_enhanced_metadata(self, sample: fo.Sample, image_info: Dict[str, Any], caption_info: Dict[str, Any]) -> Dict[str, Any]:
        """Collect enhanced metadata for full fine-tuning."""
        
        metadata = {
            "original_filepath": sample.filepath,
            "processed_image": str(image_info["output_path"]),
            "image_format": image_info["format"],
            "enhanced_processing": True,
            "primary_caption": caption_info["primary_caption"],
            "caption_variants": len(caption_info["all_captions"]),
            "semiotic_features": {},
            "quality_scores": {}
        }
        
        # Include semiotic features
        if hasattr(sample, "semiotic_features") and sample.semiotic_features:
            metadata["semiotic_features"] = sample.semiotic_features
        
        # Calculate quality scores
        metadata["quality_scores"] = {
            "caption_quality": self._score_caption_quality(caption_info["primary_caption"]),
            "semiotic_richness": len(metadata["semiotic_features"]) / 10.0,  # Normalize
            "enhancement_level": "full_finetune"
        }
        
        return metadata
    
    def _create_full_training_config(self, splits_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced training configuration for full fine-tuning."""
        
        config = {
            "pipeline_type": "full_flux_finetuning",
            "data_preparation": {
                "enhanced_quality_filtering": True,
                "semiotic_conditioning": True,
                "caption_augmentation": self.config.use_caption_augmentation,
                "target_resolution": self.config.target_resolution,
                "quality_thresholds": self.quality_thresholds
            },
            "splits": splits_info,
            "total_samples": sum(split_info["sample_count"] for split_info in splits_info.values()),
            "recommended_training_params": {
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 5e-6,
                "num_epochs": 5,
                "warmup_steps": 500,
                "gradient_checkpointing": True,
                "use_ema": True
            },
            "semiotic_features": {
                "architectural_styles": self.semiotic_vocabulary["architectural_styles"],
                "urban_moods": self.semiotic_vocabulary["urban_moods"],
                "materials": self.semiotic_vocabulary["materials"]
            }
        }
        
        return config
    
    def _save_enhanced_metadata(self, splits_info: Dict[str, Any], training_config: Dict[str, Any]):
        """Save enhanced metadata for full fine-tuning."""
        
        # Save training config
        config_path = self.metadata_dir / "full_training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        # Save dataset statistics
        stats = {
            "preparation_type": "full_flux_finetuning",
            "total_samples": training_config["total_samples"],
            "splits": {name: info["sample_count"] for name, info in splits_info.items()},
            "quality_filtering": {
                "min_semiotic_score": self.config.min_semiotic_score,
                "required_architectural_style": self.config.require_architectural_style,
                "required_urban_mood": self.config.require_urban_mood,
                "enhanced_processing": True
            },
            "semiotic_vocabulary_size": {
                name: len(vocab) for name, vocab in self.semiotic_vocabulary.items()
            }
        }
        
        stats_path = self.metadata_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced metadata saved to {self.metadata_dir}")

def main():
    """Main function for enhanced data preparation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Flux Full Fine-tuning Data Preparation")
    parser.add_argument("--base_path", type=str, default=".", help="Base path to project")
    parser.add_argument("--output_path", type=str, help="Output path for prepared data")
    parser.add_argument("--dataset_name", type=str, default="semiotic_urban_combined", help="FiftyOne dataset name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples")
    parser.add_argument("--min_semiotic_score", type=float, default=0.5, help="Minimum semiotic score")
    parser.add_argument("--target_resolution", type=int, default=1024, help="Target image resolution")
    parser.add_argument("--require_style", action="store_true", help="Require architectural style")
    parser.add_argument("--require_mood", action="store_true", help="Require urban mood")
    
    args = parser.parse_args()
    
    # Create config
    config = FullTrainingDataConfig(
        min_semiotic_score=args.min_semiotic_score,
        target_resolution=args.target_resolution,
        require_architectural_style=args.require_style,
        require_urban_mood=args.require_mood
    )
    
    # Initialize preparator
    preparator = FullFluxTrainingDataPreparator(
        base_path=args.base_path,
        output_path=args.output_path,
        config=config
    )
    
    # Load dataset
    if fo.dataset_exists(args.dataset_name):
        dataset = fo.load_dataset(args.dataset_name)
        logger.info(f"Loaded dataset '{args.dataset_name}' with {len(dataset)} samples")
    else:
        logger.error(f"Dataset '{args.dataset_name}' not found. Run 01_data_pipeline.py first.")
        return
    
    # Prepare training data
    training_config = preparator.prepare_full_training_data(
        dataset=dataset,
        max_samples=args.max_samples
    )
    
    logger.info("Enhanced full fine-tuning data preparation completed!")
    logger.info(f"Training samples: {training_config['splits']['train']['sample_count']}")
    logger.info(f"Validation samples: {training_config['splits']['val']['sample_count']}")
    logger.info(f"Test samples: {training_config['splits']['test']['sample_count']}")

if __name__ == "__main__":
    main()