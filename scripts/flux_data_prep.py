"""
Flux.1d training data preparation system that formats the semiotic-aware dataset
for fine-tuning with enhanced captions and architectural understanding.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxTrainingDataPreparator:
    """Prepare semiotic-aware training data for Flux.1d fine-tuning."""
    
    def __init__(self, base_path: str, output_path: str = None):
        """Initialize data preparation system."""
        
        self.base_path = Path(base_path)
        self.output_path = Path(output_path) if output_path else self.base_path / "data" / "flux_training"
        
        # Create output directory structure
        self.images_dir = self.output_path / "images"
        self.captions_dir = self.output_path / "captions"
        self.metadata_dir = self.output_path / "metadata"
        
        for dir_path in [self.images_dir, self.captions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Semiotic prompt templates for Flux training
        self.prompt_templates = {
            "architectural_style": "A {style} architectural scene with {details}, showcasing {mood} atmosphere",
            "urban_context": "Urban environment featuring {elements}, {style} design, {lighting} lighting, {mood} mood",
            "semiotic_rich": "Architectural photography of {style} buildings in {context}, conveying {meaning} through {materials} and {spatial_qualities}",
            "comprehensive": "{base_description}, {style} architectural style, {mood} atmosphere, {time_period}, featuring {materials}, {spatial_analysis}",
            "minimalist": "{style} {typology} with {mood} character",
            "detailed": "Architectural scene showing {style} design approach, constructed with {materials}, creating {mood} urban atmosphere, photographed during {time_period}, emphasizing {symbolic_meaning}"
        }
        
        # Quality filters
        self.quality_thresholds = {
            "min_semiotic_score": 0.3,
            "min_image_size": (512, 512),
            "max_image_size": (2048, 2048),
            "min_caption_length": 10,
            "max_caption_length": 500
        }
        
        logger.info(f"Initialized Flux training data preparation at {self.output_path}")
    
    def prepare_training_data(self, dataset: fo.Dataset, 
                            split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                            max_samples: int = None) -> Dict[str, Any]:
        """Prepare complete training dataset for Flux.1d."""
        
        logger.info(f"Preparing training data from {len(dataset)} samples")
        
        # Filter high-quality samples
        filtered_dataset = self._filter_quality_samples(dataset)
        logger.info(f"Filtered to {len(filtered_dataset)} high-quality samples")
        
        # Limit samples if specified
        if max_samples and len(filtered_dataset) > max_samples:
            filtered_dataset = filtered_dataset.take(max_samples, seed=42)
            logger.info(f"Limited to {max_samples} samples")
        
        # Split dataset
        train_samples, val_samples, test_samples = filtered_dataset.random_split(
            split_ratios, seed=42
        )
        
        # Prepare each split
        splits_info = {}
        
        for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            if len(samples) > 0:
                split_info = self._prepare_split(samples, split_name)
                splits_info[split_name] = split_info
        
        # Create training configuration
        training_config = self._create_training_config(splits_info)
        
        # Save dataset metadata
        self._save_dataset_metadata(splits_info, training_config)
        
        logger.info("Training data preparation completed successfully")
        return training_config
    
    def _filter_quality_samples(self, dataset: fo.Dataset) -> fo.Dataset:
        """Filter samples based on quality criteria."""
        
        filtered_samples = []
        
        for sample in dataset.iter_samples(progress=True):
            try:
                # Check image quality
                if not self._check_image_quality(sample.filepath):
                    continue
                
                # Check semiotic features quality
                if not self._check_semiotic_quality(sample):
                    continue
                
                # Check caption quality
                if not self._check_caption_quality(sample):
                    continue
                
                filtered_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error checking sample {sample.filepath}: {e}")
                continue
        
        # Create new dataset with filtered samples
        filtered_dataset = fo.Dataset()
        filtered_dataset.add_samples(filtered_samples)
        
        return filtered_dataset
    
    def _check_image_quality(self, image_path: str) -> bool:
        """Check if image meets quality requirements."""
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check size requirements
                if width < self.quality_thresholds["min_image_size"][0]:
                    return False
                if height < self.quality_thresholds["min_image_size"][1]:
                    return False
                if width > self.quality_thresholds["max_image_size"][0]:
                    return False
                if height > self.quality_thresholds["max_image_size"][1]:
                    return False
                
                # Check if image is corrupted
                img.verify()
                
                return True
                
        except Exception:
            return False
    
    def _check_semiotic_quality(self, sample: fo.Sample) -> bool:
        """Check if sample has sufficient semiotic richness."""
        
        # Check for semiotic features
        if not hasattr(sample, "semiotic_features"):
            return False
        
        semiotic_features = sample.semiotic_features
        if not isinstance(semiotic_features, dict):
            return False
        
        # Check semiotic score
        semiotic_score = semiotic_features.get("semiotic_score", 0)
        if semiotic_score < self.quality_thresholds["min_semiotic_score"]:
            return False
        
        # Check for key semiotic elements
        required_elements = ["architectural_style", "urban_mood"]
        for element in required_elements:
            if not semiotic_features.get(element):
                return False
        
        return True
    
    def _check_caption_quality(self, sample: fo.Sample) -> bool:
        """Check if sample has good quality captions."""
        
        # Check for captions
        caption_sources = ["original_prompt", "semiotic_captions", "enhanced_semiotic_caption"]
        
        has_caption = False
        for source in caption_sources:
            if hasattr(sample, source) and sample[source]:
                caption = sample[source]
                if isinstance(caption, str):
                    if (len(caption) >= self.quality_thresholds["min_caption_length"] and 
                        len(caption) <= self.quality_thresholds["max_caption_length"]):
                        has_caption = True
                        break
                elif isinstance(caption, dict):
                    # Check if any caption in dict meets requirements
                    for cap in caption.values():
                        if (isinstance(cap, str) and 
                            len(cap) >= self.quality_thresholds["min_caption_length"] and
                            len(cap) <= self.quality_thresholds["max_caption_length"]):
                            has_caption = True
                            break
        
        return has_caption
    
    def _prepare_split(self, samples: fo.Dataset, split_name: str) -> Dict[str, Any]:
        """Prepare a single data split."""
        
        logger.info(f"Preparing {split_name} split with {len(samples)} samples")
        
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
                
                # Process image
                image_info = self._process_image(sample, split_dir, filename)
                
                # Generate enhanced captions
                captions = self._generate_enhanced_captions(sample)
                
                # Save captions
                caption_info = self._save_captions(captions, split_captions_dir, filename)
                
                # Collect metadata
                metadata = self._collect_sample_metadata(sample, image_info, caption_info)
                sample_metadata.append(metadata)
                
                processed_samples.append({
                    "image_path": str(image_info["output_path"]),
                    "caption_path": str(caption_info["primary_caption_path"]),
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Save split metadata
        split_metadata_path = self.metadata_dir / f"{split_name}_metadata.json"
        with open(split_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
        
        split_info = {
            "split_name": split_name,
            "sample_count": len(processed_samples),
            "samples": processed_samples,
            "metadata_path": str(split_metadata_path)
        }
        
        logger.info(f"Completed {split_name} split preparation: {len(processed_samples)} samples")
        return split_info
    
    def _process_image(self, sample: fo.Sample, output_dir: Path, filename: str) -> Dict[str, Any]:
        """Process and copy image to training directory."""
        
        # Determine output format
        original_path = Path(sample.filepath)
        output_path = output_dir / f"{filename}.jpg"  # Standardize to JPG
        
        # Load and process image
        with Image.open(original_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize if too large
            max_size = self.quality_thresholds["max_image_size"]
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save in high quality
            img.save(output_path, "JPEG", quality=95, optimize=True)
        
        return {
            "original_path": str(original_path),
            "output_path": output_path,
            "size": img.size,
            "format": "JPEG"
        }
    
    def _generate_enhanced_captions(self, sample: fo.Sample) -> Dict[str, str]:
        """Generate multiple enhanced captions for training diversity."""
        
        captions = {}
        
        # Extract semiotic features
        semiotic_features = sample.semiotic_features if hasattr(sample, "semiotic_features") else {}
        
        # Extract base information
        context = self._extract_caption_context(sample, semiotic_features)
        
        # Generate different caption variants
        for template_name, template in self.prompt_templates.items():
            try:
                caption = self._fill_caption_template(template, context)
                captions[template_name] = caption
            except Exception as e:
                logger.warning(f"Error generating {template_name} caption: {e}")
                continue
        
        # Use original prompt if available
        if hasattr(sample, "original_prompt") and sample.original_prompt:
            captions["original"] = sample.original_prompt
        
        # Use BLIP-2 captions if available
        if hasattr(sample, "semiotic_captions") and sample.semiotic_captions:
            blip_captions = sample.semiotic_captions
            if isinstance(blip_captions, dict):
                captions["blip2_unified"] = blip_captions.get("unified_caption", "")
        
        return captions
    
    def _extract_caption_context(self, sample: fo.Sample, semiotic_features: Dict) -> Dict[str, str]:
        """Extract context information for caption generation."""
        
        context = {
            "style": semiotic_features.get("architectural_style", "modern"),
            "mood": semiotic_features.get("urban_mood", "contemplative"),
            "time_period": semiotic_features.get("time_period", "daytime"),
            "season": semiotic_features.get("season", ""),
            "materials": ", ".join(semiotic_features.get("materials", ["concrete", "glass"])),
            "typology": semiotic_features.get("architectural_typology", "building"),
            "cultural_context": semiotic_features.get("cultural_context", "urban"),
            "symbolic_meaning": semiotic_features.get("symbolic_meaning", "architectural expression"),
            "spatial_qualities": "geometric composition",
            "lighting": "natural lighting",
            "elements": "architectural elements",
            "details": "clean lines and modern design",
            "meaning": "urban identity",
            "spatial_analysis": "balanced spatial relationships"
        }
        
        # Create base description
        if hasattr(sample, "original_prompt") and sample.original_prompt:
            # Extract key elements from original prompt
            context["base_description"] = self._extract_base_description(sample.original_prompt)
        else:
            context["base_description"] = f"{context['style']} {context['typology']}"
        
        # Fill empty contexts with defaults
        for key, value in context.items():
            if not value or value == "":
                context[key] = self._get_default_context(key)
        
        return context
    
    def _fill_caption_template(self, template: str, context: Dict[str, str]) -> str:
        """Fill caption template with context information."""
        
        try:
            filled_caption = template.format(**context)
            
            # Clean up the caption
            filled_caption = self._clean_caption(filled_caption)
            
            return filled_caption
            
        except KeyError as e:
            logger.warning(f"Missing context key {e} for template")
            # Return simplified version
            return f"{context.get('style', 'Modern')} architectural scene with {context.get('mood', 'contemplative')} atmosphere"
    
    def _extract_base_description(self, original_prompt: str) -> str:
        """Extract base architectural description from original prompt."""
        
        # Remove generation parameters and clean up
        cleaned = re.sub(r'\b(photorealistic|high quality|street-level|isometric)\b', '', original_prompt, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Take first sentence or first 100 characters
        sentences = cleaned.split('.')
        if len(sentences) > 0:
            base_desc = sentences[0].strip()
            if len(base_desc) > 20:
                return base_desc
        
        return cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and post-process generated captions."""
        
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Remove duplicate words
        words = caption.split()
        cleaned_words = []
        for word in words:
            if not cleaned_words or word.lower() != cleaned_words[-1].lower():
                cleaned_words.append(word)
        
        caption = ' '.join(cleaned_words)
        
        # Ensure proper capitalization
        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        # Ensure proper ending
        if caption and not caption.endswith('.'):
            caption += '.'
        
        return caption
    
    def _get_default_context(self, key: str) -> str:
        """Get default context values."""
        
        defaults = {
            "style": "modern",
            "mood": "contemplative",
            "time_period": "daytime",
            "season": "spring",
            "materials": "concrete and glass",
            "typology": "building",
            "cultural_context": "urban",
            "symbolic_meaning": "architectural expression",
            "spatial_qualities": "geometric",
            "lighting": "natural",
            "elements": "architectural structures",
            "details": "clean design",
            "meaning": "urban character",
            "spatial_analysis": "balanced composition"
        }
        
        return defaults.get(key, "architectural")
    
    def _save_captions(self, captions: Dict[str, str], output_dir: Path, filename: str) -> Dict[str, Any]:
        """Save captions in various formats."""
        
        caption_info = {}
        
        # Save primary caption (comprehensive template)
        primary_caption = captions.get("comprehensive", captions.get("semiotic_rich", captions.get("original", "")))
        primary_path = output_dir / f"{filename}.txt"
        
        with open(primary_path, 'w', encoding='utf-8') as f:
            f.write(primary_caption)
        
        caption_info["primary_caption_path"] = primary_path
        caption_info["primary_caption"] = primary_caption
        
        # Save all captions as JSON
        all_captions_path = output_dir / f"{filename}_all.json"
        with open(all_captions_path, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        
        caption_info["all_captions_path"] = all_captions_path
        caption_info["caption_variants"] = list(captions.keys())
        
        return caption_info
    
    def _collect_sample_metadata(self, sample: fo.Sample, image_info: Dict, caption_info: Dict) -> Dict[str, Any]:
        """Collect comprehensive metadata for each sample."""
        
        metadata = {
            "original_filepath": sample.filepath,
            "processed_image_path": str(image_info["output_path"]),
            "image_size": image_info["size"],
            "primary_caption": caption_info["primary_caption"],
            "caption_variants_count": len(caption_info["caption_variants"]),
            "source": sample.get("source", "unknown"),
            "semiotic_type": sample.get("semiotic_type", "unknown")
        }
        
        # Add semiotic features
        if hasattr(sample, "semiotic_features"):
            semiotic_features = sample.semiotic_features
            metadata["semiotic_features"] = {
                "architectural_style": semiotic_features.get("architectural_style"),
                "urban_mood": semiotic_features.get("urban_mood"),
                "semiotic_score": semiotic_features.get("semiotic_score", 0),
                "cultural_context": semiotic_features.get("cultural_context"),
                "materials": semiotic_features.get("materials", [])
            }
        
        return metadata
    
    def _create_training_config(self, splits_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create training configuration for Flux.1d."""
        
        config = {
            "dataset_name": "semiotic_urban_flux",
            "dataset_path": str(self.output_path),
            "splits": {name: info["sample_count"] for name, info in splits_info.items()},
            "total_samples": sum(info["sample_count"] for info in splits_info.values()),
            "image_format": "JPEG",
            "caption_format": "text",
            "semiotic_enhanced": True,
            
            # Training parameters
            "recommended_settings": {
                "resolution": 1024,
                "batch_size": 1,
                "learning_rate": 1e-4,
                "max_train_steps": 2000,
                "gradient_accumulation_steps": 4,
                "save_steps": 500,
                "validation_steps": 100,
                "lora_rank": 64,
                "lora_alpha": 64,
                "lora_dropout": 0.1
            },
            
            # Semiotic-specific settings
            "semiotic_config": {
                "architectural_styles_covered": self._count_architectural_styles(splits_info),
                "urban_moods_covered": self._count_urban_moods(splits_info),
                "cultural_contexts": self._count_cultural_contexts(splits_info),
                "use_style_conditioning": True,
                "use_mood_conditioning": True,
                "semantic_guidance_scale": 7.5
            }
        }
        
        return config
    
    def _count_architectural_styles(self, splits_info: Dict[str, Any]) -> List[str]:
        """Count unique architectural styles in dataset."""
        styles = set()
        
        for split_info in splits_info.values():
            for sample in split_info["samples"]:
                style = sample["metadata"].get("semiotic_features", {}).get("architectural_style")
                if style:
                    styles.add(style)
        
        return list(styles)
    
    def _count_urban_moods(self, splits_info: Dict[str, Any]) -> List[str]:
        """Count unique urban moods in dataset."""
        moods = set()
        
        for split_info in splits_info.values():
            for sample in split_info["samples"]:
                mood = sample["metadata"].get("semiotic_features", {}).get("urban_mood")
                if mood:
                    moods.add(mood)
        
        return list(moods)
    
    def _count_cultural_contexts(self, splits_info: Dict[str, Any]) -> List[str]:
        """Count unique cultural contexts in dataset."""
        contexts = set()
        
        for split_info in splits_info.values():
            for sample in split_info["samples"]:
                context = sample["metadata"].get("semiotic_features", {}).get("cultural_context")
                if context:
                    contexts.add(context)
        
        return list(contexts)
    
    def _save_dataset_metadata(self, splits_info: Dict[str, Any], training_config: Dict[str, Any]):
        """Save complete dataset metadata and training configuration."""
        
        # Save training configuration
        config_path = self.metadata_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        # Save dataset info
        dataset_info = {
            "dataset_name": "Semiotic Urban Architecture Dataset",
            "version": "1.0",
            "description": "Semiotic-aware urban architectural dataset for Flux.1d fine-tuning",
            "total_samples": training_config["total_samples"],
            "splits": training_config["splits"],
            "preparation_date": pd.Timestamp.now().isoformat(),
            "semiotic_features": training_config["semiotic_config"],
            "data_structure": {
                "images": "images/{split}/{filename}.jpg",
                "captions": "captions/{split}/{filename}.txt",
                "metadata": "metadata/{split}_metadata.json"
            }
        }
        
        info_path = self.metadata_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # Create README
        self._create_readme(training_config)
        
        logger.info(f"Dataset metadata saved to {self.metadata_dir}")
    
    def _create_readme(self, training_config: Dict[str, Any]):
        """Create README file for the dataset."""
        
        readme_content = f"""# Semiotic Urban Architecture Dataset for Flux.1d

This dataset contains {training_config['total_samples']} semiotic-aware urban architectural images prepared for Flux.1d fine-tuning.

## Dataset Structure

```
flux_training/
├── images/
│   ├── train/          # Training images ({training_config['splits'].get('train', 0)} samples)
│   ├── val/            # Validation images ({training_config['splits'].get('val', 0)} samples)
│   └── test/           # Test images ({training_config['splits'].get('test', 0)} samples)
├── captions/
│   ├── train/          # Training captions
│   ├── val/            # Validation captions
│   └── test/           # Test captions
└── metadata/
    ├── training_config.json    # Training configuration
    ├── dataset_info.json       # Dataset information
    └── *_metadata.json         # Split-specific metadata
```

## Semiotic Features

- **Architectural Styles**: {', '.join(training_config['semiotic_config']['architectural_styles_covered'])}
- **Urban Moods**: {', '.join(training_config['semiotic_config']['urban_moods_covered'])}
- **Cultural Contexts**: {', '.join(training_config['semiotic_config']['cultural_contexts'])}

## Recommended Training Settings

```json
{json.dumps(training_config['recommended_settings'], indent=2)}
```

## Usage

This dataset is prepared for Flux.1d LoRA fine-tuning with semiotic awareness. The enhanced captions include architectural style, urban mood, cultural context, and material information to enable semantically-rich image generation.

## Caption Examples

Each image has multiple caption variants including:
- Original prompts
- Semiotic-enhanced descriptions
- BLIP-2 generated captions
- Template-based architectural descriptions

The primary captions emphasize semiotic elements for improved architectural understanding in generated images.
"""
        
        readme_path = self.output_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Main execution for data preparation."""
    
    # Initialize preparator
    base_path = Path(__file__).parent.parent
    preparator = FluxTrainingDataPreparator(str(base_path))
    
    # Load processed dataset
    dataset_name = "semiotic_urban_combined"
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        
        # Prepare training data
        training_config = preparator.prepare_training_data(
            dataset, 
            max_samples=1000  # Limit for initial testing
        )
        
        print(f"Training data prepared successfully!")
        print(f"Dataset location: {preparator.output_path}")
        print(f"Total samples: {training_config['total_samples']}")
        print(f"Splits: {training_config['splits']}")
        
    else:
        print(f"Dataset {dataset_name} not found. Run the full pipeline first.")

if __name__ == "__main__":
    main()