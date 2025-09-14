"""
BLIP-2 captioning system that processes exported data pipeline files
and generates enhanced semiotic-aware captions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportBLIPCaptioner:
    """BLIP-2 captioner for processing exported data pipeline files."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "auto"):
        """Initialize the BLIP-2 model."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading BLIP-2 model: {model_name} on {self.device}")
        
        # Load processor and model
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Semiotic prompts
        self.semiotic_prompt = (
            "Describe this architectural scene with focus on the architectural style, "
            "urban mood, materials, lighting conditions, and spatial qualities. "
            "Include cultural and symbolic meanings."
        )
    
    def process_exported_data(self, data_pipeline_path: str, output_path: str):
        """Process exported data pipeline files and enhance captions."""
        
        data_path = Path(data_pipeline_path)
        output_dir = Path(output_path)
        
        # Process training set
        train_images = data_path / "images" / "train"
        train_labels = data_path / "labels" / "train"
        
        if train_images.exists() and train_labels.exists():
            logger.info("Processing training set...")
            self._process_split(train_images, train_labels, output_dir / "train")
        
        # Process validation set
        val_images = data_path / "images" / "val"
        val_labels = data_path / "labels" / "val"
        
        if val_images.exists() and val_labels.exists():
            logger.info("Processing validation set...")
            self._process_split(val_images, val_labels, output_dir / "val")
    
    def _process_split(self, images_dir: Path, labels_dir: Path, output_dir: Path):
        """Process a single split (train/val)."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(images_dir.glob("*"))
        logger.info(f"Found {len(image_files)} images to process")
        
        processed = 0
        enhanced = 0
        
        for image_file in image_files:
            try:
                # Find corresponding label file
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    logger.warning(f"No label file found for {image_file.name}")
                    continue
                
                # Read original caption
                with open(label_file, 'r', encoding='utf-8') as f:
                    original_caption = f.read().strip()
                
                # Check if this needs enhancement (simple OID captions)
                if self._needs_enhancement(original_caption):
                    # Generate enhanced caption
                    enhanced_caption = self._generate_enhanced_caption(image_file, original_caption)
                    enhanced += 1
                else:
                    # Keep original rich caption
                    enhanced_caption = original_caption
                
                # Save enhanced caption
                output_file = output_dir / f"{image_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_caption)
                
                processed += 1
                
                if processed % 50 == 0:
                    logger.info(f"Processed {processed}/{len(image_files)} images, enhanced {enhanced}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                continue
        
        logger.info(f"Completed {processed} images, enhanced {enhanced} captions")
    
    def _needs_enhancement(self, caption: str) -> bool:
        """Check if caption needs enhancement (simple OID captions)."""
        
        # Simple heuristic: if caption contains generic patterns, it needs enhancement
        simple_patterns = [
            "Urban scene with",
            "unknown architecture, neutral atmosphere",
            "Building, Castle, Convenience store"
        ]
        
        for pattern in simple_patterns:
            if pattern in caption:
                return True
        
        return False
    
    def _generate_enhanced_caption(self, image_path: Path, original_caption: str) -> str:
        """Generate enhanced semiotic caption."""
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Generate semiotic-aware caption
            inputs = self.processor(
                image, 
                text=self.semiotic_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            enhanced_caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            
            # Clean up caption
            enhanced_caption = self._clean_caption(enhanced_caption)
            
            # Combine with original objects if useful
            if "Building" in original_caption:
                objects = self._extract_objects(original_caption)
                if objects:
                    enhanced_caption = f"{enhanced_caption} The scene includes {', '.join(objects)}."
            
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error generating enhanced caption: {e}")
            return original_caption
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and post-process generated caption."""
        
        # Remove prompt text if it appears in output
        prompt_phrases = [
            "describe this", "this image shows", "the image depicts",
            "this architectural scene", "this scene shows"
        ]
        
        caption_lower = caption.lower()
        for phrase in prompt_phrases:
            if caption_lower.startswith(phrase):
                caption = caption[len(phrase):].strip()
                break
        
        # Ensure proper capitalization
        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        # Remove redundant punctuation
        caption = re.sub(r'\.+$', '.', caption)
        if not caption.endswith('.'):
            caption += '.'
        
        return caption
    
    def _extract_objects(self, caption: str) -> List[str]:
        """Extract object names from OID caption."""
        
        if "Urban scene with" in caption:
            # Extract objects from "Urban scene with Building, Castle, ..."
            objects_part = caption.split("Urban scene with")[1].split(",")
            objects = []
            for obj in objects_part:
                obj = obj.strip()
                if obj and "unknown architecture" not in obj and "neutral atmosphere" not in obj:
                    objects.append(obj.lower())
            return objects[:3]  # Limit to first 3 objects
        
        return []

def main():
    """Main execution."""
    
    # Initialize captioner
    captioner = ExportBLIPCaptioner()
    
    # Set paths
    base_path = Path(__file__).parent.parent
    data_pipeline_path = base_path / "data" / "outputs" / "01_data_pipeline"
    output_path = base_path / "data" / "outputs" / "02_blip2_captions"
    
    # Process exported data
    if data_pipeline_path.exists():
        logger.info(f"Processing data from: {data_pipeline_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        captioner.process_exported_data(str(data_pipeline_path), str(output_path))
        
        logger.info("BLIP-2 captioning completed!")
    else:
        logger.error(f"Data pipeline output not found at: {data_pipeline_path}")
        logger.error("Please run data_pipeline.py first.")

if __name__ == "__main__":
    main()