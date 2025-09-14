"""
BLIP-2 based captioning system for generating semiotic-aware descriptions
of urban and architectural images.
"""

import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import fiftyone as fo
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import re
from dataclasses import dataclass
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemioticPrompt:
    """Templates for semiotic-aware prompting."""
    
    BASE_ARCHITECTURE = "Describe the architectural style and urban design elements in this image"
    MOOD_ATMOSPHERE = "Describe the mood, atmosphere, and emotional qualities of this urban scene"
    SPATIAL_RELATIONS = "Describe the spatial relationships, composition, and visual hierarchy in this urban environment"
    MATERIALS_TEXTURES = "Describe the materials, textures, and surface qualities visible in this architectural scene"
    LIGHTING_TIME = "Describe the lighting conditions, time of day, and seasonal characteristics of this urban image"
    CULTURAL_CONTEXT = "Describe the cultural and social context suggested by this urban environment"
    SEMIOTIC_ANALYSIS = "Analyze the symbolic meanings, cultural references, and semiotic elements in this urban scene"

class SemioticBLIPCaptioner:
    """BLIP-2 based captioning system with semiotic awareness."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "auto"):
        """Initialize the BLIP-2 model for semiotic captioning."""
        
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
        
        self.prompts = SemioticPrompt()
        
        # Semiotic vocabulary for urban environments
        self.semiotic_vocabulary = {
            "architectural_styles": [
                "brutalist", "modernist", "postmodern", "baroque", "gothic", "renaissance",
                "art deco", "minimalist", "futuristic", "industrial", "neoclassical",
                "bauhaus", "deconstructivist", "high-tech", "organic", "vernacular"
            ],
            "urban_moods": [
                "contemplative", "vibrant", "tense", "bustling", "serene", "dynamic",
                "melancholic", "energetic", "peaceful", "chaotic", "mysterious",
                "imposing", "welcoming", "austere", "luxurious", "utilitarian"
            ],
            "spatial_qualities": [
                "monumental", "intimate", "expansive", "compressed", "hierarchical",
                "fragmented", "unified", "layered", "transparent", "enclosed",
                "open", "flowing", "rigid", "organic", "geometric"
            ],
            "materials": [
                "concrete", "glass", "steel", "stone", "brick", "wood", "limestone",
                "marble", "metal", "stucco", "granite", "sandstone", "aluminum",
                "copper", "terra cotta", "ceramic", "composite"
            ],
            "lighting_qualities": [
                "dramatic", "soft", "harsh", "warm", "cool", "diffused", "directional",
                "atmospheric", "golden", "blue hour", "overcast", "bright", "shadowy",
                "high contrast", "low contrast", "ethereal"
            ]
        }
    
    def generate_semiotic_caption(self, image_path: str, 
                                 prompt_type: str = "comprehensive") -> Dict[str, str]:
        """Generate comprehensive semiotic caption for an image."""
        
        image = Image.open(image_path).convert("RGB")
        
        if prompt_type == "comprehensive":
            return self._generate_comprehensive_caption(image)
        else:
            return self._generate_specific_caption(image, prompt_type)
    
    def _generate_comprehensive_caption(self, image: Image.Image) -> Dict[str, str]:
        """Generate comprehensive semiotic analysis."""
        
        captions = {}
        
        # Generate different aspects of semiotic analysis
        prompt_configs = [
            ("architectural_analysis", self.prompts.BASE_ARCHITECTURE),
            ("mood_atmosphere", self.prompts.MOOD_ATMOSPHERE),
            ("spatial_relations", self.prompts.SPATIAL_RELATIONS),
            ("materials_textures", self.prompts.MATERIALS_TEXTURES),
            ("lighting_time", self.prompts.LIGHTING_TIME),
            ("cultural_context", self.prompts.CULTURAL_CONTEXT),
            ("semiotic_elements", self.prompts.SEMIOTIC_ANALYSIS)
        ]
        
        for aspect, prompt in prompt_configs:
            caption = self._generate_caption_with_prompt(image, prompt)
            captions[aspect] = caption
        
        # Generate unified semiotic-aware caption
        captions["unified_caption"] = self._create_unified_caption(captions)
        
        return captions
    
    def _generate_specific_caption(self, image: Image.Image, aspect: str) -> Dict[str, str]:
        """Generate caption for specific semiotic aspect."""
        
        prompt_map = {
            "architecture": self.prompts.BASE_ARCHITECTURE,
            "mood": self.prompts.MOOD_ATMOSPHERE,
            "spatial": self.prompts.SPATIAL_RELATIONS,
            "materials": self.prompts.MATERIALS_TEXTURES,
            "lighting": self.prompts.LIGHTING_TIME,
            "cultural": self.prompts.CULTURAL_CONTEXT,
            "semiotic": self.prompts.SEMIOTIC_ANALYSIS
        }
        
        if aspect not in prompt_map:
            aspect = "architecture"  # default
        
        caption = self._generate_caption_with_prompt(image, prompt_map[aspect])
        return {aspect: caption}
    
    def _generate_caption_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using specific prompt."""
        
        try:
            # Prepare inputs
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            
            # Clean up caption
            caption = self._clean_caption(caption)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error generating caption"
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and post-process generated caption."""
        
        # Remove prompt text if it appears in output
        prompt_phrases = [
            "describe the", "this image shows", "the image depicts",
            "in this image", "this is a", "this shows"
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
    
    def _create_unified_caption(self, aspect_captions: Dict[str, str]) -> str:
        """Create unified semiotic-aware caption from multiple aspects."""
        
        # Extract key elements from each aspect
        architectural_style = self._extract_vocabulary_terms(
            aspect_captions.get("architectural_analysis", ""),
            self.semiotic_vocabulary["architectural_styles"]
        )
        
        mood = self._extract_vocabulary_terms(
            aspect_captions.get("mood_atmosphere", ""),
            self.semiotic_vocabulary["urban_moods"]
        )
        
        materials = self._extract_vocabulary_terms(
            aspect_captions.get("materials_textures", ""),
            self.semiotic_vocabulary["materials"]
        )
        
        lighting = self._extract_vocabulary_terms(
            aspect_captions.get("lighting_time", ""),
            self.semiotic_vocabulary["lighting_qualities"]
        )
        
        # Construct unified caption
        caption_parts = []
        
        if architectural_style:
            caption_parts.append(f"{architectural_style[0]} architectural style")
        
        if mood:
            caption_parts.append(f"{mood[0]} urban atmosphere")
        
        if materials:
            caption_parts.append(f"featuring {materials[0]} materials")
        
        if lighting:
            caption_parts.append(f"under {lighting[0]} lighting conditions")
        
        # Base description from architectural analysis
        base_desc = aspect_captions.get("architectural_analysis", "Urban architectural scene")
        
        if caption_parts:
            unified = f"{base_desc}, characterized by {', '.join(caption_parts)}."
        else:
            unified = f"{base_desc}."
        
        return unified
    
    def _extract_vocabulary_terms(self, text: str, vocabulary: List[str]) -> List[str]:
        """Extract relevant vocabulary terms from text."""
        
        text_lower = text.lower()
        found_terms = []
        
        for term in vocabulary:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms

    def caption_directory(self, input_dir: str, output_file: str, recursive: bool = True) -> None:
        """Caption all images under input_dir and write a JSON file.

        The JSON maps relative image paths (from input_dir) to the generated
        semiotic captions dictionary (including unified_caption).
        """
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        base = Path(input_dir)
        files = (base.rglob("*") if recursive else base.glob("*"))
        images = [p for p in files if p.suffix.lower() in img_exts]

        logger.info(f"Found {len(images)} images under {base}")
        captions: Dict[str, Dict[str, str]] = {}

        for idx, img_path in enumerate(images, 1):
            try:
                caps = self.generate_semiotic_caption(str(img_path))
                rel = str(img_path.relative_to(base))
                captions[rel] = caps
                if idx % 25 == 0:
                    logger.info(f"Captioned {idx}/{len(images)} images")
            except Exception as e:
                logger.error(f"Failed to caption {img_path}: {e}")

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote captions JSON: {out_path} ({len(captions)} items)")
    
    def process_dataset(self, dataset: fo.Dataset, 
                       output_field: str = "semiotic_captions",
                       batch_size: int = 1) -> None:
        """Process entire FiftyOne dataset to add semiotic captions."""
        
        logger.info(f"Processing {len(dataset)} samples for semiotic captioning")
        
        processed = 0
        for sample in dataset.iter_samples(progress=True):
            try:
                # Generate semiotic captions
                captions = self.generate_semiotic_caption(sample.filepath)
                
                # Add captions to sample
                sample[output_field] = captions
                sample.save()
                
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing {sample.filepath}: {e}")
                continue
        
        logger.info(f"Completed processing {processed} samples")
    
    def enhance_existing_captions(self, dataset: fo.Dataset,
                                 caption_field: str = "original_prompt",
                                 enhanced_field: str = "enhanced_semiotic_caption") -> None:
        """Enhance existing captions with semiotic analysis."""
        
        for sample in dataset.iter_samples(progress=True):
            try:
                if not hasattr(sample, caption_field) or not sample[caption_field]:
                    continue
                
                original_caption = sample[caption_field]
                
                # Generate semiotic analysis
                semiotic_captions = self.generate_semiotic_caption(sample.filepath)
                
                # Combine original with semiotic analysis
                enhanced_caption = self._combine_captions(
                    original_caption, 
                    semiotic_captions["unified_caption"]
                )
                
                sample[enhanced_field] = enhanced_caption
                sample["semiotic_analysis"] = semiotic_captions
                sample.save()
                
            except Exception as e:
                logger.error(f"Error enhancing caption for {sample.filepath}: {e}")
                continue
    
    def _combine_captions(self, original: str, semiotic: str) -> str:
        """Combine original caption with semiotic analysis."""
        
        # Simple combination - can be made more sophisticated
        return f"{original} This scene demonstrates {semiotic.lower()}"

def main():
    """CLI: caption an image directory into a captions.json; fallback to demo."""
    parser = argparse.ArgumentParser(description="BLIP-2 Semiotic Captioner (02)")
    parser.add_argument("--input_dir", default=None, help="Directory with images to caption")
    parser.add_argument("--output_file", default=None, help="Output JSON path for captions")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b", help="HF model id")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    args = parser.parse_args()

    # If CLI paths provided, run directory mode
    if args.input_dir:
        default_out = Path(__file__).parent.parent / "data" / "outputs" / "02_blip2_captioner" / "captions.json"
        output_file = args.output_file if args.output_file else str(default_out)
        captioner = SemioticBLIPCaptioner(model_name=args.model, device=args.device)
        captioner.caption_directory(args.input_dir, output_file, recursive=(not args.no_recursive))
        return

    # Fallback: small demo using FiftyOne dataset if available
    captioner = SemioticBLIPCaptioner(model_name=args.model, device=args.device)
    dataset_name = "semiotic_urban_combined"
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        test_view = dataset.take(5)
        captioner.process_dataset(test_view)
        session = fo.launch_app(dataset, port=5151)
        print("Semiotic captions generated. View results at http://localhost:5151")
        try:
            session.wait()
        except KeyboardInterrupt:
            print("Shutting down...")
    else:
        print(f"Dataset {dataset_name} not found. Provide --input_dir to caption a folder.")

if __name__ == "__main__":
    main()