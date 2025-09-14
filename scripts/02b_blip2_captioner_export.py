"""
BLIP-2 captioning system that processes exported data pipeline files
and generates enhanced semiotic-aware captions.
"""

import os
import json
import logging
import argparse
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
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "auto",
        gpu_id: int = 0,
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize the BLIP-2 model with GPU controls.

        Args:
            model_name: HF model id to load.
            device: "auto" | "cpu" | "cuda". If cuda and available, prefer GPU.
            gpu_id: Target GPU index (used when device=="cuda").
            dtype: "auto" | "fp16" | "bf16" | "fp32". Default auto picks fp16 on CUDA else fp32.
            load_in_8bit: If True and bitsandbytes available, load in 8-bit.
            load_in_4bit: If True and bitsandbytes available, load in 4-bit.
        """

        # Resolve desired device
        requested_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
        if requested_cuda and torch.cuda.is_available():
            # Restrict visibility to the requested GPU before model load
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
            # In case multiple GPUs remain visible, also set current device
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Resolve dtype
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:  # auto
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(
            f"Loading BLIP-2 model: {model_name} on {self.device} (gpu_id={gpu_id if self.device=='cuda' else 'N/A'}, dtype={torch_dtype})"
        )

        # Set local models cache directory
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        cache_dir = str(models_dir)

        # Check if model exists locally to avoid re-download
        model_cache_name = model_name.replace("/", "--")
        local_model_path = models_dir / f"models--{model_cache_name}"
        
        if local_model_path.exists():
            logger.info(f"Using existing local model at: {local_model_path}")
            # Use local_files_only to prevent download attempts
            self.processor = Blip2Processor.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True
            )
        else:
            logger.info(f"Model not found locally, will download to: {local_model_path}")
            # Load processor and model with download
            self.processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        # Quantization (optional)
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        if self.device == "cuda":
            # Use device_map auto within the constrained CUDA_VISIBLE_DEVICES
            model_kwargs["device_map"] = "auto"

        # Load model with same local check
        if local_model_path.exists():
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
                **model_kwargs,
            )
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                **model_kwargs,
            )

        if self.device != "cuda":
            # CPU path
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

    parser = argparse.ArgumentParser(description="BLIP-2 captioner export")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b", help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id to use when device=cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="Computation dtype")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    parser.add_argument("--input", default=None, help="Override data pipeline input path")
    parser.add_argument("--output", default=None, help="Override captions output path")

    args = parser.parse_args()

    # Initialize captioner with GPU controls
    captioner = ExportBLIPCaptioner(
        model_name=args.model,
        device=args.device,
        gpu_id=args.gpu,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # Set paths
    base_path = Path(__file__).parent.parent
    data_pipeline_path = (
        Path(args.input) if args.input else base_path / "data" / "outputs" / "01_data_pipeline"
    )
    output_path = (
        Path(args.output) if args.output else base_path / "data" / "outputs" / "blip2_captioner_export"
    )

    # Process exported data
    if data_pipeline_path.exists():
        logger.info(f"Processing data from: {data_pipeline_path}")
        logger.info(f"Output will be saved to: {output_path}")

        if captioner.device != "cuda":
            logger.warning(
                "CUDA not available or CPU selected. Running BLIP-2 on CPU will be slow."
            )

        captioner.process_exported_data(str(data_pipeline_path), str(output_path))

        logger.info("BLIP-2 captioning completed!")
    else:
        logger.error(f"Data pipeline output not found at: {data_pipeline_path}")
        logger.error("Please run data_pipeline.py first.")

if __name__ == "__main__":
    main()