"""
Inference pipeline for generating semiotic-aware architectural images
using the fine-tuned Flux.1d model with advanced prompt engineering.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass
import argparse
import time
import random

# Import libraries with fallback handling
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not available. Install with: pip install gradio")

try:
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers not available. Install with: pip install diffusers")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available. Install with: pip install peft")

# Import evaluation system for quality assessment
try:
    from evaluation_pipeline import SemioticImageEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Evaluation pipeline not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    
    # Generation parameters
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    
    # Semiotic conditioning
    use_semiotic_conditioning: bool = True
    architectural_style: Optional[str] = None
    urban_mood: Optional[str] = None
    time_period: Optional[str] = None
    materials: Optional[List[str]] = None
    
    # Advanced parameters
    use_style_transfer: bool = False
    style_strength: float = 0.7
    negative_prompt: Optional[str] = None

class SemioticFluxInferencePipeline:
    """Advanced inference pipeline for semiotic-aware architectural image generation."""
    
    def __init__(self, model_path: str, lora_path: Optional[str] = None, device: str = "auto"):
        """Initialize the inference pipeline."""
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required. Install with: pip install diffusers")
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_path = model_path
        self.lora_path = lora_path
        
        # Load models
        self._load_models()
        
        # Initialize prompt engineering system
        self._init_prompt_engineering()
        
        # Initialize evaluation system (optional)
        self.evaluator = None
        
        logger.info("Semiotic Flux inference pipeline initialized successfully")
    
    def _load_models(self):
        """Load the Flux pipeline and LoRA weights."""
        
        logger.info(f"Loading Flux model from {self.model_path}")
        
        # Load base pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load LoRA weights if provided
        if self.lora_path and Path(self.lora_path).exists() and PEFT_AVAILABLE:
            logger.info(f"Loading LoRA weights from {self.lora_path}")
            
            # Load LoRA adapter
            self.pipeline.transformer = PeftModel.from_pretrained(
                self.pipeline.transformer,
                self.lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        
        # Move to device if not using device_map
        if self.device != "cuda":
            self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except ImportError:
                logger.warning("xformers not available, using default attention")
    
    def _init_prompt_engineering(self):
        """Initialize advanced prompt engineering system."""
        
        # Semiotic conditioning tokens
        self.semiotic_tokens = {
            "architectural_styles": {
                "modernist": "<modernist>",
                "brutalist": "<brutalist>",
                "postmodern": "<postmodern>",
                "minimalist": "<minimalist>",
                "baroque": "<baroque>",
                "gothic": "<gothic>",
                "industrial": "<industrial>",
                "art_deco": "<art_deco>"
            },
            
            "urban_moods": {
                "contemplative": "<contemplative>",
                "vibrant": "<vibrant>",
                "tense": "<tense>",
                "serene": "<serene>",
                "dramatic": "<dramatic>",
                "peaceful": "<peaceful>",
                "energetic": "<energetic>",
                "melancholic": "<melancholic>"
            },
            
            "time_periods": {
                "dawn": "<dawn>",
                "morning": "<morning>",
                "afternoon": "<afternoon>",
                "dusk": "<dusk>",
                "night": "<night>",
                "golden_hour": "<golden_hour>",
                "blue_hour": "<blue_hour>"
            },
            
            "materials": {
                "concrete": "<concrete>",
                "glass": "<glass>",
                "steel": "<steel>",
                "stone": "<stone>",
                "wood": "<wood>",
                "brick": "<brick>"
            }
        }
        
        # Style enhancement templates
        self.style_templates = {
            "modernist": "clean geometric lines, minimal ornamentation, functional design, large windows, open spaces",
            "brutalist": "raw concrete, massive geometric forms, fortress-like appearance, monumental scale",
            "postmodern": "eclectic mix of styles, colorful facade, playful elements, historical references",
            "minimalist": "pure geometric forms, white surfaces, essential elements only, maximum simplicity",
            "baroque": "ornate decoration, curved forms, dramatic lighting, rich materials, opulent details",
            "gothic": "pointed arches, ribbed vaults, flying buttresses, vertical emphasis, stone masonry",
            "industrial": "exposed structural elements, metal and brick, utilitarian aesthetic, raw materials",
            "art_deco": "streamlined forms, geometric patterns, metallic finishes, stylized ornamentation"
        }
        
        # Mood enhancement templates
        self.mood_templates = {
            "contemplative": "peaceful atmosphere, soft lighting, quiet spaces, reflective mood",
            "vibrant": "energetic colors, dynamic composition, bustling activity, lively atmosphere",
            "tense": "dramatic contrasts, sharp angles, conflicting elements, urban pressure",
            "serene": "harmonious proportions, balanced composition, tranquil setting, calm atmosphere",
            "dramatic": "bold contrasts, striking lighting, powerful forms, emotional impact",
            "peaceful": "gentle transitions, soft materials, natural integration, restful quality",
            "energetic": "dynamic lines, bright colors, movement suggestion, active environment",
            "melancholic": "muted colors, weathered surfaces, nostalgic elements, contemplative sadness"
        }
        
        # Quality enhancement prompts
        self.quality_enhancers = [
            "architectural photography",
            "professional photography",
            "high resolution",
            "detailed rendering",
            "photorealistic",
            "sharp focus",
            "perfect composition",
            "award winning photography"
        ]
        
        # Negative prompts for better quality
        self.default_negative_prompts = [
            "blurry", "out of focus", "low quality", "pixelated", "distorted",
            "unrealistic", "cartoon", "anime", "sketch", "drawing",
            "watermark", "text", "signature", "cropped", "cut off"
        ]
    
    def enhance_prompt(self, prompt: str, config: GenerationConfig) -> Tuple[str, str]:
        """Enhance prompt with semiotic conditioning and quality improvements."""
        
        enhanced_prompt = prompt
        
        # Add semiotic conditioning tokens
        if config.use_semiotic_conditioning:
            conditioning_tokens = []
            
            # Add architectural style token
            if config.architectural_style and config.architectural_style in self.semiotic_tokens["architectural_styles"]:
                style_token = self.semiotic_tokens["architectural_styles"][config.architectural_style]
                conditioning_tokens.append(style_token)
                
                # Add style description
                style_desc = self.style_templates.get(config.architectural_style, "")
                if style_desc:
                    enhanced_prompt += f", {style_desc}"
            
            # Add urban mood token
            if config.urban_mood and config.urban_mood in self.semiotic_tokens["urban_moods"]:
                mood_token = self.semiotic_tokens["urban_moods"][config.urban_mood]
                conditioning_tokens.append(mood_token)
                
                # Add mood description
                mood_desc = self.mood_templates.get(config.urban_mood, "")
                if mood_desc:
                    enhanced_prompt += f", {mood_desc}"
            
            # Add time period token
            if config.time_period and config.time_period in self.semiotic_tokens["time_periods"]:
                time_token = self.semiotic_tokens["time_periods"][config.time_period]
                conditioning_tokens.append(time_token)
                enhanced_prompt += f", photographed during {config.time_period}"
            
            # Add material tokens
            if config.materials:
                for material in config.materials:
                    if material in self.semiotic_tokens["materials"]:
                        material_token = self.semiotic_tokens["materials"][material]
                        conditioning_tokens.append(material_token)
                enhanced_prompt += f", featuring {', '.join(config.materials)} materials"
            
            # Add architectural context token
            conditioning_tokens.append("<architectural>")
            
            # Prepend conditioning tokens
            if conditioning_tokens:
                enhanced_prompt = " ".join(conditioning_tokens) + " " + enhanced_prompt
        
        # Add quality enhancers
        quality_terms = random.sample(self.quality_enhancers, min(3, len(self.quality_enhancers)))
        enhanced_prompt += f", {', '.join(quality_terms)}"
        
        # Create negative prompt
        negative_prompt = config.negative_prompt or ", ".join(self.default_negative_prompts)
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt, negative_prompt
    
    def generate_images(self, prompt: str, config: GenerationConfig) -> List[Image.Image]:
        """Generate images using the semiotic-aware Flux model."""
        
        logger.info(f"Generating {config.num_images} image(s) with prompt: {prompt}")
        
        # Enhance prompt
        enhanced_prompt, negative_prompt = self.enhance_prompt(prompt, config)
        
        # Set up generation parameters
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
        
        # Generate images
        try:
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    height=config.height,
                    width=config.width,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    num_images_per_prompt=config.num_images,
                    generator=generator
                )
            
            images = result.images
            logger.info(f"Successfully generated {len(images)} image(s)")
            
            return images
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            return []
    
    def generate_with_evaluation(self, prompt: str, config: GenerationConfig, 
                               evaluate: bool = True) -> Dict[str, Any]:
        """Generate images and optionally evaluate them."""
        
        # Generate images
        start_time = time.time()
        images = self.generate_images(prompt, config)
        generation_time = time.time() - start_time
        
        result = {
            "images": images,
            "generation_time": generation_time,
            "config": config,
            "enhanced_prompt": self.enhance_prompt(prompt, config)[0]
        }
        
        # Evaluate images if requested
        if evaluate and images and EVALUATION_AVAILABLE:
            if self.evaluator is None:
                self.evaluator = SemioticImageEvaluator(device=self.device)
            
            # Save temporary images for evaluation
            temp_paths = []
            for i, img in enumerate(images):
                temp_path = f"temp_gen_{i}.png"
                img.save(temp_path)
                temp_paths.append(temp_path)
            
            try:
                # Evaluate images
                evaluation_report = self.evaluator.evaluate_batch(
                    temp_paths, [prompt] * len(images)
                )
                
                result["evaluation"] = evaluation_report
                
                # Clean up temporary files
                for temp_path in temp_paths:
                    Path(temp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                result["evaluation"] = None
        
        return result
    
    def batch_generate(self, prompts: List[str], config: GenerationConfig, 
                      output_dir: str = "generated_images") -> Dict[str, Any]:
        """Generate images for a batch of prompts."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        all_images = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
            
            # Generate images
            generation_result = self.generate_with_evaluation(prompt, config, evaluate=False)
            
            # Save images
            saved_paths = []
            for j, image in enumerate(generation_result["images"]):
                filename = f"prompt_{i:03d}_image_{j:03d}.png"
                image_path = output_path / filename
                image.save(image_path)
                saved_paths.append(str(image_path))
            
            result_entry = {
                "prompt": prompt,
                "image_paths": saved_paths,
                "generation_time": generation_result["generation_time"],
                "enhanced_prompt": generation_result["enhanced_prompt"]
            }
            
            results.append(result_entry)
            all_images.extend(generation_result["images"])
        
        # Save batch metadata
        batch_metadata = {
            "total_prompts": len(prompts),
            "total_images": len(all_images),
            "generation_config": config.__dict__,
            "results": results
        }
        
        metadata_path = output_path / "batch_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch generation completed. {len(all_images)} images saved to {output_path}")
        
        return batch_metadata
    
    def create_gradio_interface(self):
        """Create Gradio interface for interactive generation."""
        
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for the web interface. Install with: pip install gradio")
        
        def generate_interface(prompt, style, mood, time_period, materials, 
                             num_images, steps, guidance, seed, evaluate):
            """Interface function for Gradio."""
            
            # Parse materials
            material_list = [m.strip() for m in materials.split(",") if m.strip()] if materials else None
            
            # Create config
            config = GenerationConfig(
                architectural_style=style if style != "None" else None,
                urban_mood=mood if mood != "None" else None,
                time_period=time_period if time_period != "None" else None,
                materials=material_list,
                num_images=int(num_images),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed) if seed else None
            )
            
            # Generate images
            result = self.generate_with_evaluation(prompt, config, evaluate=bool(evaluate))
            
            images = result["images"]
            info = f"Generation Time: {result['generation_time']:.2f}s\n"
            info += f"Enhanced Prompt: {result['enhanced_prompt']}"
            
            if result.get("evaluation"):
                eval_summary = result["evaluation"]["summary"]["aggregate_metrics"]
                info += f"\n\nEvaluation Scores:\n"
                info += f"Overall: {eval_summary['overall_score']['mean']:.3f}\n"
                info += f"Semiotic: {eval_summary['semiotic_score']['mean']:.3f}\n"
                info += f"CLIP: {eval_summary['clip_score']['mean']:.3f}"
            
            return images, info
        
        # Define interface components
        interface = gr.Interface(
            fn=generate_interface,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Describe the architectural scene you want to generate..."),
                gr.Dropdown(
                    choices=["None"] + list(self.semiotic_tokens["architectural_styles"].keys()),
                    label="Architectural Style",
                    value="None"
                ),
                gr.Dropdown(
                    choices=["None"] + list(self.semiotic_tokens["urban_moods"].keys()),
                    label="Urban Mood",
                    value="None"
                ),
                gr.Dropdown(
                    choices=["None"] + list(self.semiotic_tokens["time_periods"].keys()),
                    label="Time Period",
                    value="None"
                ),
                gr.Textbox(label="Materials (comma-separated)", placeholder="concrete, glass, steel"),
                gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images"),
                gr.Slider(minimum=10, maximum=50, value=20, step=5, label="Inference Steps"),
                gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Guidance Scale"),
                gr.Number(label="Seed (optional)", precision=0),
                gr.Checkbox(label="Evaluate Generated Images", value=False)
            ],
            outputs=[
                gr.Gallery(label="Generated Images", columns=2, rows=2, height="auto"),
                gr.Textbox(label="Generation Info", lines=10)
            ],
            title="Semiotic Architectural Image Generator",
            description="Generate semiotic-aware architectural images using fine-tuned Flux.1d",
            examples=[
                [
                    "Modern glass office building in downtown urban setting",
                    "modernist", "contemplative", "afternoon", "glass, steel",
                    1, 20, 7.5, 42, False
                ],
                [
                    "Brutalist concrete housing complex with dramatic lighting",
                    "brutalist", "dramatic", "dusk", "concrete",
                    1, 25, 8.0, 123, False
                ],
                [
                    "Minimalist white residential building with natural surroundings",
                    "minimalist", "serene", "morning", "concrete, glass",
                    1, 20, 7.0, 456, False
                ]
            ]
        )
        
        return interface

def create_default_config() -> GenerationConfig:
    """Create default generation configuration."""
    
    return GenerationConfig(
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_images=1,
        use_semiotic_conditioning=True
    )

def main():
    """Main execution for inference pipeline."""
    
    parser = argparse.ArgumentParser(description="Semiotic Flux Inference Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base Flux model")
    parser.add_argument("--lora_path", type=str, help="Path to fine-tuned LoRA weights")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--style", type=str, help="Architectural style")
    parser.add_argument("--mood", type=str, help="Urban mood")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--batch_file", type=str, help="JSON file with batch prompts")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate generated images")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = SemioticFluxInferencePipeline(
            model_path=args.model_path,
            lora_path=args.lora_path
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    if args.gradio:
        # Launch Gradio interface
        try:
            interface = pipeline.create_gradio_interface()
            interface.launch(share=True)
        except Exception as e:
            print(f"Error launching Gradio interface: {e}")
        
    elif args.batch_file:
        # Batch generation
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        prompts = batch_data.get("prompts", [])
        config = GenerationConfig(**batch_data.get("config", {}))
        
        pipeline.batch_generate(prompts, config, args.output_dir)
        
    elif args.prompt:
        # Single prompt generation
        config = create_default_config()
        config.architectural_style = args.style
        config.urban_mood = args.mood
        
        result = pipeline.generate_with_evaluation(args.prompt, config, evaluate=args.evaluate)
        
        # Save images
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image in enumerate(result["images"]):
            image_path = output_path / f"generated_{i:03d}.png"
            image.save(image_path)
            print(f"Saved: {image_path}")
        
        # Save metadata
        metadata = {
            "prompt": args.prompt,
            "enhanced_prompt": result["enhanced_prompt"],
            "generation_time": result["generation_time"],
            "config": config.__dict__
        }
        
        if result.get("evaluation"):
            metadata["evaluation"] = result["evaluation"]
        
        metadata_path = output_path / "generation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Generation completed in {result['generation_time']:.2f}s")
        
    else:
        print("Please provide either --prompt, --batch_file, or --gradio option")
        print("Use --help for more information")

if __name__ == "__main__":
    main()