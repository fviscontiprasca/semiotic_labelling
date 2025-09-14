#!/usr/bin/env python3
"""
Flux Full Fine-tuning Inference Pipeline - Phase 08
===================================================

This script handles inference using fully fine-tuned Flux.1d models, optimized for 
enhanced semiotic conditioning and architectural image generation.

Unlike LoRA inference, this script loads fully fine-tuned models with all parameters
adapted to the training data. Features include enhanced prompt engineering, semiotic
token conditioning, and optimized memory management for full model inference.

Author: [Your Name]
Project: Semiocity - Semiotic Labelling Pipeline
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from transformers import AutoTokenizer, AutoModel
import fiftyone as fo
from sentence_transformers import SentenceTransformer

# Ensure script can find other pipeline modules
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

from semiotic_extractor import SemioticExtractor

@dataclass
class FullInferenceConfig:
    """Configuration for full fine-tuning inference"""
    # Model paths
    model_path: str = "models/flux_full_finetuned"
    base_model: str = "black-forest-labs/FLUX.1-dev"
    
    # Output settings
    output_dir: str = "outputs/flux_full_inference"
    num_samples: int = 4
    save_metadata: bool = True
    
    # Generation parameters
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    
    # Semiotic conditioning
    use_semiotic_conditioning: bool = True
    semiotic_weight: float = 1.2
    enhance_prompts: bool = True
    
    # Memory optimization
    enable_memory_efficient_attention: bool = True
    enable_sequential_cpu_offload: bool = False
    use_torch_compile: bool = False
    
    # Advanced settings
    seed: Optional[int] = None
    batch_size: int = 1
    negative_prompt: str = "blurry, low quality, distorted, unfinished"
    
    # Prompt templates
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "architectural": "A {style} {building_type} with {semiotic_features}, architectural photography, high quality, detailed",
        "urban": "Urban landscape featuring {semiotic_features}, {style} architecture, city planning, professional photography",
        "contextual": "{building_type} in {context} showing {semiotic_features}, {style} design, atmospheric lighting",
        "semiotic_rich": "Architectural composition emphasizing {semiotic_features}, {style} aesthetic, symbolic elements, professional rendering"
    })

class SemioticFluxFullInference:
    """Enhanced inference pipeline for fully fine-tuned Flux models"""
    
    def __init__(self, config: FullInferenceConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.pipeline = None
        self.semiotic_extractor = None
        self.sentence_transformer = None
        
        # Cache for semiotic features
        self.semiotic_cache = {}
        
        self._setup_output_directories()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/inference.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_output_directories(self):
        """Create necessary output directories"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/images").mkdir(exist_ok=True)
        Path(f"{self.config.output_dir}/metadata").mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the fully fine-tuned Flux model"""
        self.logger.info(f"Loading full fine-tuned model from {self.config.model_path}")
        
        try:
            # Load the fully fine-tuned pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                
            # Apply memory optimizations
            if self.config.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
                
            if self.config.enable_sequential_cpu_offload:
                self.pipeline.enable_sequential_cpu_offload()
                
            # Compile for faster inference if requested
            if self.config.use_torch_compile and hasattr(torch, 'compile'):
                self.pipeline.unet = torch.compile(self.pipeline.unet)
                
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Fallback to base model
            self.logger.info("Falling back to base model")
            self.pipeline = FluxPipeline.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
    
    def load_semiotic_components(self):
        """Load semiotic analysis components"""
        if self.config.use_semiotic_conditioning:
            self.logger.info("Loading semiotic analysis components")
            
            # Load semiotic extractor
            self.semiotic_extractor = SemioticExtractor()
            
            # Load sentence transformer for semantic analysis
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.logger.info("Semiotic components loaded")
    
    def enhance_prompt_with_semiotic(self, base_prompt: str, semiotic_features: Optional[List[str]] = None) -> str:
        """Enhance prompt with semiotic conditioning"""
        if not self.config.use_semiotic_conditioning or not semiotic_features:
            return base_prompt
            
        # Extract key semiotic elements
        semiotic_tokens = [f"<{feature.lower().replace(' ', '_')}>" for feature in semiotic_features[:3]]
        semiotic_string = " ".join(semiotic_tokens)
        
        # Choose appropriate template
        template_key = self._select_prompt_template(base_prompt, semiotic_features)
        template = self.config.prompt_templates.get(template_key, "{prompt}")
        
        # Extract semantic components
        style, building_type, context = self._extract_semantic_components(base_prompt)
        
        # Apply template with semiotic features
        if "{semiotic_features}" in template:
            enhanced_prompt = template.format(
                style=style,
                building_type=building_type,
                context=context,
                semiotic_features=", ".join(semiotic_features[:3])
            )
        else:
            enhanced_prompt = base_prompt
            
        # Add semiotic tokens
        enhanced_prompt = f"{semiotic_string} {enhanced_prompt}"
        
        self.logger.debug(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    
    def _select_prompt_template(self, prompt: str, semiotic_features: List[str]) -> str:
        """Select appropriate prompt template based on content"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['building', 'tower', 'house', 'structure']):
            return "architectural"
        elif any(word in prompt_lower for word in ['city', 'urban', 'street', 'district']):
            return "urban"
        elif len(semiotic_features) >= 2:
            return "semiotic_rich"
        else:
            return "contextual"
    
    def _extract_semantic_components(self, prompt: str) -> Tuple[str, str, str]:
        """Extract semantic components from prompt"""
        # Simple extraction - can be enhanced with NLP
        style_keywords = ['modern', 'classical', 'brutalist', 'gothic', 'contemporary', 'traditional']
        building_keywords = ['building', 'tower', 'house', 'structure', 'complex', 'center']
        context_keywords = ['city', 'urban', 'downtown', 'residential', 'commercial', 'historic']
        
        prompt_lower = prompt.lower()
        
        style = next((word for word in style_keywords if word in prompt_lower), "contemporary")
        building_type = next((word for word in building_keywords if word in prompt_lower), "building")
        context = next((word for word in context_keywords if word in prompt_lower), "urban setting")
        
        return style, building_type, context
    
    def analyze_prompt_semiotic_features(self, prompt: str) -> List[str]:
        """Analyze prompt to extract potential semiotic features"""
        if not self.semiotic_extractor:
            return []
            
        try:
            # Use cached features if available
            if prompt in self.semiotic_cache:
                return self.semiotic_cache[prompt]
                
            # Extract features using the semiotic extractor
            # This is a simplified version - in practice, you might need image context
            features = []
            
            # Analyze prompt semantically
            embedding = self.sentence_transformer.encode([prompt])
            
            # Extract architectural and urban semiotic indicators
            architectural_terms = [
                'symmetry', 'hierarchy', 'monumentality', 'transparency', 'solidity',
                'verticality', 'horizontality', 'rhythm', 'proportion', 'scale'
            ]
            
            urban_terms = [
                'connectivity', 'centrality', 'density', 'accessibility', 'visibility',
                'permeability', 'enclosure', 'openness', 'boundaries', 'transitions'
            ]
            
            # Simple keyword matching - enhance with semantic similarity
            prompt_lower = prompt.lower()
            for term in architectural_terms + urban_terms:
                if term in prompt_lower or any(syn in prompt_lower for syn in self._get_synonyms(term)):
                    features.append(term)
            
            # Cache results
            self.semiotic_cache[prompt] = features[:5]  # Limit to top 5
            return features[:5]
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze semiotic features: {e}")
            return []
    
    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for semiotic terms"""
        synonym_map = {
            'symmetry': ['balanced', 'harmony', 'regular'],
            'hierarchy': ['order', 'ranking', 'structure'],
            'monumentality': ['grandeur', 'imposing', 'majestic'],
            'transparency': ['openness', 'clarity', 'visibility'],
            'verticality': ['height', 'tall', 'upward'],
            'connectivity': ['connection', 'linked', 'networked'],
            'centrality': ['central', 'focal', 'hub'],
            'density': ['compact', 'concentrated', 'crowded']
        }
        return synonym_map.get(term, [])
    
    def generate_images(self, prompts: List[str], semiotic_features_list: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
        """Generate images from prompts with semiotic conditioning"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        results = []
        
        # Set random seed if specified
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Generating images for prompt {i+1}/{len(prompts)}: {prompt[:100]}...")
            
            # Get semiotic features
            semiotic_features = None
            if semiotic_features_list and i < len(semiotic_features_list):
                semiotic_features = semiotic_features_list[i]
            elif self.config.use_semiotic_conditioning:
                semiotic_features = self.analyze_prompt_semiotic_features(prompt)
            
            # Enhance prompt
            enhanced_prompt = self.enhance_prompt_with_semiotic(prompt, semiotic_features)
            
            try:
                # Generate images
                start_time = time.time()
                
                images = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=self.config.negative_prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    height=self.config.height,
                    width=self.config.width,
                    num_images_per_prompt=self.config.num_samples
                ).images
                
                generation_time = time.time() - start_time
                
                # Save images and metadata
                batch_results = []
                for j, image in enumerate(images):
                    # Save image
                    image_filename = f"image_{i:03d}_{j:02d}.png"
                    image_path = Path(self.config.output_dir) / "images" / image_filename
                    image.save(image_path)
                    
                    # Prepare metadata
                    metadata = {
                        'prompt': prompt,
                        'enhanced_prompt': enhanced_prompt,
                        'semiotic_features': semiotic_features or [],
                        'generation_time': generation_time / len(images),
                        'config': {
                            'num_inference_steps': self.config.num_inference_steps,
                            'guidance_scale': self.config.guidance_scale,
                            'height': self.config.height,
                            'width': self.config.width,
                            'seed': self.config.seed
                        },
                        'model_path': self.config.model_path,
                        'timestamp': time.time()
                    }
                    
                    # Save metadata
                    if self.config.save_metadata:
                        metadata_filename = f"metadata_{i:03d}_{j:02d}.json"
                        metadata_path = Path(self.config.output_dir) / "metadata" / metadata_filename
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    batch_results.append({
                        'image': image,
                        'image_path': str(image_path),
                        'metadata': metadata
                    })
                
                results.extend(batch_results)
                self.logger.info(f"Generated {len(images)} images in {generation_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to generate images for prompt {i}: {e}")
                continue
        
        return results
    
    def run_inference_from_file(self, prompts_file: str) -> List[Dict[str, Any]]:
        """Run inference from a file containing prompts"""
        self.logger.info(f"Loading prompts from {prompts_file}")
        
        try:
            with open(prompts_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                prompts = data
                semiotic_features_list = None
            elif isinstance(data, dict):
                prompts = data.get('prompts', [])
                semiotic_features_list = data.get('semiotic_features', None)
            else:
                raise ValueError("Invalid file format")
                
            return self.generate_images(prompts, semiotic_features_list)
            
        except Exception as e:
            self.logger.error(f"Failed to load prompts from file: {e}")
            return []
    
    def create_generation_report(self, results: List[Dict[str, Any]]) -> str:
        """Create a detailed generation report"""
        report_path = Path(self.config.output_dir) / "generation_report.json"
        
        report = {
            'summary': {
                'total_images': len(results),
                'successful_generations': len([r for r in results if 'image' in r]),
                'average_generation_time': np.mean([r['metadata']['generation_time'] for r in results]) if results else 0,
                'model_path': self.config.model_path,
                'timestamp': time.time()
            },
            'config': {
                'num_inference_steps': self.config.num_inference_steps,
                'guidance_scale': self.config.guidance_scale,
                'image_size': f"{self.config.width}x{self.config.height}",
                'semiotic_conditioning': self.config.use_semiotic_conditioning,
                'semiotic_weight': self.config.semiotic_weight
            },
            'results': [
                {
                    'image_path': r['image_path'],
                    'prompt': r['metadata']['prompt'],
                    'enhanced_prompt': r['metadata']['enhanced_prompt'],
                    'semiotic_features': r['metadata']['semiotic_features'],
                    'generation_time': r['metadata']['generation_time']
                }
                for r in results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Generation report saved to {report_path}")
        return str(report_path)

def main():
    parser = argparse.ArgumentParser(description="Flux Full Fine-tuning Inference Pipeline")
    parser.add_argument("--model-path", type=str, default="models/flux_full_finetuned",
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompts-file", type=str,
                        help="JSON file containing prompts to generate")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/flux_full_inference",
                        help="Output directory for generated images")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of images to generate per prompt")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducible generation")
    parser.add_argument("--no-semiotic", action="store_true",
                        help="Disable semiotic conditioning")
    parser.add_argument("--semiotic-weight", type=float, default=1.2,
                        help="Weight for semiotic conditioning")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable CPU offload for memory efficiency")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FullInferenceConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        use_semiotic_conditioning=not args.no_semiotic,
        semiotic_weight=args.semiotic_weight,
        enable_sequential_cpu_offload=args.cpu_offload,
        use_torch_compile=args.compile
    )
    
    # Initialize inference pipeline
    inference = SemioticFluxFullInference(config)
    
    # Load model and components
    print("Loading model...")
    inference.load_model()
    inference.load_semiotic_components()
    
    # Generate images
    results = []
    if args.prompts_file:
        results = inference.run_inference_from_file(args.prompts_file)
    elif args.prompt:
        results = inference.generate_images([args.prompt])
    else:
        # Default example prompts
        example_prompts = [
            "A modern glass office building with clean geometric lines",
            "A brutalist concrete residential complex in an urban setting",
            "A classical government building with columns and symmetrical facade",
            "A futuristic sustainable tower with green architecture elements"
        ]
        results = inference.generate_images(example_prompts)
    
    # Create report
    if results:
        report_path = inference.create_generation_report(results)
        print(f"\nGeneration complete!")
        print(f"Generated {len(results)} images")
        print(f"Results saved to: {config.output_dir}")
        print(f"Report saved to: {report_path}")
    else:
        print("No images were generated successfully.")

if __name__ == "__main__":
    main()