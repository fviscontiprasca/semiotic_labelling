"""
File-based data preparation for full Flux.1d fine-tuning.
Optimized for full model training with enhanced semiotic conditioning,
working directly with JSON files instead of FiftyOne datasets.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
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
    
    # Enhanced semiotic conditioning
    require_architectural_style: bool = True
    require_urban_mood: bool = True
    enhanced_prompt_templates: bool = True

class FileBasedFluxDataPreparator:
    """File-based data preparation for full Flux.1d fine-tuning."""
    
    def __init__(self, base_path: str, output_path: str = None, config: FullTrainingDataConfig = None):
        """Initialize file-based data preparation system."""
        
        self.base_path = Path(base_path)
        # Align with full fine-tuning outputs structure
        default_output = self.base_path / "data" / "outputs" / "05_flux_full_training_data"
        self.output_path = Path(output_path) if output_path else default_output
        
        self.config = config or FullTrainingDataConfig()
        
        # Input directories
        self.images_input_dir = self.base_path / "data" / "outputs" / "01_data_pipeline" / "images"
        self.semiotic_input_dir = self.base_path / "data" / "outputs" / "04_semiotic_features"
        
        # Create output directory structure
        self.images_dir = self.output_path / "images"
        self.captions_dir = self.output_path / "captions"
        self.metadata_dir = self.output_path / "metadata"
        
        for dir_path in [self.images_dir, self.captions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced semiotic prompt templates utilizing comprehensive analysis data
        self.enhanced_prompt_templates = {
            "comprehensive_semiotic": (
                "{architectural_style} architecture from {time_period} with {dominant_colors} {color_temperature} palette, "
                "featuring {segmentation_count} distinct elements: {architectural_elements_detailed}, "
                "conveying {symbolic_meaning} in {urban_mood} {cultural_context} setting"
            ),
            "detailed_segmentation": (
                "{architectural_style} building with {segmentation_density} segmented components including {element_types}, "
                "{dominant_colors} {color_temperature} tones expressing {urban_mood} atmosphere, "
                "architectural ratio {architectural_ratio} with {cultural_indicators}"
            ),
            "color_and_elements": (
                "{color_temperature} {dominant_colors} architectural composition showing {architectural_style} design, "
                "{segmentation_count} segmented elements ({architectural_elements_detailed}) in {time_period} lighting, "
                "complexity score {complexity_score} reflecting {symbolic_meaning}"
            ),
            "spatial_detailed": (
                "Geometric {architectural_style} composition with {spatial_qualities} and {aspect_ratio} aspect ratio, "
                "featuring {element_types} across {segmentation_density} segments, "
                "{color_temperature} {dominant_colors} palette in {urban_mood} {cultural_context}"
            ),
            "semiotic_comprehensive": (
                "Comprehensive semiotic analysis: {architectural_style} architecture with {complexity_score} complexity, "
                "{segmentation_count} elements ({architectural_elements_detailed}), {dominant_colors} {color_temperature} palette, "
                "expressing {symbolic_meaning} through {urban_mood} atmosphere in {cultural_context} setting"
            ),
            "material_and_mood": (
                "{architectural_style} building from {time_period} featuring {materials} construction, "
                "{color_temperature} {dominant_colors} surfaces with {segmentation_density} architectural segments, "
                "conveying {symbolic_meaning} through {urban_mood} material expression"
            ),
            "technical_analysis": (
                "Technical architectural analysis: {architectural_style} design with {architectural_ratio} architectural ratio, "
                "{segmentation_count} segmented components including {element_types}, "
                "{complexity_score} complexity in {color_temperature} {dominant_colors} {time_period} setting"
            ),
            "cultural_context": (
                "{cultural_context} {architectural_style} architecture expressing {symbolic_meaning}, "
                "featuring {segmentation_count} elements ({architectural_elements_detailed}) with {cultural_indicators}, "
                "{dominant_colors} {color_temperature} palette creating {urban_mood} atmosphere"
            ),
            "visual_richness": (
                "Visually rich {architectural_style} composition with {descriptive_richness} richness score, "
                "{segmentation_density} architectural segments including {element_types}, "
                "{color_temperature} {dominant_colors} tones in {urban_mood} {time_period} lighting"
            ),
            "narrative_architectural": (
                "{time_period} {architectural_style} architecture telling the story of {symbolic_meaning}, "
                "through {segmentation_count} distinct elements ({architectural_elements_detailed}), "
                "{dominant_colors} {color_temperature} palette reflecting {urban_mood} {cultural_context} heritage"
            )
        }
        
        # Semiotic token vocabulary for augmentation
        self.semiotic_vocabulary = {
            "architectural_styles": [
                "modern", "contemporary", "brutalist", "classical", "gothic", "baroque",
                "art_deco", "bauhaus", "functionalist", "postmodern", "deconstructivist",
                "minimalist", "industrial", "neo-classical", "victorian", "colonial"
            ],
            "urban_moods": [
                "vibrant", "melancholic", "energetic", "serene", "dystopian", "utopian",
                "industrial", "residential", "commercial", "historic", "futuristic",
                "abandoned", "bustling", "quiet", "dramatic", "peaceful"
            ]
        }
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare enhanced training data for full Flux.1d fine-tuning."""
        
        logger.info("🚀 Starting file-based Flux.1d training data preparation")
        
        # Check input directories
        if not self.images_input_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_input_dir}")
        if not self.semiotic_input_dir.exists():
            raise FileNotFoundError(f"Semiotic features directory not found: {self.semiotic_input_dir}")
        
        # Get all semiotic feature files
        semiotic_files = list(self.semiotic_input_dir.glob("*_semiotic_features.json"))
        logger.info(f"Found {len(semiotic_files)} semiotic feature files")
        
        # Process each split
        splits_data = {}
        
        for split in ["train", "val"]:
            split_files = [f for f in semiotic_files if f.name.startswith(split)]
            logger.info(f"Processing {len(split_files)} files for {split} split")
            
            if split_files:
                splits_data[split] = self._prepare_split(split_files, split)
        
        # Generate comprehensive metadata
        metadata = self._generate_metadata(splits_data)
        
        # Save metadata
        metadata_path = self.metadata_dir / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Full Flux.1d training data preparation complete!")
        logger.info(f"📁 Output directory: {self.output_path}")
        logger.info(f"📊 Training samples: {metadata.get('train_samples', 0)}")
        logger.info(f"📊 Validation samples: {metadata.get('val_samples', 0)}")
        
        return metadata
    
    def _prepare_split(self, semiotic_files: List[Path], split_name: str) -> Dict[str, Any]:
        """Prepare data split from semiotic feature files."""
        
        split_dir = self.images_dir / split_name
        split_captions_dir = self.captions_dir / split_name
        
        split_dir.mkdir(exist_ok=True)
        split_captions_dir.mkdir(exist_ok=True)
        
        processed_samples = []
        skipped_samples = []
        
        for i, semiotic_file in enumerate(semiotic_files):
            try:
                # Load semiotic features
                with open(semiotic_file, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                
                # Extract semiotic analysis section
                semiotic_analysis = features.get('semiotic_analysis', {})
                
                # Quality filtering
                quality_reason = self._passes_quality_check_detailed(features, semiotic_analysis)
                if quality_reason:
                    skipped_samples.append({
                        'file': semiotic_file.name,
                        'reason': f'quality_check_failed: {quality_reason}'
                    })
                    continue
                
                # Find corresponding image
                image_path = self._find_image_path(features)
                if not image_path or not image_path.exists():
                    skipped_samples.append({
                        'file': semiotic_file.name,
                        'reason': 'image_not_found'
                    })
                    continue
                
                # Generate filename
                file_hash = hashlib.md5(str(semiotic_file).encode()).hexdigest()[:8]
                filename = f"{split_name}_{i:06d}_{file_hash}"
                
                # Process image
                processed_image_path = self._process_image(image_path, split_dir, filename)
                
                # Generate enhanced captions
                captions = self._generate_enhanced_captions(features, semiotic_analysis)
                
                # Save captions
                caption_file = split_captions_dir / f"{filename}.txt"
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(captions['primary'])
                
                # Save metadata for this sample
                sample_metadata = {
                    'filename': filename,
                    'original_image_path': str(image_path),
                    'processed_image_path': str(processed_image_path),
                    'caption_file': str(caption_file),
                    'semiotic_score': semiotic_analysis.get('semiotic_score', 0),
                    'architectural_style': semiotic_analysis.get('architectural_style', ''),
                    'urban_mood': semiotic_analysis.get('urban_mood', ''),
                    'materials': semiotic_analysis.get('materials', ''),
                    'cultural_context': semiotic_analysis.get('cultural_context', ''),
                    'symbolic_meaning': semiotic_analysis.get('symbolic_meaning', ''),
                    'captions': captions,
                    'quality_score': self._calculate_quality_score(features, semiotic_analysis)
                }
                
                processed_samples.append(sample_metadata)
                
            except Exception as e:
                logger.error(f"Error processing {semiotic_file.name}: {e}")
                skipped_samples.append({
                    'file': semiotic_file.name,
                    'reason': f'processing_error: {str(e)}'
                })
        
        split_data = {
            'samples': processed_samples,
            'skipped': skipped_samples,
            'total_processed': len(processed_samples),
            'total_skipped': len(skipped_samples)
        }
        
        logger.info(f"Split {split_name}: {len(processed_samples)} processed, {len(skipped_samples)} skipped")
        
        # Log sample skip reasons for debugging
        if skipped_samples:
            reason_counts = {}
            for skip in skipped_samples[:20]:  # Show first 20 reasons
                reason = skip['reason']
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if len(reason_counts) <= 5:  # Show first 5 unique reasons
                    logger.info(f"  Skip example: {skip['file']} - {reason}")
            
            logger.info(f"  Top skip reasons: {dict(list(reason_counts.items())[:5])}")
        
        return split_data
    
    def _passes_quality_check(self, features: Dict, semiotic_analysis: Dict) -> bool:
        """Enhanced quality filtering for full fine-tuning."""
        return self._passes_quality_check_detailed(features, semiotic_analysis) is None
    
    def _passes_quality_check_detailed(self, features: Dict, semiotic_analysis: Dict) -> Optional[str]:
        """Enhanced quality filtering with detailed failure reasons."""
        
        # Check semiotic score
        semiotic_score = semiotic_analysis.get("semiotic_score", 0)
        if semiotic_score < self.config.min_semiotic_score:
            return f"semiotic_score_too_low_{semiotic_score:.3f}"
        
        # Check required architectural elements
        if self.config.require_architectural_style:
            style = semiotic_analysis.get("architectural_style", "")
            if not style:
                return "missing_architectural_style"
            if style not in self.semiotic_vocabulary["architectural_styles"]:
                return f"invalid_architectural_style_{style}"
        
        if self.config.require_urban_mood:
            mood = semiotic_analysis.get("urban_mood", "")
            if not mood:
                return "missing_urban_mood"
            if mood not in self.semiotic_vocabulary["urban_moods"]:
                return f"invalid_urban_mood_{mood}"
        
        # Check for critical required elements (relaxed for materials since they're empty lists)
        critical_elements = ["cultural_context", "symbolic_meaning"]
        for element in critical_elements:
            value = semiotic_analysis.get(element)
            if not value or (isinstance(value, str) and value.strip() == ""):
                return f"missing_{element}"
        
        # Materials can be empty for now - we'll use fallback values
        # TODO: Fix semiotic extractor to properly extract materials
        
        return None
    
    def _find_image_path(self, features: Dict) -> Optional[Path]:
        """Find the corresponding image file."""
        
        # Try different ways to get image path
        image_path_str = features.get('image_path') or features.get('filepath')
        
        if image_path_str:
            # If absolute path exists
            image_path = Path(image_path_str)
            if image_path.exists():
                return image_path
            
            # Try relative to our images directory
            filename = Path(image_path_str).name
            for split in ["train", "val"]:
                candidate = self.images_input_dir / split / filename
                if candidate.exists():
                    return candidate
        
        # Try based on filename from features
        filename = features.get('filename')
        if filename:
            # Remove _semiotic_features suffix and add image extension
            base_name = filename.replace('_semiotic_features', '')
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                for split in ["train", "val"]:
                    candidate = self.images_input_dir / split / f"{base_name}{ext}"
                    if candidate.exists():
                        return candidate
        
        return None
    
    def _process_image(self, input_path: Path, output_dir: Path, filename: str) -> Path:
        """Process and optimize image for training."""
        
        # Determine output format
        if self.config.use_webp_compression:
            output_path = output_dir / f"{filename}.webp"
        else:
            output_path = output_dir / f"{filename}.jpg"
        
        # Load and process image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if needed
            width, height = img.size
            if width < self.config.min_image_size[0] or height < self.config.min_image_size[1]:
                # Upscale small images
                scale_factor = max(
                    self.config.min_image_size[0] / width,
                    self.config.min_image_size[1] / height
                )
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            elif width > self.config.max_image_size[0] or height > self.config.max_image_size[1]:
                # Downscale large images
                scale_factor = min(
                    self.config.max_image_size[0] / width,
                    self.config.max_image_size[1] / height
                )
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save processed image
            if self.config.use_webp_compression:
                img.save(output_path, 'WEBP', quality=self.config.webp_quality, optimize=True)
            else:
                img.save(output_path, 'JPEG', quality=90, optimize=True)
        
        return output_path
    
    def _generate_enhanced_captions(self, features: Dict, semiotic_analysis: Optional[Dict] = None) -> Dict[str, str]:
        """Generate multiple enhanced captions utilizing rich semiotic analysis."""
        
        # Get base caption from various sources
        base_caption = ""
        if 'textual' in features:
            base_caption = features['textual'].get('blip2_caption') or features['textual'].get('caption', '')
        
        captions = {}
        
        # Extract nested semiotic analysis from features if not provided separately
        if semiotic_analysis is None and 'semiotic_analysis' in features:
            semiotic_analysis = features['semiotic_analysis']
        elif semiotic_analysis is None:
            semiotic_analysis = {}
        
        # Extract rich semiotic data for template population
        template_data = self._extract_rich_template_data(features, semiotic_analysis)
        
        if self.config.enhanced_prompt_templates:
            # Generate diverse captions using different templates and data combinations
            template_options = list(self.enhanced_prompt_templates.keys())
            
            # Select template based on available data richness and variety
            selected_template = self._select_optimal_template(template_data, semiotic_analysis)
            
            try:
                primary_caption = self.enhanced_prompt_templates[selected_template].format(**template_data)
            except KeyError as e:
                logger.warning(f"Template formatting error with {selected_template}: {e}")
                # Fallback to a simpler but more reliable template
                primary_caption = self._generate_fallback_caption(template_data)
        else:
            primary_caption = base_caption
        
        captions['primary'] = primary_caption
        captions['base'] = base_caption
        
        # Generate additional caption variations for diversity
        if self.config.use_caption_augmentation:
            captions.update(self._generate_caption_variations(template_data, semiotic_analysis))
        
        return captions
    
    def _extract_rich_template_data(self, features: Dict, semiotic_analysis: Optional[Dict]) -> Dict[str, str]:
        """Extract comprehensive rich data from semiotic analysis for template population."""
        
        # Ensure semiotic_analysis is not None
        if not semiotic_analysis:
            semiotic_analysis = {}
        
        # Basic architectural data
        architectural_style = semiotic_analysis.get('architectural_style', 'contemporary')
        urban_mood = semiotic_analysis.get('urban_mood', 'urban')
        cultural_context = semiotic_analysis.get('cultural_context', 'urban')
        symbolic_meaning = semiotic_analysis.get('symbolic_meaning', 'architectural expression')
        time_period = semiotic_analysis.get('time_period', 'daytime')
        
        # Handle materials
        materials = semiotic_analysis.get('materials', [])
        if isinstance(materials, list):
            materials_str = ', '.join(materials) if materials else 'mixed materials'
        else:
            materials_str = materials if materials else 'mixed materials'
        
        # Extract color information
        dominant_colors = self._extract_color_description(features, semiotic_analysis)
        color_temperature = semiotic_analysis.get('color_temperature', 'neutral')
        
        # Segmentation and architectural element data
        architectural_types_detected = semiotic_analysis.get('architectural_types_detected', [])
        segmentation_density = semiotic_analysis.get('segmentation_density', 0)
        architectural_ratio = semiotic_analysis.get('architectural_segment_ratio', 0.0)
        
        # Count of segmented elements
        segmentation_count = len(architectural_types_detected)
        
        # Element types description
        element_types = self._format_architectural_types(architectural_types_detected)
        
        # Detailed architectural elements with positioning
        architectural_elements_detailed = self._extract_detailed_architectural_elements(features, architectural_types_detected)
        
        # Cultural indicators
        cultural_indicators = semiotic_analysis.get('cultural_indicators', [])
        cultural_indicators_str = ', '.join(cultural_indicators) if cultural_indicators else 'urban cultural elements'
        
        # Complexity and richness scores
        complexity_score = round(semiotic_analysis.get('complexity_score', 0.0), 2)
        descriptive_richness = round(semiotic_analysis.get('descriptive_richness', 0.0), 2)
        
        # Spatial analysis
        spatial_info = features.get('spatial', {})
        aspect_ratio = round(spatial_info.get('aspect_ratio', 1.0), 2)
        dimensions = spatial_info.get('dimensions', {})
        width = dimensions.get('width', 0)
        height = dimensions.get('height', 0)
        
        # Generate descriptions
        spatial_qualities = self._generate_spatial_description_enhanced(spatial_info, architectural_ratio)
        architectural_elements = self._extract_architectural_elements(features)
        
        # Enhanced color description
        color_description = f"{color_temperature} {dominant_colors}"
        
        # Visual elements summary
        visual_elements = self._summarize_visual_elements(features, semiotic_analysis)
        
        # Complexity description
        complexity_description = self._generate_complexity_description(features, semiotic_analysis)
        
        # Visual complexity assessment
        visual_complexity = self._assess_visual_complexity(features, semiotic_analysis)
        
        return {
            'architectural_style': architectural_style,
            'urban_mood': urban_mood,
            'materials': materials_str,
            'cultural_context': cultural_context,
            'symbolic_meaning': symbolic_meaning,
            'time_period': time_period,
            'dominant_colors': dominant_colors,
            'color_temperature': color_temperature,
            'segmentation_count': str(segmentation_count),
            'segmentation_density': str(segmentation_density),
            'architectural_ratio': f"{architectural_ratio:.2f}",
            'element_types': element_types,
            'architectural_elements_detailed': architectural_elements_detailed,
            'cultural_indicators': cultural_indicators_str,
            'complexity_score': str(complexity_score),
            'descriptive_richness': str(descriptive_richness),
            'aspect_ratio': f"{aspect_ratio}:1",
            'spatial_qualities': spatial_qualities,
            'architectural_elements': architectural_elements,
            'color_description': color_description,
            'visual_elements': visual_elements,
            'visual_complexity': visual_complexity,
            'complexity_description': complexity_description,
            'filename': semiotic_analysis.get('filename', features.get('filename', 'default'))
        }
    
    def _extract_color_description(self, features: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Extract rich color description from visual analysis."""
        
        # Try to get from semiotic analysis first
        if semiotic_analysis and 'dominant_colors' in semiotic_analysis:
            colors = semiotic_analysis['dominant_colors']
            if isinstance(colors, list) and colors:
                return ', '.join(colors[:3])  # Top 3 colors
        
        # Fallback to visual analysis
        if 'visual' in features and 'color_palette' in features['visual']:
            palette = features['visual']['color_palette'].get('palette', [])
            if palette:
                color_names = []
                for color_info in palette[:3]:  # Top 3 colors
                    hex_color = color_info.get('hex', '')
                    if hex_color:
                        color_names.append(hex_color)
                return ', '.join(color_names) if color_names else 'mixed colors'
        
        return 'mixed colors'
    
    def _extract_architectural_elements(self, features: Dict) -> str:
        """Extract architectural elements from segmentation analysis."""
        
        elements = []
        
        if 'semiotic_segmentation' in features:
            segments = features['semiotic_segmentation']
            for segment in segments[:5]:  # Top 5 segments
                arch_type = segment.get('architectural_type', '')
                if arch_type and arch_type not in elements:
                    # Convert technical terms to descriptive ones
                    descriptive_name = self._convert_to_descriptive(arch_type)
                    if descriptive_name:
                        elements.append(descriptive_name)
        
        if not elements:
            elements = ['structural elements', 'facade details']
        
        return ', '.join(elements[:3])  # Limit to 3 elements
    
    def _format_architectural_types(self, architectural_types: List[str]) -> str:
        """Format architectural types into readable description."""
        
        if not architectural_types:
            return 'mixed architectural elements'
        
        # Convert technical terms to descriptive ones
        descriptive_types = []
        for arch_type in architectural_types[:5]:  # Limit to top 5
            descriptive = self._convert_to_descriptive(arch_type)
            if descriptive and descriptive not in descriptive_types:
                descriptive_types.append(descriptive)
        
        if len(descriptive_types) == 1:
            return descriptive_types[0]
        elif len(descriptive_types) == 2:
            return f"{descriptive_types[0]} and {descriptive_types[1]}"
        else:
            return ', '.join(descriptive_types[:-1]) + f", and {descriptive_types[-1]}"
    
    def _extract_detailed_architectural_elements(self, features: Dict, architectural_types: List[str]) -> str:
        """Extract detailed architectural elements with positioning information."""
        
        elements_detail = []
        
        # Get segmentation data if available
        if 'semiotic_segmentation' in features:
            segments = features['semiotic_segmentation'][:8]  # Top 8 segments
            
            element_counts = {}
            for segment in segments:
                arch_type = segment.get('architectural_type', '')
                if arch_type:
                    descriptive_name = self._convert_to_descriptive(arch_type)
                    if descriptive_name:
                        element_counts[descriptive_name] = element_counts.get(descriptive_name, 0) + 1
            
            # Format with counts
            for element, count in list(element_counts.items())[:4]:  # Top 4 element types
                if count > 1:
                    elements_detail.append(f"{count} {element}")
                else:
                    elements_detail.append(element)
        
        # Fallback to architectural types if no segmentation
        if not elements_detail and architectural_types:
            for arch_type in architectural_types[:3]:
                descriptive = self._convert_to_descriptive(arch_type)
                if descriptive:
                    elements_detail.append(descriptive)
        
        # Default fallback
        if not elements_detail:
            elements_detail = ['structural components', 'facade elements']
        
        return ', '.join(elements_detail)
    
    def _generate_spatial_description_enhanced(self, spatial_info: Dict, architectural_ratio: float) -> str:
        """Generate enhanced spatial description with architectural ratio."""
        
        descriptions = []
        
        # Aspect ratio analysis
        aspect_ratio = spatial_info.get('aspect_ratio', 1.0)
        if aspect_ratio > 1.5:
            descriptions.append('horizontal emphasis')
        elif aspect_ratio < 0.7:
            descriptions.append('vertical emphasis')
        else:
            descriptions.append('balanced proportions')
        
        # Architectural ratio analysis
        if architectural_ratio > 0.8:
            descriptions.append('architecturally dominant composition')
        elif architectural_ratio > 0.5:
            descriptions.append('moderate architectural presence')
        else:
            descriptions.append('contextual architectural integration')
        
        # Edge density for complexity
        edge_density = spatial_info.get('edge_density', 0)
        if edge_density > 0.3:
            descriptions.append('high geometric complexity')
        elif edge_density > 0.15:
            descriptions.append('moderate structural detail')
        else:
            descriptions.append('clean geometric forms')
        
        return ', '.join(descriptions[:2])  # Limit to 2 main descriptions
    
    def _convert_to_descriptive(self, technical_term: str) -> str:
        """Convert technical architectural terms to descriptive language."""
        
        conversion_map = {
            'structural_element': 'structural components',
            'facade_element': 'facade details',
            'window_element': 'window systems',
            'decorative_element': 'decorative features',
            'vertical_element': 'vertical structures',
            'horizontal_element': 'horizontal elements',
            'detail_or_texture': 'textural surfaces',
            'window_or_opening': 'openings and apertures',
            'urban_furniture': 'urban fixtures',
            'signage_or_graphics': 'graphic elements',
            'roofing_element': 'roofing structures',
            'entrance_element': 'entrance features',
            'balcony_element': 'balcony structures',
            'column_element': 'columnar elements'
        }
        
        return conversion_map.get(technical_term, technical_term.replace('_', ' '))
    
    def _generate_complexity_description(self, features: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Generate description of visual complexity."""
        
        if not semiotic_analysis:
            return 'architectural composition'
        
        # Check segmentation density
        seg_density = semiotic_analysis.get('segmentation_density', 0)
        
        if seg_density > 100:
            return 'highly detailed composition'
        elif seg_density > 50:
            return 'moderately complex design'
        else:
            return 'simplified geometric forms'
    
    def _generate_spatial_description(self, features: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Generate spatial qualities description."""
        
        if not semiotic_analysis:
            return 'balanced proportional relationships'
        
        # Base spatial qualities on architectural style and mood
        style = semiotic_analysis.get('architectural_style', 'contemporary')
        mood = semiotic_analysis.get('urban_mood', 'urban')
        
        spatial_map = {
            ('contemporary', 'industrial'): 'clean geometric lines',
            ('modern', 'vibrant'): 'dynamic spatial flow',
            ('brutalist', 'dramatic'): 'monumental mass composition',
            ('minimalist', 'serene'): 'essential spatial clarity'
        }
        
        return spatial_map.get((style, mood), 'balanced proportional relationships')
    
    def _generate_texture_description(self, features: Dict) -> str:
        """Generate texture qualities description."""
        
        # Analyze color palette variance for texture indication
        if 'visual' in features and 'color_palette' in features['visual']:
            palette = features['visual']['color_palette'].get('palette', [])
            if len(palette) > 5:
                return 'varied textural'
            elif len(palette) > 3:
                return 'moderately textured'
        
        return 'smooth finished'
    
    def _summarize_visual_elements(self, features: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Summarize key visual elements."""
        
        elements = []
        
        # Add architectural elements
        arch_elements = self._extract_architectural_elements(features)
        if arch_elements != 'structural elements, facade details':
            elements.append(arch_elements)
        
        # Add material qualities
        if semiotic_analysis:
            materials = semiotic_analysis.get('materials', [])
            if materials:
                elements.append('material contrasts')
        
        # Add spatial elements
        elements.append('spatial composition')
        
        return ', '.join(elements[:3])
    
    def _extract_segmentation_elements(self, features: Dict) -> str:
        """Extract elements specifically from segmentation analysis."""
        
        if 'semiotic_segmentation' in features:
            segments = features['semiotic_segmentation']
            element_types = set()
            
            for segment in segments[:8]:  # Check more segments for variety
                arch_type = segment.get('architectural_type', '')
                if arch_type:
                    element_types.add(self._convert_to_descriptive(arch_type))
            
            if element_types:
                return ', '.join(list(element_types)[:3])
        
        return 'architectural components'
    
    def _assess_visual_complexity(self, features: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Assess overall visual complexity level."""
        
        complexity_factors = 0
        
        if semiotic_analysis:
            # Check segmentation density
            seg_density = semiotic_analysis.get('segmentation_density', 0)
            if seg_density > 100:
                complexity_factors += 2
            elif seg_density > 50:
                complexity_factors += 1
            
            # Check architectural segment ratio
            arch_ratio = semiotic_analysis.get('architectural_segment_ratio', 0)
            if arch_ratio > 0.8:
                complexity_factors += 1
        
        # Check color palette diversity
        if 'visual' in features and 'color_palette' in features['visual']:
            palette_count = len(features['visual']['color_palette'].get('palette', []))
            if palette_count > 6:
                complexity_factors += 1
        
        if complexity_factors >= 3:
            return 'high visual complexity'
        elif complexity_factors >= 2:
            return 'moderate complexity'
        else:
            return 'visual simplicity'
    
    def _select_optimal_template(self, template_data: Dict, semiotic_analysis: Optional[Dict]) -> str:
        """Select the best template based on available data and variety needs."""
        
        # Create variety by rotating through templates based on filename hash
        filename = template_data.get('filename', 'default')
        template_options = list(self.enhanced_prompt_templates.keys())
        
        # Use filename hash to ensure consistent but varied template selection
        import hashlib
        hash_value = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
        template_index = hash_value % len(template_options)
        
        return template_options[template_index]
    
    def _generate_fallback_caption(self, template_data: Dict) -> str:
        """Generate a reliable fallback caption when template formatting fails."""
        
        return (f"{template_data.get('architectural_style', 'contemporary')} architecture "
                f"with {template_data.get('dominant_colors', 'mixed colors')} palette, "
                f"featuring {template_data.get('architectural_elements', 'structural elements')} "
                f"in {template_data.get('urban_mood', 'urban')} atmosphere")
    
    def _generate_caption_variations(self, template_data: Dict, semiotic_analysis: Optional[Dict]) -> Dict[str, str]:
        """Generate additional caption variations for training diversity."""
        
        variations = {}
        
        # Style-focused variation
        variations['style_focused'] = (
            f"{template_data['architectural_style']} architecture with "
            f"{template_data['visual_complexity']}, featuring {template_data['architectural_elements']}"
        )
        
        # Color-focused variation  
        variations['color_focused'] = (
            f"Architectural photography with {template_data['color_description']} tones, "
            f"showcasing {template_data['architectural_style']} design elements"
        )
        
        # Context-focused variation
        variations['contextual'] = (
            f"{template_data['cultural_context']} architectural scene with "
            f"{template_data['spatial_qualities']}, {template_data['time_period']} lighting"
        )
        
        return variations
    
    def _calculate_quality_score(self, features: Dict, semiotic_analysis: Dict) -> float:
        """Calculate comprehensive quality score."""
        
        score = 0.0
        
        # Semiotic score (40% weight)
        semiotic_score = semiotic_analysis.get('semiotic_score', 0)
        score += semiotic_score * 0.4
        
        # Completeness of semiotic fields (30% weight)
        required_fields = ['architectural_style', 'urban_mood', 'materials', 'cultural_context', 'symbolic_meaning']
        completeness = sum(1 for field in required_fields if semiotic_analysis.get(field)) / len(required_fields)
        score += completeness * 0.3
        
        # Caption quality (20% weight)
        if 'textual' in features and 'blip2_caption' in features['textual']:
            caption = features['textual']['blip2_caption']
            caption_score = min(len(caption.split()) / 20, 1.0)  # Normalize to 20 words
            score += caption_score * 0.2
        
        # Visual complexity (10% weight)
        if 'visual' in features and 'color_palette' in features['visual']:
            palette = features['visual']['color_palette'].get('palette', [])
            complexity_score = min(len(palette) / 10, 1.0)  # Normalize to 10 colors
            score += complexity_score * 0.1
        
        return min(score, 1.0)
    
    def _generate_metadata(self, splits_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive training metadata."""
        
        metadata = {
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'config': {
                'min_semiotic_score': self.config.min_semiotic_score,
                'min_image_size': self.config.min_image_size,
                'max_image_size': self.config.max_image_size,
                'target_resolution': self.config.target_resolution,
                'use_webp_compression': self.config.use_webp_compression,
                'webp_quality': self.config.webp_quality,
                'enhanced_prompt_templates': self.config.enhanced_prompt_templates
            },
            'splits': {}
        }
        
        total_samples = 0
        total_skipped = 0
        
        for split_name, split_data in splits_data.items():
            split_samples = split_data['samples']
            split_skipped = split_data['skipped']
            
            # Calculate statistics
            if split_samples:
                semiotic_scores = [s['semiotic_score'] for s in split_samples]
                quality_scores = [s['quality_score'] for s in split_samples]
                
                architectural_styles = [s['architectural_style'] for s in split_samples if s['architectural_style']]
                urban_moods = [s['urban_mood'] for s in split_samples if s['urban_mood']]
                
                split_metadata = {
                    'total_samples': len(split_samples),
                    'skipped_samples': len(split_skipped),
                    'semiotic_score_stats': {
                        'mean': np.mean(semiotic_scores),
                        'std': np.std(semiotic_scores),
                        'min': np.min(semiotic_scores),
                        'max': np.max(semiotic_scores)
                    },
                    'quality_score_stats': {
                        'mean': np.mean(quality_scores),
                        'std': np.std(quality_scores),
                        'min': np.min(quality_scores),
                        'max': np.max(quality_scores)
                    },
                    'architectural_styles': list(set(architectural_styles)),
                    'urban_moods': list(set(urban_moods)),
                    'style_distribution': {style: architectural_styles.count(style) for style in set(architectural_styles)},
                    'mood_distribution': {mood: urban_moods.count(mood) for mood in set(urban_moods)}
                }
            else:
                split_metadata = {
                    'total_samples': 0,
                    'skipped_samples': len(split_skipped),
                    'error': 'No valid samples found'
                }
            
            metadata['splits'][split_name] = split_metadata
            total_samples += len(split_samples)
            total_skipped += len(split_skipped)
        
        metadata['summary'] = {
            'total_samples': total_samples,
            'total_skipped': total_skipped,
            'processing_success_rate': total_samples / (total_samples + total_skipped) if (total_samples + total_skipped) > 0 else 0
        }
        
        return metadata

def main():
    """Main execution function for standalone use."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare file-based Flux.1d training data")
    parser.add_argument("--base_path", type=str, default=".", help="Base path to the project")
    parser.add_argument("--output_path", type=str, help="Output path for training data")
    parser.add_argument("--min_semiotic_score", type=float, default=0.5, help="Minimum semiotic score")
    parser.add_argument("--webp_compression", action="store_true", help="Use WebP compression")
    
    args = parser.parse_args()
    
    # Create config
    config = FullTrainingDataConfig(
        min_semiotic_score=args.min_semiotic_score,
        use_webp_compression=args.webp_compression
    )
    
    # Initialize preparator
    preparator = FileBasedFluxDataPreparator(
        base_path=args.base_path,
        output_path=args.output_path,
        config=config
    )
    
    # Prepare training data
    metadata = preparator.prepare_training_data()
    
    print(f"\n✅ Training data preparation complete!")
    print(f"📊 Total samples: {metadata['summary']['total_samples']}")
    print(f"📁 Output: {preparator.output_path}")

if __name__ == "__main__":
    main()