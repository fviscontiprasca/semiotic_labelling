#!/usr/bin/env python3
"""
Simple Semiotic Feature Extractor for Urban Architecture Dataset
Processes images directly from filesystem without problematic dependencies
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class SimpleSemioticExtractor:
    """Extract basic semiotic features from urban architecture images."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the semiotic extractor."""
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load required models."""
        logger.info("Loading models...")
        
        # Load sentence transformer for textual analysis
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Sentence Transformer loaded")
        
        # Define architectural concepts for analysis
        self._define_architectural_concepts()
        
        logger.info("✓ All models loaded successfully")
    
    def _define_architectural_concepts(self):
        """Define architectural concepts and terms for analysis."""
        
        self.architectural_styles = [
            "modernist", "brutalist", "classical", "gothic", "baroque", "art deco",
            "international style", "bauhaus", "postmodern", "deconstructivist",
            "vernacular", "traditional", "contemporary", "minimalist", "industrial"
        ]
        
        self.urban_elements = [
            "residential building", "office tower", "commercial center", "public space",
            "transportation hub", "cultural building", "educational facility",
            "healthcare building", "religious building", "mixed-use development"
        ]
        
        self.materials = [
            "concrete", "glass", "steel", "brick", "stone", "wood", "metal",
            "composite", "ceramic", "plastic", "fabric", "membrane"
        ]
        
        self.spatial_qualities = [
            "open", "enclosed", "transparent", "opaque", "vertical", "horizontal",
            "monumental", "intimate", "dynamic", "static", "fragmented", "unified"
        ]
        
        self.cultural_indicators = [
            "luxury", "affordable", "institutional", "commercial", "domestic",
            "sacred", "secular", "public", "private", "formal", "informal"
        ]
        
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def safe_normalize(self, vector, epsilon=1e-8):
        """Safely normalize a vector, handling zero vectors."""
        norm = np.linalg.norm(vector)
        if norm < epsilon:
            return vector
        return vector / norm
    
    def process_images_direct(self, data_dir: str, output_dir: str):
        """Process images directly from data directory."""
        
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for split in ['train', 'val']:
            split_dir = data_path / 'outputs' / '01_data_pipeline' / 'images' / split
            if split_dir.exists():
                for img_file in split_dir.glob('*.jpg'):
                    image_files.append((img_file, split))
                for img_file in split_dir.glob('*.png'):
                    image_files.append((img_file, split))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        if len(image_files) == 0:
            logger.error("No images found in the data directory")
            return
        
        # Process each image
        all_results = []
        
        for i, (img_path, split) in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
                
                # Load and process image
                features = self._extract_features_from_image(img_path, split)
                
                if features:
                    # Add metadata
                    features['image_path'] = str(img_path)
                    features['split'] = split
                    features['filename'] = img_path.name
                    
                    all_results.append(features)
                    
                    # Save individual result
                    result_file = output_path / f"{img_path.stem}_semiotic_features.json"
                    with open(result_file, 'w') as f:
                        json.dump(self.convert_numpy_types(features), f, indent=2)
                
                # Progress update every 50 images
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Save combined results
        if all_results:
            combined_file = output_path / "all_semiotic_features.json"
            with open(combined_file, 'w') as f:
                json.dump(self.convert_numpy_types(all_results), f, indent=2)
            
            logger.info(f"✓ Processing complete! Saved {len(all_results)} results to {output_path}")
            
            # Generate summary statistics
            self._generate_summary_stats(all_results, output_path)
        else:
            logger.error("No results generated")
    
    def _extract_features_from_image(self, img_path: Path, split: str) -> Optional[Dict[str, Any]]:
        """Extract semiotic features from a single image."""
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Extract visual features (without CLIP)
            visual_features = self._extract_visual_features(image)
            
            # Extract textual features (from captions)
            textual_features = self._extract_textual_features(img_path, split)
            
            # Extract segmentation features (from SAM analysis)
            segmentation_features = self._extract_segmentation_features(img_path, split)
            
            # Extract spatial features
            spatial_features = self._extract_spatial_features(image)
            
            # Perform semiotic analysis
            semiotic_analysis = self._perform_semiotic_analysis(
                textual_features, visual_features, spatial_features, segmentation_features
            )
            
            # Combine all features
            combined_features = {
                "visual": visual_features,
                "textual": textual_features,
                "segmentation": segmentation_features,
                "spatial": spatial_features,
                "semiotic_analysis": semiotic_analysis,
                "metadata": {
                    "image_size": image.size,
                    "split": split,
                    "processing_device": self.device
                }
            }
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {img_path}: {e}")
            return None
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features without CLIP."""
        
        features = {}
        
        try:
            # Extract color palette
            features["color_palette"] = self._extract_color_palette(image)
            
            # Extract basic visual statistics
            img_array = np.array(image)
            features["brightness"] = float(np.mean(img_array))
            features["contrast"] = float(np.std(img_array))
            
            # Channel statistics
            if len(img_array.shape) == 3:
                features["color_channels"] = {
                    "red_mean": float(np.mean(img_array[:,:,0])),
                    "green_mean": float(np.mean(img_array[:,:,1])),
                    "blue_mean": float(np.mean(img_array[:,:,2])),
                    "red_std": float(np.std(img_array[:,:,0])),
                    "green_std": float(np.std(img_array[:,:,1])),
                    "blue_std": float(np.std(img_array[:,:,2]))
                }
                
        except Exception as e:
            logger.error(f"Error in visual feature extraction: {e}")
            features["error"] = str(e)
            
        return features
    
    def _extract_textual_features(self, img_path: Path, split: str) -> Dict[str, Any]:
        """Extract textual features from image metadata and captions."""
        
        features = {}
        
        try:
            # Basic metadata
            features["filename"] = img_path.name
            features["split"] = split
            
            # Load captions from centralized JSON file
            captions_path = Path("data/outputs/02_blip2_captioner/captions.json")
            
            if captions_path.exists():
                # Load captions if not already cached
                if not hasattr(self, '_captions_cache'):
                    with open(captions_path, 'r', encoding='utf-8') as f:
                        self._captions_cache = json.load(f)
                
                # Construct the key as it appears in the captions file
                img_key = f"{split}/{img_path.name}"
                
                if img_key in self._captions_cache:
                    caption_data = self._captions_cache[img_key]
                    
                    # Extract the main caption (architectural_analysis seems to be the primary one)
                    if 'architectural_analysis' in caption_data:
                        caption = caption_data['architectural_analysis']
                        features["caption"] = caption
                        features["caption_embedding"] = self.sentence_model.encode(caption).tolist()
                        features["semantic_analysis"] = self._analyze_semantic_content(caption)
                        
                        # Store additional caption types
                        features["architectural_analysis"] = caption
                        if 'mood_atmosphere' in caption_data:
                            features["mood_atmosphere"] = caption_data['mood_atmosphere']
                        if 'technical_analysis' in caption_data:
                            features["technical_analysis"] = caption_data['technical_analysis']
                        if 'urban_context' in caption_data:
                            features["urban_context"] = caption_data['urban_context']
                    else:
                        features["caption"] = None
                        logger.debug(f"No architectural_analysis caption found for {img_path.name}")
                else:
                    features["caption"] = None
                    logger.debug(f"No caption data found for key {img_key}")
                    
            else:
                features["caption"] = None
                logger.warning(f"Captions file not found at {captions_path}")
                
        except Exception as e:
            logger.error(f"Error in textual feature extraction: {e}")
            features["error"] = str(e)
            
        return features
    
    def _extract_segmentation_features(self, img_path: Path, split: str) -> Dict[str, Any]:
        """Extract features from SAM segmentation analysis."""
        
        features = {
            'total_segments': 0,
            'architectural_segments': 0,
            'segmentation_complexity': 0.0,
            'average_segment_size': 0.0,
            'segment_size_variance': 0.0,
            'architectural_types': [],
            'segment_analysis': None
        }
        
        try:
            # Construct path to SAM analysis file
            sam_analysis_path = Path("data/outputs/03_sam_segmentation/analysis") / f"{img_path.stem}_sam_analysis.json"
            
            if sam_analysis_path.exists():
                with open(sam_analysis_path, 'r', encoding='utf-8') as f:
                    sam_data = json.load(f)
                
                # Extract basic segmentation statistics
                if 'segment_analysis' in sam_data:
                    segment_analysis = sam_data['segment_analysis']
                    features['total_segments'] = segment_analysis.get('total_segments', 0)
                    features['architectural_segments'] = segment_analysis.get('architectural_count', 0)
                    
                    # Extract architectural types from segments
                    architectural_types = []
                    segment_areas = []
                    
                    if 'architectural_segments' in segment_analysis:
                        for segment in segment_analysis['architectural_segments']:
                            if 'architectural_type' in segment:
                                arch_type = segment['architectural_type']
                                if arch_type not in architectural_types:
                                    architectural_types.append(arch_type)
                            
                            # Collect segment areas for statistics
                            if 'area' in segment:
                                segment_areas.append(segment['area'])
                    
                    features['architectural_types'] = architectural_types
                    
                    # Calculate segmentation complexity and statistics
                    if segment_areas:
                        features['average_segment_size'] = float(np.mean(segment_areas))
                        features['segment_size_variance'] = float(np.var(segment_areas))
                        features['segmentation_complexity'] = len(segment_areas) / max(segment_areas) if segment_areas else 0.0
                    
                    # Store the full segment analysis for detailed analysis
                    features['segment_analysis'] = segment_analysis
                    
                else:
                    logger.debug(f"No segment_analysis found in SAM data for {img_path.name}")
                    
            else:
                logger.debug(f"No SAM analysis file found for {img_path.name}")
                
        except Exception as e:
            logger.error(f"Error extracting segmentation features for {img_path}: {e}")
            features["error"] = str(e)
            
        return features
    
    def _extract_spatial_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract spatial and compositional features."""
        
        features = {}
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic spatial properties
            height, width = img_array.shape[:2]
            features["aspect_ratio"] = width / height
            features["dimensions"] = {"width": width, "height": height}
            
            # Edge detection for complexity analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            features["edge_density"] = edge_density
            
            # Composition analysis
            features["composition"] = self._analyze_composition(img_array)
            
        except Exception as e:
            logger.error(f"Error in spatial feature extraction: {e}")
            features["error"] = str(e)
            
        return features
    
    def _extract_color_palette(self, image: Image.Image, n_colors: int = 5) -> Dict[str, Any]:
        """Extract dominant color palette."""
        
        try:
            # Downsize image for faster processing
            small_image = image.resize((150, 150))
            img_array = np.array(small_image)
            
            # Reshape for clustering
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels to speed up clustering
            n_samples = min(1000, len(pixels))
            sampled_pixels = pixels[np.random.choice(len(pixels), n_samples, replace=False)]
            
            # Perform k-means clustering
            try:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=100)
                kmeans.fit(sampled_pixels)
                
                colors = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_
                
                # Calculate color frequencies
                unique_labels, counts = np.unique(labels, return_counts=True)
                frequencies = counts / len(labels)
                
                palette = []
                for i, (color, freq) in enumerate(zip(colors, frequencies)):
                    palette.append({
                        "color": color.tolist(),
                        "frequency": float(freq),
                        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    })
                
                return {
                    "palette": palette,
                    "dominant_color": colors[0].tolist(),
                    "color_diversity": float(len(unique_labels))
                }
                
            except Exception as e:
                logger.warning(f"K-means clustering failed: {e}")
                return {"error": "Color extraction failed"}
                
        except Exception as e:
            logger.error(f"Error in color palette extraction: {e}")
            return {"error": str(e)}
    
    def _analyze_composition(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition and spatial layout."""
        
        composition = {}
        
        try:
            height, width = img_array.shape[:2]
            
            # Rule of thirds analysis
            third_h = height // 3
            third_w = width // 3
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate intensity in different regions
            regions = {
                "top_left": gray[:third_h, :third_w],
                "top_center": gray[:third_h, third_w:2*third_w],
                "top_right": gray[:third_h, 2*third_w:],
                "center_left": gray[third_h:2*third_h, :third_w],
                "center": gray[third_h:2*third_h, third_w:2*third_w],
                "center_right": gray[third_h:2*third_h, 2*third_w:],
                "bottom_left": gray[2*third_h:, :third_w],
                "bottom_center": gray[2*third_h:, third_w:2*third_w],
                "bottom_right": gray[2*third_h:, 2*third_w:]
            }
            
            region_stats = {}
            for region_name, region in regions.items():
                region_stats[region_name] = {
                    "mean_intensity": float(np.mean(region)),
                    "std_intensity": float(np.std(region))
                }
            
            composition["regions"] = region_stats
            
            # Overall composition metrics
            composition["overall_brightness"] = float(np.mean(gray))
            composition["contrast"] = float(np.std(gray))
            
        except Exception as e:
            logger.error(f"Error in composition analysis: {e}")
            composition["error"] = str(e)
            
        return composition
    
    def _analyze_semantic_content(self, caption: str) -> Dict[str, Any]:
        """Analyze semantic content of caption."""
        
        analysis = {}
        
        try:
            # Basic text analysis
            words = caption.lower().split()
            analysis["word_count"] = len(words)
            analysis["unique_words"] = len(set(words))
            
            # Check for architectural terms
            arch_terms = []
            all_arch_terms = [term.lower() for term in self.architectural_styles + self.urban_elements + self.materials]
            
            for word in words:
                if word in all_arch_terms:
                    arch_terms.append(word)
            
            analysis["architectural_terms"] = arch_terms
            analysis["architectural_density"] = len(arch_terms) / len(words) if words else 0
            
            # Style indicators
            style_indicators = []
            for style in self.architectural_styles:
                if style.lower() in caption.lower():
                    style_indicators.append(style)
            analysis["style_indicators"] = style_indicators
            
            # Material indicators
            material_indicators = []
            for material in self.materials:
                if material.lower() in caption.lower():
                    material_indicators.append(material)
            analysis["material_indicators"] = material_indicators
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            analysis["error"] = str(e)
            
        return analysis
    
    def _perform_semiotic_analysis(self, textual: Dict, visual: Dict, spatial: Dict, segmentation: Dict) -> Dict[str, Any]:
        """Perform comprehensive semiotic analysis."""
        
        analysis = {}
        
        try:
            # Complexity assessment
            complexity_score = 0.0
            if "edge_density" in spatial:
                complexity_score += spatial["edge_density"]
            if "color_palette" in visual and "color_diversity" in visual["color_palette"]:
                complexity_score += visual["color_palette"]["color_diversity"] / 10
            if "segmentation_complexity" in segmentation:
                complexity_score += segmentation["segmentation_complexity"] / 10
            
            analysis["complexity_score"] = complexity_score
            
            # Segmentation-based analysis
            if segmentation.get("architectural_types"):
                analysis["architectural_types_detected"] = segmentation["architectural_types"]
            if segmentation.get("total_segments"):
                analysis["segmentation_density"] = segmentation["total_segments"]
            if segmentation.get("architectural_segments"):
                analysis["architectural_segment_ratio"] = (
                    segmentation["architectural_segments"] / max(segmentation["total_segments"], 1)
                )
            
            # Style analysis from text
            if textual.get("semantic_analysis") and "style_indicators" in textual["semantic_analysis"]:
                analysis["detected_styles"] = textual["semantic_analysis"]["style_indicators"]
            
            # Material analysis from text
            if textual.get("semantic_analysis") and "material_indicators" in textual["semantic_analysis"]:
                analysis["detected_materials"] = textual["semantic_analysis"]["material_indicators"]
            
            # Extract architectural style and mood for downstream compatibility
            if textual.get("caption"):
                architectural_style = self._extract_architectural_style(textual["caption"])
                if architectural_style:
                    analysis["architectural_style"] = architectural_style
                
                urban_mood = self._extract_urban_mood(textual["caption"])
                if urban_mood:
                    analysis["urban_mood"] = urban_mood
                    
                # Add other semiotic features for downstream compatibility
                analysis["semiotic_score"] = complexity_score
                analysis["architectural_typology"] = "building"  # Default value
                analysis["time_period"] = "daytime"  # Default value
                analysis["materials"] = analysis.get("detected_materials", ["concrete", "glass"])
                analysis["cultural_context"] = "urban"  # Default value
                analysis["symbolic_meaning"] = "architectural expression"  # Default value
            
            # Cultural context (simplified)
            if textual.get("caption"):
                analysis["cultural_indicators"] = self._extract_cultural_indicators(textual["caption"])
            
            # Visual-textual consistency
            if textual.get("caption"):
                analysis["description_length"] = len(textual["caption"].split())
                analysis["descriptive_richness"] = textual.get("semantic_analysis", {}).get("architectural_density", 0.0)
            
            # Color analysis
            if "color_palette" in visual and "palette" in visual["color_palette"]:
                dominant_colors = visual["color_palette"]["palette"][:3]  # Top 3 colors
                analysis["dominant_colors"] = [color["hex"] for color in dominant_colors]
                
                # Simple color temperature analysis
                avg_color = np.mean([color["color"] for color in dominant_colors], axis=0)
                if avg_color[0] > avg_color[2]:  # More red than blue
                    analysis["color_temperature"] = "warm"
                else:
                    analysis["color_temperature"] = "cool"
            
        except Exception as e:
            logger.error(f"Error in semiotic analysis: {e}")
            analysis["error"] = str(e)
            
        return analysis
    
    def _extract_cultural_indicators(self, caption: str) -> List[str]:
        """Extract cultural indicators from caption."""
        
        indicators = []
        caption_lower = caption.lower()
        
        for indicator in self.cultural_indicators:
            if indicator in caption_lower:
                indicators.append(indicator)
                
        return indicators
    
    def _extract_architectural_style(self, caption: str) -> str:
        """Extract architectural style from caption text for downstream compatibility."""
        style_keywords = {
            "modern": ["modern", "contemporary", "sleek", "minimalist", "clean lines"],
            "brutalist": ["brutalist", "concrete", "massive", "monumental", "raw"],
            "classical": ["classical", "neoclassical", "columns", "symmetrical"],
            "gothic": ["gothic", "pointed arches", "flying buttresses", "cathedral"],
            "baroque": ["baroque", "ornate", "decorative", "elaborate"],
            "art_deco": ["art deco", "geometric", "streamlined", "stylized"],
            "industrial": ["industrial", "factory", "warehouse", "steel", "brick"],
            "postmodern": ["postmodern", "eclectic", "playful", "mixed styles"],
            "traditional": ["traditional", "vernacular", "local style", "heritage"],
            "futuristic": ["futuristic", "sci-fi", "avant-garde", "experimental"]
        }
        
        caption_lower = caption.lower()
        detected_styles = []
        
        for style, keywords in style_keywords.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    detected_styles.append(style)
                    break
        
        # Return the most specific style found, or the first one
        return detected_styles[0] if detected_styles else "contemporary"
    
    def _extract_urban_mood(self, caption: str) -> str:
        """Extract urban mood from caption text for downstream compatibility."""
        mood_keywords = {
            "vibrant": ["vibrant", "lively", "energetic", "bustling", "dynamic"],
            "serene": ["serene", "peaceful", "calm", "quiet", "tranquil"],
            "dystopian": ["dystopian", "dark", "gritty", "bleak", "oppressive"],
            "nostalgic": ["nostalgic", "vintage", "retro", "historical", "timeless"],
            "futuristic": ["futuristic", "high-tech", "advanced", "cutting-edge"],
            "melancholic": ["melancholic", "somber", "moody", "atmospheric"],
            "dramatic": ["dramatic", "striking", "bold", "imposing", "monumental"],
            "intimate": ["intimate", "cozy", "human-scale", "neighborhood"],
            "industrial": ["industrial", "urban", "metropolitan", "city"],
            "organic": ["organic", "natural", "flowing", "curved", "biomorphic"]
        }
        
        caption_lower = caption.lower()
        detected_moods = []
        
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    detected_moods.append(mood)
                    break
        
        # Return the most specific mood found, or a default
        return detected_moods[0] if detected_moods else "urban"
    
    def _generate_summary_stats(self, results: List[Dict], output_path: Path):
        """Generate summary statistics for the processed dataset."""
        
        try:
            stats = {
                "total_images": len(results),
                "splits": {},
                "style_distribution": {},
                "material_distribution": {},
                "complexity_stats": [],
                "color_temperature_distribution": {}
            }
            
            # Collect statistics
            for result in results:
                # Split distribution
                split = result.get("split", "unknown")
                stats["splits"][split] = stats["splits"].get(split, 0) + 1
                
                # Style distribution
                if "semiotic_analysis" in result and "detected_styles" in result["semiotic_analysis"]:
                    for style in result["semiotic_analysis"]["detected_styles"]:
                        stats["style_distribution"][style] = stats["style_distribution"].get(style, 0) + 1
                
                # Material distribution
                if "semiotic_analysis" in result and "detected_materials" in result["semiotic_analysis"]:
                    for material in result["semiotic_analysis"]["detected_materials"]:
                        stats["material_distribution"][material] = stats["material_distribution"].get(material, 0) + 1
                
                # Complexity scores
                if "semiotic_analysis" in result and "complexity_score" in result["semiotic_analysis"]:
                    stats["complexity_stats"].append(result["semiotic_analysis"]["complexity_score"])
                
                # Color temperature
                if "semiotic_analysis" in result and "color_temperature" in result["semiotic_analysis"]:
                    temp = result["semiotic_analysis"]["color_temperature"]
                    stats["color_temperature_distribution"][temp] = stats["color_temperature_distribution"].get(temp, 0) + 1
            
            # Calculate complexity statistics
            if stats["complexity_stats"]:
                stats["complexity_summary"] = {
                    "mean": float(np.mean(stats["complexity_stats"])),
                    "std": float(np.std(stats["complexity_stats"])),
                    "min": float(np.min(stats["complexity_stats"])),
                    "max": float(np.max(stats["complexity_stats"]))
                }
            
            # Save statistics
            stats_file = output_path / "dataset_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✓ Summary statistics saved to {stats_file}")
            
            # Print summary
            logger.info(f"Dataset Summary:")
            logger.info(f"  Total images: {stats['total_images']}")
            logger.info(f"  Splits: {stats['splits']}")
            if stats["complexity_stats"]:
                logger.info(f"  Complexity: mean={stats['complexity_summary']['mean']:.3f}, std={stats['complexity_summary']['std']:.3f}")
            logger.info(f"  Top styles: {dict(list(sorted(stats['style_distribution'].items(), key=lambda x: x[1], reverse=True))[:5])}")
            logger.info(f"  Top materials: {dict(list(sorted(stats['material_distribution'].items(), key=lambda x: x[1], reverse=True))[:5])}")
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")


def main():
    """Main execution function."""
    
    # Configuration
    data_dir = "data"
    output_dir = "data/outputs/04_semiotic_features"
    
    # Initialize extractor
    logger.info("Initializing Simple Semiotic Extractor...")
    extractor = SimpleSemioticExtractor()
    
    # Process images
    logger.info("Starting image processing...")
    extractor.process_images_direct(data_dir, output_dir)
    
    logger.info("✓ Semiotic feature extraction complete!")


if __name__ == "__main__":
    main()