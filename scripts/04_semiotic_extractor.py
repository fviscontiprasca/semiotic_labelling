"""
Semiotic feature extraction system that combines BLIP-2 captions, SAM segmentation,
and architectural analysis to create rich semantic embeddings for Flux.1d training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import fiftyone as fo
import cv2
from PIL import Image
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemioticFeatures:
    """Container for comprehensive semiotic feature representation."""
    
    # Textual features
    caption_embedding: np.ndarray
    architectural_style: Optional[str] = None
    urban_mood: Optional[str] = None
    time_period: Optional[str] = None
    season: Optional[str] = None
    materials: List[str] = None
    
    # Visual features
    clip_embedding: np.ndarray = None
    color_palette: List[Tuple[int, int, int]] = None
    composition_features: Dict[str, float] = None
    
    # Spatial features
    object_density: float = 0.0
    spatial_hierarchy: str = None
    dominant_elements: List[str] = None
    
    # Semiotic interpretations
    cultural_context: Optional[str] = None
    social_implications: List[str] = None
    symbolic_meaning: Optional[str] = None
    architectural_typology: Optional[str] = None
    
    # Multi-modal fusion
    unified_embedding: np.ndarray = None
    semiotic_score: float = 0.0

class SemioticFeatureExtractor:
    """Advanced semiotic feature extraction for architectural images."""
    
    def __init__(self, device: str = "auto"):
        """Initialize feature extraction models and vocabularies."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing semiotic feature extractor on {self.device}")
        
        # Load multi-modal models
        self._load_models()
        
        # Initialize semiotic vocabularies
        self._init_semiotic_vocabularies()
        
        # Feature weights for fusion
        self.feature_weights = {
            "textual": 0.4,
            "visual": 0.3,
            "spatial": 0.2,
            "semiotic": 0.1
        }
    
    def _load_models(self):
        """Load required models for feature extraction."""
        
        # CLIP for visual-textual alignment
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        # Sentence transformer for text embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # TF-IDF for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        logger.info("Models loaded successfully")
    
    def _init_semiotic_vocabularies(self):
        """Initialize comprehensive semiotic vocabularies."""
        
        self.semiotic_vocab = {
            "architectural_styles": {
                "modernist": ["clean lines", "minimal", "geometric", "functional", "glass", "steel"],
                "brutalist": ["concrete", "massive", "angular", "fortress-like", "monumental"],
                "postmodern": ["eclectic", "playful", "colorful", "decorative", "historical references"],
                "baroque": ["ornate", "dramatic", "curved", "decorative", "gilded"],
                "minimalist": ["simple", "pure", "essential", "white", "sparse"],
                "industrial": ["exposed", "metal", "brick", "utilitarian", "raw"],
                "gothic": ["pointed arches", "vertical", "ornate", "stone", "religious"],
                "art deco": ["streamlined", "geometric patterns", "luxury", "metallic", "stylized"]
            },
            
            "urban_moods": {
                "contemplative": ["peaceful", "quiet", "reflective", "serene", "meditative"],
                "vibrant": ["energetic", "colorful", "dynamic", "bustling", "lively"],
                "tense": ["dramatic", "conflict", "pressure", "stress", "anxiety"],
                "melancholic": ["nostalgic", "sad", "lonely", "abandoned", "decay"],
                "imposing": ["monumental", "powerful", "dominant", "overwhelming", "authoritative"],
                "intimate": ["cozy", "human-scale", "personal", "comfortable", "welcoming"]
            },
            
            "spatial_qualities": {
                "monumental": ["large scale", "impressive", "towering", "grand", "massive"],
                "intimate": ["small scale", "cozy", "personal", "human-sized", "approachable"],
                "hierarchical": ["ordered", "structured", "layered", "ranked", "organized"],
                "fragmented": ["broken", "disconnected", "scattered", "divided", "partial"],
                "flowing": ["continuous", "smooth", "organic", "curved", "seamless"],
                "rigid": ["structured", "angular", "geometric", "fixed", "systematic"]
            },
            
            "cultural_contexts": {
                "european": ["classical", "historical", "stone", "tradition", "heritage"],
                "american": ["modern", "commercial", "glass", "innovation", "efficiency"],
                "asian": ["harmony", "balance", "nature", "wood", "spirituality"],
                "mediterranean": ["warm", "earthy", "courtyard", "outdoor living", "community"],
                "nordic": ["minimal", "natural", "wood", "light", "sustainability"]
            }
        }
        
        # Compile patterns for text analysis
        self.style_patterns = {}
        for style, keywords in self.semiotic_vocab["architectural_styles"].items():
            pattern = "|".join([re.escape(kw) for kw in keywords])
            self.style_patterns[style] = re.compile(pattern, re.IGNORECASE)
    
    def extract_features(self, sample: fo.Sample) -> SemioticFeatures:
        """Extract comprehensive semiotic features from a FiftyOne sample."""
        
        try:
            # Load image
            image = Image.open(sample.filepath).convert("RGB")
            
            # Extract textual features
            textual_features = self._extract_textual_features(sample)
            
            # Extract visual features
            visual_features = self._extract_visual_features(image)
            
            # Extract spatial features
            spatial_features = self._extract_spatial_features(sample)
            
            # Perform semiotic analysis
            semiotic_analysis = self._perform_semiotic_analysis(
                textual_features, visual_features, spatial_features
            )
            
            # Create unified embedding
            unified_embedding = self._create_unified_embedding(
                textual_features, visual_features, spatial_features, semiotic_analysis
            )
            
            # Create semiotic features object
            features = SemioticFeatures(
                caption_embedding=textual_features["caption_embedding"],
                architectural_style=textual_features.get("architectural_style"),
                urban_mood=textual_features.get("urban_mood"),
                time_period=textual_features.get("time_period"),
                season=textual_features.get("season"),
                materials=textual_features.get("materials", []),
                
                clip_embedding=visual_features["clip_embedding"],
                color_palette=visual_features.get("color_palette", []),
                composition_features=visual_features.get("composition_features", {}),
                
                object_density=spatial_features.get("density", 0.0),
                spatial_hierarchy=spatial_features.get("hierarchy"),
                dominant_elements=spatial_features.get("dominant_elements", []),
                
                cultural_context=semiotic_analysis.get("cultural_context"),
                social_implications=semiotic_analysis.get("social_implications", []),
                symbolic_meaning=semiotic_analysis.get("symbolic_meaning"),
                architectural_typology=semiotic_analysis.get("typology"),
                
                unified_embedding=unified_embedding,
                semiotic_score=semiotic_analysis.get("semiotic_score", 0.0)
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {sample.filepath}: {e}")
            return self._create_empty_features()
    
    def _extract_textual_features(self, sample: fo.Sample) -> Dict[str, Any]:
        """Extract features from textual descriptions and captions."""
        
        features = {}
        
        # Gather all available text
        text_sources = []
        
        if hasattr(sample, "original_prompt") and sample.original_prompt:
            text_sources.append(sample.original_prompt)
        
        if hasattr(sample, "semiotic_captions") and sample.semiotic_captions:
            captions = sample.semiotic_captions
            if isinstance(captions, dict):
                for caption in captions.values():
                    if isinstance(caption, str):
                        text_sources.append(caption)
        
        # Combine all text
        combined_text = " ".join(text_sources) if text_sources else ""
        
        if not combined_text:
            return self._create_empty_textual_features()
        
        # Generate sentence embedding
        features["caption_embedding"] = self.sentence_model.encode(combined_text)
        
        # Extract architectural style
        features["architectural_style"] = self._extract_architectural_style(combined_text)
        
        # Extract mood
        features["urban_mood"] = self._extract_urban_mood(combined_text)
        
        # Extract temporal features
        features["time_period"] = self._extract_time_period(combined_text)
        features["season"] = self._extract_season(combined_text)
        
        # Extract materials
        features["materials"] = self._extract_materials(combined_text)
        
        return features
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features using CLIP and image analysis."""
        
        features = {}
        
        # CLIP embedding
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            features["clip_embedding"] = image_features.cpu().numpy().flatten()
        
        # Color analysis
        features["color_palette"] = self._extract_color_palette(image)
        
        # Composition analysis
        features["composition_features"] = self._analyze_composition(image)
        
        return features
    
    def _extract_spatial_features(self, sample: fo.Sample) -> Dict[str, Any]:
        """Extract spatial features from segmentation and detection data."""
        
        features = {}
        
        # Extract from SAM segmentation if available
        if hasattr(sample, "semiotic_segmentation_analysis"):
            seg_analysis = sample.semiotic_segmentation_analysis
            
            if isinstance(seg_analysis, dict):
                features["density"] = seg_analysis.get("density_analysis", {}).get("total_coverage", 0.0)
                features["hierarchy"] = seg_analysis.get("architectural_hierarchy", {}).get("hierarchy_type")
                features["dominant_elements"] = seg_analysis.get("architectural_hierarchy", {}).get("dominant_elements", [])
                features["functional_composition"] = seg_analysis.get("functional_composition", {})
        
        return features
    
    def _perform_semiotic_analysis(self, textual: Dict, visual: Dict, spatial: Dict) -> Dict[str, Any]:
        """Perform high-level semiotic analysis combining all features."""
        
        analysis = {}
        
        # Cultural context analysis
        analysis["cultural_context"] = self._analyze_cultural_context(textual)
        
        # Social implications
        analysis["social_implications"] = self._analyze_social_implications(textual, spatial)
        
        # Symbolic meaning
        analysis["symbolic_meaning"] = self._analyze_symbolic_meaning(textual, visual)
        
        # Architectural typology
        analysis["typology"] = self._determine_architectural_typology(textual, spatial)
        
        # Calculate semiotic richness score
        analysis["semiotic_score"] = self._calculate_semiotic_score(textual, visual, spatial)
        
        return analysis
    
    def _create_unified_embedding(self, textual: Dict, visual: Dict, 
                                spatial: Dict, semiotic: Dict) -> np.ndarray:
        """Create unified multi-modal embedding."""
        
        # Normalize and combine embeddings
        embeddings = []
        
        # Textual embedding
        if "caption_embedding" in textual:
            text_emb = textual["caption_embedding"]
            text_emb = text_emb / np.linalg.norm(text_emb)
            embeddings.append(text_emb * self.feature_weights["textual"])
        
        # Visual embedding
        if "clip_embedding" in visual:
            vis_emb = visual["clip_embedding"]
            vis_emb = vis_emb / np.linalg.norm(vis_emb)
            embeddings.append(vis_emb * self.feature_weights["visual"])
        
        # Spatial features as embedding
        spatial_vec = self._vectorize_spatial_features(spatial)
        if spatial_vec is not None:
            spatial_vec = spatial_vec / np.linalg.norm(spatial_vec)
            embeddings.append(spatial_vec * self.feature_weights["spatial"])
        
        # Semiotic features as embedding
        semiotic_vec = self._vectorize_semiotic_features(semiotic)
        if semiotic_vec is not None:
            semiotic_vec = semiotic_vec / np.linalg.norm(semiotic_vec)
            embeddings.append(semiotic_vec * self.feature_weights["semiotic"])
        
        # Concatenate or average embeddings
        if embeddings:
            # Pad to same length if needed
            max_len = max(emb.shape[0] for emb in embeddings)
            padded_embeddings = []
            for emb in embeddings:
                if emb.shape[0] < max_len:
                    padded = np.pad(emb, (0, max_len - emb.shape[0]), 'constant')
                    padded_embeddings.append(padded)
                else:
                    padded_embeddings.append(emb[:max_len])
            
            unified = np.mean(padded_embeddings, axis=0)
            return unified / np.linalg.norm(unified)
        
        return np.zeros(512)  # Default embedding size
    
    def _extract_architectural_style(self, text: str) -> Optional[str]:
        """Extract architectural style from text."""
        text_lower = text.lower()
        
        for style, pattern in self.style_patterns.items():
            if pattern.search(text_lower):
                return style
        
        return None
    
    def _extract_urban_mood(self, text: str) -> Optional[str]:
        """Extract urban mood from text."""
        text_lower = text.lower()
        
        for mood, keywords in self.semiotic_vocab["urban_moods"].items():
            if any(keyword in text_lower for keyword in keywords):
                return mood
        
        return None
    
    def _extract_time_period(self, text: str) -> Optional[str]:
        """Extract time period from text."""
        time_patterns = {
            "dawn": r"\b(dawn|sunrise|early morning)\b",
            "morning": r"\b(morning)\b",
            "afternoon": r"\b(afternoon|midday)\b",
            "evening": r"\b(evening|dusk|sunset)\b",
            "night": r"\b(night|nighttime)\b"
        }
        
        text_lower = text.lower()
        for period, pattern in time_patterns.items():
            if re.search(pattern, text_lower):
                return period
        
        return None
    
    def _extract_season(self, text: str) -> Optional[str]:
        """Extract season from text."""
        seasons = ["spring", "summer", "autumn", "fall", "winter"]
        
        text_lower = text.lower()
        for season in seasons:
            if season in text_lower:
                return season
        
        return None
    
    def _extract_materials(self, text: str) -> List[str]:
        """Extract mentioned materials from text."""
        materials = [
            "concrete", "glass", "steel", "stone", "brick", "wood",
            "limestone", "marble", "metal", "stucco", "granite"
        ]
        
        found_materials = []
        text_lower = text.lower()
        
        for material in materials:
            if material in text_lower:
                found_materials.append(material)
        
        return found_materials
    
    def _extract_color_palette(self, image: Image.Image) -> List[Tuple[int, int, int]]:
        """Extract dominant color palette from image."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Simple color quantization using k-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]
    
    def _analyze_composition(self, image: Image.Image) -> Dict[str, float]:
        """Analyze image composition features."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Calculate basic composition metrics
        features = {}
        
        # Contrast
        features["contrast"] = gray.std()
        
        # Brightness
        features["brightness"] = gray.mean()
        
        # Edge density (complexity)
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Symmetry (simplified)
        left_half = gray[:, :gray.shape[1]//2]
        right_half = gray[:, gray.shape[1]//2:]
        right_half_flipped = np.fliplr(right_half)
        
        if left_half.shape == right_half_flipped.shape:
            features["horizontal_symmetry"] = np.corrcoef(
                left_half.flatten(), right_half_flipped.flatten()
            )[0, 1]
        else:
            features["horizontal_symmetry"] = 0.0
        
        return features
    
    def _analyze_cultural_context(self, textual: Dict) -> Optional[str]:
        """Analyze cultural context from textual features."""
        # Simple rule-based analysis
        style = textual.get("architectural_style")
        
        if style in ["baroque", "gothic", "neoclassical"]:
            return "european"
        elif style in ["modernist", "art deco"]:
            return "american"
        elif style == "minimalist":
            return "nordic"
        
        return None
    
    def _analyze_social_implications(self, textual: Dict, spatial: Dict) -> List[str]:
        """Analyze social implications from features."""
        implications = []
        
        # From architectural style
        style = textual.get("architectural_style")
        if style == "brutalist":
            implications.append("institutional_power")
        elif style == "minimalist":
            implications.append("exclusivity")
        elif style == "vernacular":
            implications.append("community_oriented")
        
        # From spatial features
        density = spatial.get("density", 0)
        if density > 0.6:
            implications.append("urban_intensity")
        elif density < 0.2:
            implications.append("spacious_living")
        
        return implications
    
    def _analyze_symbolic_meaning(self, textual: Dict, visual: Dict) -> Optional[str]:
        """Analyze symbolic meaning from combined features."""
        style = textual.get("architectural_style")
        mood = textual.get("urban_mood")
        
        if style == "gothic" and mood == "contemplative":
            return "spiritual_transcendence"
        elif style == "brutalist" and mood == "imposing":
            return "institutional_authority"
        elif style == "modernist" and mood == "vibrant":
            return "progressive_optimism"
        
        return None
    
    def _determine_architectural_typology(self, textual: Dict, spatial: Dict) -> Optional[str]:
        """Determine architectural typology from features."""
        dominant_elements = spatial.get("dominant_elements", [])
        
        if "skyscraper" in dominant_elements:
            return "high_rise_commercial"
        elif "house" in dominant_elements:
            return "residential"
        elif "building" in dominant_elements:
            return "mixed_use"
        
        return None
    
    def _calculate_semiotic_score(self, textual: Dict, visual: Dict, spatial: Dict) -> float:
        """Calculate overall semiotic richness score."""
        score = 0.0
        
        # Points for textual richness
        if textual.get("architectural_style"):
            score += 0.2
        if textual.get("urban_mood"):
            score += 0.2
        if textual.get("materials"):
            score += 0.1 * len(textual["materials"])
        
        # Points for visual complexity
        if visual.get("composition_features", {}).get("edge_density", 0) > 0.1:
            score += 0.2
        
        # Points for spatial complexity
        if spatial.get("density", 0) > 0:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _vectorize_spatial_features(self, spatial: Dict) -> Optional[np.ndarray]:
        """Convert spatial features to vector representation."""
        features = []
        
        features.append(spatial.get("density", 0.0))
        
        # Encode hierarchy as one-hot
        hierarchy_types = ["strong_hierarchy", "moderate_hierarchy", "uniform_scale"]
        hierarchy = spatial.get("hierarchy", "uniform_scale")
        hierarchy_vec = [1.0 if h == hierarchy else 0.0 for h in hierarchy_types]
        features.extend(hierarchy_vec)
        
        return np.array(features) if features else None
    
    def _vectorize_semiotic_features(self, semiotic: Dict) -> Optional[np.ndarray]:
        """Convert semiotic features to vector representation."""
        features = []
        
        features.append(semiotic.get("semiotic_score", 0.0))
        
        # Encode cultural context as one-hot
        contexts = ["european", "american", "asian", "mediterranean", "nordic"]
        context = semiotic.get("cultural_context")
        context_vec = [1.0 if c == context else 0.0 for c in contexts]
        features.extend(context_vec)
        
        return np.array(features) if features else None
    
    def _create_empty_features(self) -> SemioticFeatures:
        """Create empty feature object for error cases."""
        return SemioticFeatures(
            caption_embedding=np.zeros(384),  # sentence-transformers default
            clip_embedding=np.zeros(512),     # CLIP default
            unified_embedding=np.zeros(512)
        )
    
    def _create_empty_textual_features(self) -> Dict[str, Any]:
        """Create empty textual features."""
        return {
            "caption_embedding": np.zeros(384),
            "architectural_style": None,
            "urban_mood": None,
            "time_period": None,
            "season": None,
            "materials": []
        }
    
    def process_dataset(self, dataset: fo.Dataset, 
                       features_field: str = "semiotic_features") -> None:
        """Process entire dataset to extract semiotic features."""
        
        logger.info(f"Processing {len(dataset)} samples for semiotic feature extraction")
        
        processed = 0
        for sample in dataset.iter_samples(progress=True):
            try:
                # Extract features
                features = self.extract_features(sample)
                
                # Convert to serializable format
                features_dict = asdict(features)
                
                # Convert numpy arrays to lists for JSON serialization
                for key, value in features_dict.items():
                    if isinstance(value, np.ndarray):
                        features_dict[key] = value.tolist()
                
                # Add to sample
                sample[features_field] = features_dict
                sample.save()
                
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing {sample.filepath}: {e}")
                continue
        
        logger.info(f"Completed feature extraction for {processed} samples")
    
    def _extract_features_from_data(self, image_path: Path, seg_analysis: Dict, captions: Optional[Dict]) -> SemioticFeatures:
        """Extract features directly from image path and analysis data."""
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Extract textual features
            text_sources = []
            if captions:
                if isinstance(captions, dict):
                    for caption_type, caption_text in captions.items():
                        if caption_text:
                            text_sources.append(caption_text)
                elif isinstance(captions, str):
                    text_sources.append(captions)
            
            combined_text = " ".join(text_sources) if text_sources else "architectural building urban scene"
            
            # Generate text embedding
            caption_embedding = self.sentence_model.encode(combined_text)
            
            # Extract visual features
            visual_features = self._extract_visual_features(image)
            
            # Extract spatial features from segmentation analysis
            spatial_features = {}
            if seg_analysis:
                spatial_features["density"] = seg_analysis.get("density_analysis", {}).get("total_coverage", 0.0)
                spatial_features["hierarchy"] = seg_analysis.get("architectural_hierarchy", {}).get("hierarchy_type")
                spatial_features["dominant_elements"] = seg_analysis.get("architectural_hierarchy", {}).get("dominant_elements", [])
                spatial_features["functional_composition"] = seg_analysis.get("functional_composition", {})
            
            # Perform semiotic analysis
            textual_dict = {"caption_embedding": caption_embedding}
            semiotic_analysis = self._perform_semiotic_analysis(textual_dict, visual_features, spatial_features)
            
            # Create SemioticFeatures object
            features = SemioticFeatures(
                caption_embedding=caption_embedding if isinstance(caption_embedding, np.ndarray) else np.array(caption_embedding),
                clip_embedding=np.array(visual_features.get("clip_embedding")) if visual_features.get("clip_embedding") is not None else np.zeros(512),
                color_palette=visual_features.get("color_palette", []),
                composition_features=visual_features.get("composition_features", {}),
                object_density=float(spatial_features.get("density", 0.0)),
                spatial_hierarchy=spatial_features.get("hierarchy") or "uniform_scale",
                dominant_elements=spatial_features.get("dominant_elements", []),
                cultural_context=semiotic_analysis.get("cultural_context"),
                architectural_style=semiotic_analysis.get("typology"),
                materials=semiotic_analysis.get("materials", [])
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            # Return minimal features object
            return SemioticFeatures(
                caption_embedding=np.zeros(384),  # Default sentence transformer size
                object_density=0.0,
                dominant_elements=[]
            )

def process_sam_outputs(input_dir: str, output_dir: str):
    """Process SAM segmentation outputs directly without FiftyOne."""
    
    import argparse
    from pathlib import Path
    import json
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    extractor = SemioticFeatureExtractor()
    
    # Load SAM segmentation summary
    segmentation_summary_file = input_path / "segmentation_summary.json"
    if not segmentation_summary_file.exists():
        logger.error(f"Segmentation summary not found: {segmentation_summary_file}")
        return
    
    with open(segmentation_summary_file, 'r') as f:
        segmentation_data = json.load(f)
    
    # Load BLIP-2 captions if available
    blip_captions = {}
    blip_dir = input_path.parent / "02_blip2_captioner"
    if blip_dir.exists():
        blip_files = list(blip_dir.glob("*.json"))
        for blip_file in blip_files:
            with open(blip_file, 'r') as f:
                blip_data = json.load(f)
                blip_captions.update(blip_data)
    
    logger.info(f"Processing {len(segmentation_data)} images with semiotic analysis")
    
    results = {}
    processed = 0
    
    for image_key, seg_analysis in segmentation_data.items():
        try:
            # Get image path
            if "/" in image_key:
                split, image_name = image_key.split("/", 1)
                image_path = input_path.parent / "01_data_pipeline" / "images" / split / image_name
            else:
                image_path = input_path.parent / "01_data_pipeline" / "images" / "train" / image_key
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Extract features directly from data
            features = extractor._extract_features_from_data(
                image_path=image_path,
                seg_analysis=seg_analysis.get("semiotic_segmentation_analysis", {}),
                captions=blip_captions.get(image_path.name, None)
            )
            
            # Convert to serializable format
            features_dict = asdict(features)
            for key, value in features_dict.items():
                if isinstance(value, np.ndarray):
                    features_dict[key] = value.tolist()
            
            results[image_key] = {
                "image_path": str(image_path),
                "semiotic_features": features_dict,
                "segmentation_summary": seg_analysis,
                "timestamp": seg_analysis.get("timestamp")
            }
            
            processed += 1
            if processed % 5 == 0:
                logger.info(f"Processed {processed}/{len(segmentation_data)} images")
                
        except Exception as e:
            logger.error(f"Error processing {image_key}: {e}")
            continue
    
    # Save results
    output_file = output_path / "semiotic_features.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary = {
        "timestamp": results[list(results.keys())[0]]["timestamp"] if results else None,
        "total_images": len(segmentation_data),
        "processed_images": processed,
        "failed_images": len(segmentation_data) - processed,
        "output_file": str(output_file),
        "feature_types": ["caption_embedding", "clip_embedding", "color_palette", "composition_features", "object_density", "spatial_hierarchy", "dominant_elements", "cultural_context", "architectural_style", "materials"] if processed > 0 else []
    }
    
    summary_file = output_path / "semiotic_features_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Semiotic feature extraction complete!")
    logger.info(f"Processed: {processed}/{len(segmentation_data)} images")
    logger.info(f"Features saved to: {output_file}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return results

def main():
    """Main execution with command-line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Semiotic Feature Extraction")
    parser.add_argument("--input_data", help="Input directory with SAM segmentation data")
    parser.add_argument("--output_features", help="Output directory for semiotic features")
    parser.add_argument("--fiftyone_mode", action="store_true", help="Use FiftyOne dataset mode (default)")
    
    args = parser.parse_args()
    
    if args.input_data and args.output_features:
        # Process SAM outputs directly
        process_sam_outputs(args.input_data, args.output_features)
        
    else:
        # Original FiftyOne mode
        # Initialize feature extractor
        extractor = SemioticFeatureExtractor()
        
        # Load dataset
        dataset_name = "semiotic_urban_combined"
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
            
            # Process subset for testing
            test_view = dataset.take(3)
            extractor.process_dataset(test_view)
            
            # Launch FiftyOne to view results
            session = fo.launch_app(dataset, port=5151)
            print("Semiotic features extracted. View results at http://localhost:5151")
            
            try:
                session.wait()
            except KeyboardInterrupt:
                print("Shutting down...")
        else:
            print(f"Dataset {dataset_name} not found. Run the full pipeline first.")

if __name__ == "__main__":
    main()