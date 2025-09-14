"""
Comprehensive evaluation pipeline for semiotic-aware architectural image generation.
Evaluates generated images across multiple dimensions including semiotic coherence,
architectural accuracy, and visual quality.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# ML libraries with error handling
try:
    import torch
    import clip
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML libraries not available: {e}")

# Import pipeline components with error handling
try:
    from blip2_captioner import SemioticBLIPCaptioner
    from yolo_segmenter import UrbanYOLOSegmenter
    from semiotic_extractor import SemioticFeatureExtractor
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"Pipeline components not available: {e}")

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for generated images."""
    
    # Semiotic coherence metrics
    style_accuracy: float = 0.0
    mood_accuracy: float = 0.0
    semiotic_consistency: float = 0.0
    
    # Architectural accuracy metrics
    object_detection_score: float = 0.0
    spatial_coherence: float = 0.0
    architectural_realism: float = 0.0
    
    # Visual quality metrics
    clip_score: float = 0.0
    image_quality_score: float = 0.0
    
    # Text-image alignment metrics
    caption_alignment: float = 0.0
    prompt_following: float = 0.0
    semantic_consistency: float = 0.0
    
    # Composite scores
    semiotic_score: float = 0.0
    overall_score: float = 0.0

class SemioticImageEvaluator:
    """Comprehensive evaluator for semiotic-aware generated images."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the evaluator with all necessary models."""
        
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available")
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize evaluation models
        self._load_evaluation_models()
        
        # Initialize pipeline components if available
        if PIPELINE_AVAILABLE:
            try:
                self.blip_captioner = SemioticBLIPCaptioner(device=self.device)
                self.yolo_segmenter = UrbanYOLOSegmenter()
                self.semiotic_extractor = SemioticFeatureExtractor(device=self.device)
                logger.info("Pipeline components loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load pipeline components: {e}")
                self.blip_captioner = None
                self.yolo_segmenter = None
                self.semiotic_extractor = None
        else:
            self.blip_captioner = None
            self.yolo_segmenter = None
            self.semiotic_extractor = None
        
        # Define evaluation criteria
        self._init_evaluation_criteria()
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def _load_evaluation_models(self):
        """Load models needed for evaluation."""
        
        # Load CLIP for text-image alignment
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.error(f"Could not load CLIP: {e}")
            self.clip_model = None
            self.clip_preprocess = None
        
        # Load sentence transformer for text similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def _init_evaluation_criteria(self):
        """Initialize evaluation criteria and keywords."""
        
        self.evaluation_criteria = {
            "architectural_styles": {
                "modernist": ["modern", "clean", "minimal", "geometric", "glass", "steel"],
                "brutalist": ["concrete", "massive", "geometric", "fortress", "raw", "monumental"],
                "postmodern": ["eclectic", "colorful", "playful", "historical", "mixed"],
                "minimalist": ["minimal", "simple", "clean", "white", "essential"],
                "baroque": ["ornate", "decorative", "curved", "dramatic", "rich"],
                "gothic": ["pointed", "arches", "vertical", "stone", "cathedral"],
                "industrial": ["exposed", "metal", "brick", "utilitarian", "warehouse"],
                "art_deco": ["streamlined", "geometric", "metallic", "stylized"]
            },
            
            "urban_moods": {
                "contemplative": ["peaceful", "quiet", "reflective", "calm", "serene"],
                "vibrant": ["energetic", "colorful", "lively", "bustling", "dynamic"],
                "tense": ["dramatic", "sharp", "conflicting", "pressure", "intense"],
                "serene": ["tranquil", "balanced", "harmonious", "peaceful"],
                "dramatic": ["bold", "striking", "powerful", "contrasting"],
                "peaceful": ["gentle", "soft", "natural", "restful"],
                "energetic": ["dynamic", "bright", "active", "movement"],
                "melancholic": ["muted", "weathered", "nostalgic", "sad"]
            }
        }
    
    def evaluate_single_image(self, image_path: str, original_prompt: str,
                            expected_style: str = None, expected_mood: str = None) -> EvaluationMetrics:
        """Evaluate a single generated image comprehensively."""
        
        logger.info(f"Evaluating image: {Path(image_path).name}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Generate captions and analysis if pipeline is available
            generated_captions = {}
            segmentation_result = None
            semiotic_features = None
            
            if self.blip_captioner:
                try:
                    generated_captions = self.blip_captioner.generate_semiotic_caption(image_path)
                except Exception as e:
                    logger.warning(f"Caption generation failed: {e}")
            
            if self.yolo_segmenter:
                try:
                    segmentation_result = self.yolo_segmenter.segment_image(image_path)
                except Exception as e:
                    logger.warning(f"Segmentation failed: {e}")
            
            if self.semiotic_extractor:
                try:
                    # Create a sample-like object for feature extraction
                    temp_sample = type('Sample', (), {
                        'filepath': image_path,
                        'original_prompt': original_prompt,
                        'semiotic_captions': generated_captions,
                        'semiotic_segmentation_analysis': segmentation_result.semiotic_analysis if segmentation_result else {}
                    })()
                    
                    semiotic_features = self.semiotic_extractor.extract_features(temp_sample)
                except Exception as e:
                    logger.warning(f"Feature extraction failed: {e}")
            
            # Calculate individual metrics
            metrics = EvaluationMetrics()
            
            # Semiotic coherence evaluation
            metrics.style_accuracy = self._evaluate_style_accuracy(
                generated_captions, original_prompt, expected_style
            )
            
            metrics.mood_accuracy = self._evaluate_mood_accuracy(
                generated_captions, original_prompt, expected_mood
            )
            
            metrics.semiotic_consistency = self._evaluate_semiotic_consistency(
                semiotic_features, generated_captions
            )
            
            # Architectural accuracy evaluation
            if segmentation_result:
                metrics.object_detection_score = self._evaluate_object_detection(
                    segmentation_result, original_prompt
                )
                
                metrics.spatial_coherence = self._evaluate_spatial_coherence(
                    segmentation_result.semiotic_analysis if hasattr(segmentation_result, 'semiotic_analysis') else {}
                )
                
                metrics.architectural_realism = self._evaluate_architectural_realism(
                    image, segmentation_result
                )
            
            # Visual quality evaluation
            metrics.clip_score = self._calculate_clip_score(image, original_prompt)
            metrics.image_quality_score = self._evaluate_image_quality(image)
            
            # Text-image alignment evaluation
            metrics.caption_alignment = self._evaluate_caption_alignment(
                generated_captions, original_prompt
            )
            
            metrics.prompt_following = self._evaluate_prompt_following(
                image, original_prompt, semiotic_features
            )
            
            metrics.semantic_consistency = self._evaluate_semantic_consistency(
                generated_captions, semiotic_features
            )
            
            # Calculate composite scores
            metrics.semiotic_score = self._calculate_semiotic_score(metrics)
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            logger.info(f"Evaluation completed. Overall score: {metrics.overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating image {image_path}: {e}")
            return EvaluationMetrics()  # Return empty metrics on error
    
    def _evaluate_style_accuracy(self, captions: Dict[str, str], 
                               original_prompt: str, 
                               expected_style: str = None) -> float:
        """Evaluate accuracy of architectural style detection."""
        
        # Extract style from generated captions
        detected_styles = set()
        
        for caption in captions.values():
            if isinstance(caption, str):
                caption_lower = caption.lower()
                for style, keywords in self.evaluation_criteria["architectural_styles"].items():
                    if any(keyword in caption_lower for keyword in keywords):
                        detected_styles.add(style)
        
        # Extract expected style from original prompt if not provided
        if expected_style is None:
            prompt_lower = original_prompt.lower()
            for style, keywords in self.evaluation_criteria["architectural_styles"].items():
                if style in prompt_lower or any(keyword in prompt_lower for keyword in keywords):
                    expected_style = style
                    break
        
        # Calculate accuracy
        if expected_style and expected_style in detected_styles:
            return 1.0
        elif expected_style:
            return 0.0
        else:
            # If no expected style, give partial credit for detecting any style
            return 0.5 if detected_styles else 0.0
    
    def _evaluate_mood_accuracy(self, captions: Dict[str, str], 
                              original_prompt: str, 
                              expected_mood: str = None) -> float:
        """Evaluate accuracy of urban mood detection."""
        
        # Similar logic to style accuracy
        detected_moods = set()
        
        for caption in captions.values():
            if isinstance(caption, str):
                caption_lower = caption.lower()
                for mood, keywords in self.evaluation_criteria["urban_moods"].items():
                    if mood in caption_lower or any(keyword in caption_lower for keyword in keywords):
                        detected_moods.add(mood)
        
        if expected_mood is None:
            prompt_lower = original_prompt.lower()
            for mood, keywords in self.evaluation_criteria["urban_moods"].items():
                if mood in prompt_lower or any(keyword in prompt_lower for keyword in keywords):
                    expected_mood = mood
                    break
        
        if expected_mood and expected_mood in detected_moods:
            return 1.0
        elif expected_mood:
            return 0.0
        else:
            return 0.5 if detected_moods else 0.0
    
    def _evaluate_semiotic_consistency(self, semiotic_features, captions: Dict[str, str]) -> float:
        """Evaluate consistency between extracted features and generated captions."""
        
        if not semiotic_features or not captions:
            return 0.0
        
        consistency_score = 0.0
        num_checks = 0
        
        # Check architectural style consistency
        if hasattr(semiotic_features, 'architectural_style') and semiotic_features.architectural_style:
            style = semiotic_features.architectural_style
            style_mentioned = any(
                style in caption.lower() for caption in captions.values() if isinstance(caption, str)
            )
            consistency_score += 1.0 if style_mentioned else 0.0
            num_checks += 1
        
        # Check mood consistency  
        if hasattr(semiotic_features, 'urban_mood') and semiotic_features.urban_mood:
            mood = semiotic_features.urban_mood
            mood_mentioned = any(
                mood in caption.lower() for caption in captions.values() if isinstance(caption, str)
            )
            consistency_score += 1.0 if mood_mentioned else 0.0
            num_checks += 1
        
        return consistency_score / num_checks if num_checks > 0 else 0.0
    
    def _evaluate_object_detection(self, segmentation_result, original_prompt: str) -> float:
        """Evaluate accuracy of architectural object detection."""
        
        if not hasattr(segmentation_result, 'classes') or not segmentation_result.classes:
            return 0.0
        
        # Count relevant architectural objects
        architectural_objects = [
            "building", "house", "skyscraper", "tower", "castle", 
            "office building", "convenience store", "lighthouse"
        ]
        
        detected_architectural = sum(
            1 for cls in segmentation_result.classes if cls in architectural_objects
        )
        
        # Score based on detection rate and confidence
        if detected_architectural > 0:
            if hasattr(segmentation_result, 'confidences') and segmentation_result.confidences:
                avg_confidence = np.mean([
                    conf for cls, conf in zip(segmentation_result.classes, segmentation_result.confidences)
                    if cls in architectural_objects
                ])
                return avg_confidence
            else:
                return 0.5  # Default score if no confidence scores
        
        return 0.0
    
    def _evaluate_spatial_coherence(self, semiotic_analysis: Dict[str, Any]) -> float:
        """Evaluate spatial coherence of the architectural scene."""
        
        if not semiotic_analysis:
            return 0.0
        
        score = 0.0
        
        # Check density appropriateness
        density_analysis = semiotic_analysis.get("density_analysis", {})
        if density_analysis.get("density_type") in ["medium_density", "high_density"]:
            score += 0.3
        
        # Check architectural hierarchy
        hierarchy = semiotic_analysis.get("architectural_hierarchy", {})
        if hierarchy.get("hierarchy_type") in ["strong_hierarchy", "moderate_hierarchy"]:
            score += 0.4
        
        # Check functional composition
        composition = semiotic_analysis.get("functional_composition", {})
        if composition.get("functional_diversity", 0) > 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_architectural_realism(self, image: Image.Image, segmentation_result) -> float:
        """Evaluate architectural realism of the generated image."""
        
        realism_score = 0.0
        
        # Convert image to numpy for analysis
        img_array = np.array(image)
        
        # Check image quality indicators
        # 1. Color distribution realism
        color_variance = np.var(img_array, axis=(0, 1))
        color_score = 1.0 if np.mean(color_variance) > 100 else 0.5
        realism_score += color_score * 0.3
        
        # 2. Edge coherence (architectural edges should be well-defined)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        edge_score = 1.0 if 0.05 < edge_density < 0.3 else 0.5
        realism_score += edge_score * 0.4
        
        # 3. Object detection confidence (higher confidence = more realistic)
        if hasattr(segmentation_result, 'confidences') and segmentation_result.confidences:
            avg_confidence = np.mean(segmentation_result.confidences)
            conf_score = avg_confidence
            realism_score += conf_score * 0.3
        
        return min(realism_score, 1.0)
    
    def _calculate_clip_score(self, image: Image.Image, text: str) -> float:
        """Calculate CLIP score for text-image alignment."""
        
        if not self.clip_model:
            return 0.0
        
        try:
            # Preprocess image and text
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([text]).to(self.device)
            
            # Calculate features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                clip_score = torch.cosine_similarity(image_features, text_features).item()
            
            return max(0.0, clip_score)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating CLIP score: {e}")
            return 0.0
    
    def _evaluate_image_quality(self, image: Image.Image) -> float:
        """Evaluate technical image quality."""
        
        img_array = np.array(image)
        
        # Check resolution
        height, width = img_array.shape[:2]
        resolution_score = 1.0 if min(height, width) >= 512 else 0.5
        
        # Check for artifacts (very basic)
        # Calculate image sharpness using Laplacian variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)  # Normalize
        
        # Check color saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        saturation_score = min(saturation / 255, 1.0)
        
        # Combine scores
        quality_score = (resolution_score * 0.4 + sharpness_score * 0.4 + saturation_score * 0.2)
        
        return quality_score
    
    def _evaluate_caption_alignment(self, generated_captions: Dict[str, str], 
                                  original_prompt: str) -> float:
        """Evaluate alignment between generated captions and original prompt."""
        
        if not generated_captions or not self.sentence_model:
            return 0.0
        
        # Use sentence transformer to calculate semantic similarity
        original_embedding = self.sentence_model.encode([original_prompt])
        
        similarities = []
        for caption in generated_captions.values():
            if isinstance(caption, str) and caption.strip():
                caption_embedding = self.sentence_model.encode([caption])
                similarity = cosine_similarity(original_embedding, caption_embedding)[0][0]
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _evaluate_prompt_following(self, image: Image.Image, 
                                 original_prompt: str, 
                                 semiotic_features) -> float:
        """Evaluate how well the image follows the original prompt."""
        
        # This is a composite score based on multiple factors
        score = 0.0
        
        # 1. CLIP score (text-image alignment)
        clip_score = self._calculate_clip_score(image, original_prompt)
        score += clip_score * 0.5
        
        # 2. Semiotic feature alignment
        if semiotic_features and hasattr(semiotic_features, 'semiotic_score'):
            score += semiotic_features.semiotic_score * 0.3
        
        # 3. Architectural relevance (presence of architectural elements)
        prompt_lower = original_prompt.lower()
        architectural_terms = ["building", "architecture", "urban", "city", "structure"]
        architectural_relevance = any(term in prompt_lower for term in architectural_terms)
        score += 0.2 if architectural_relevance else 0.0
        
        return min(score, 1.0)
    
    def _evaluate_semantic_consistency(self, generated_captions: Dict[str, str], 
                                     semiotic_features) -> float:
        """Evaluate semantic consistency across all generated descriptions."""
        
        if not generated_captions or not self.sentence_model:
            return 0.0
        
        # Calculate pairwise similarities between captions
        caption_texts = [caption for caption in generated_captions.values() 
                        if isinstance(caption, str) and caption.strip()]
        
        if len(caption_texts) < 2:
            return 1.0  # Perfect consistency if only one caption
        
        # Encode all captions
        embeddings = self.sentence_model.encode(caption_texts)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1), 
                    embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_semiotic_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate composite semiotic awareness score."""
        
        semiotic_components = [
            metrics.style_accuracy * 0.3,
            metrics.mood_accuracy * 0.3,
            metrics.semiotic_consistency * 0.2,
            metrics.semantic_consistency * 0.2
        ]
        
        return sum(semiotic_components)
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall quality score."""
        
        components = [
            metrics.semiotic_score * 0.3,
            metrics.clip_score * 0.2,
            metrics.architectural_realism * 0.2,
            metrics.image_quality_score * 0.15,
            metrics.prompt_following * 0.15
        ]
        
        return sum(components)
    
    def evaluate_batch(self, image_paths: List[str], 
                      prompts: List[str],
                      output_path: str = None) -> Dict[str, Any]:
        """Evaluate a batch of images and generate comprehensive report."""
        
        logger.info(f"Evaluating batch of {len(image_paths)} images")
        
        all_metrics = []
        detailed_results = []
        
        for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
            logger.info(f"Evaluating image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            metrics = self.evaluate_single_image(image_path, prompt)
            all_metrics.append(metrics)
            
            detailed_results.append({
                "image_path": image_path,
                "prompt": prompt,
                "metrics": asdict(metrics)
            })
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(all_metrics)
        
        # Generate evaluation report
        report = {
            "summary": {
                "total_images": len(image_paths),
                "evaluation_date": pd.Timestamp.now().isoformat(),
                "aggregate_metrics": aggregate_stats
            },
            "detailed_results": detailed_results,
            "analysis": self._generate_analysis(all_metrics)
        }
        
        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Also create visualizations if possible
            if PLOTTING_AVAILABLE:
                self._create_evaluation_visualizations(all_metrics, output_path.parent)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _calculate_aggregate_stats(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics across all evaluated images."""
        
        stats = {}
        
        # Get all metric fields
        metric_fields = [field for field in asdict(metrics_list[0]).keys()]
        
        for field in metric_fields:
            values = [getattr(m, field) for m in metrics_list]
            
            stats[field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return stats
    
    def _generate_analysis(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, Any]:
        """Generate analysis and insights from evaluation results."""
        
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Calculate means for analysis
        mean_metrics = {
            field: np.mean([getattr(m, field) for m in metrics_list])
            for field in asdict(metrics_list[0]).keys()
        }
        
        # Identify strengths
        if mean_metrics["clip_score"] > 0.8:
            analysis["strengths"].append("Strong text-image alignment (high CLIP scores)")
        
        if mean_metrics["semiotic_score"] > 0.7:
            analysis["strengths"].append("Good semiotic awareness and consistency")
        
        if mean_metrics["architectural_realism"] > 0.7:
            analysis["strengths"].append("High architectural realism")
        
        # Identify weaknesses
        if mean_metrics["style_accuracy"] < 0.5:
            analysis["weaknesses"].append("Low architectural style accuracy")
        
        if mean_metrics["object_detection_score"] < 0.5:
            analysis["weaknesses"].append("Poor architectural object detection")
        
        if mean_metrics["semantic_consistency"] < 0.6:
            analysis["weaknesses"].append("Inconsistent semantic representation")
        
        # Generate recommendations
        if mean_metrics["semiotic_score"] < 0.6:
            analysis["recommendations"].append(
                "Improve semiotic conditioning in training data and prompts"
            )
        
        if mean_metrics["architectural_realism"] < 0.6:
            analysis["recommendations"].append(
                "Increase training data with high-quality architectural images"
            )
        
        if mean_metrics["prompt_following"] < 0.6:
            analysis["recommendations"].append(
                "Enhance prompt engineering and conditioning mechanisms"
            )
        
        return analysis
    
    def _create_evaluation_visualizations(self, metrics_list: List[EvaluationMetrics], 
                                        output_dir: Path):
        """Create visualization plots for evaluation results."""
        
        # Create metrics DataFrame
        metrics_data = [asdict(m) for m in metrics_list]
        df = pd.DataFrame(metrics_data)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Semiotic Image Evaluation Results', fontsize=16)
        
        # 1. Overall score distribution
        axes[0, 0].hist(df['overall_score'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Semiotic metrics comparison
        semiotic_cols = ['style_accuracy', 'mood_accuracy', 'semiotic_consistency']
        semiotic_means = df[semiotic_cols].mean()
        axes[0, 1].bar(range(len(semiotic_cols)), semiotic_means, 
                       color=['lightcoral', 'lightgreen', 'gold'])
        axes[0, 1].set_title('Semiotic Metrics Comparison')
        axes[0, 1].set_xticks(range(len(semiotic_cols)))
        axes[0, 1].set_xticklabels([col.replace('_', ' ').title() for col in semiotic_cols])
        axes[0, 1].set_ylabel('Mean Score')
        
        # 3. Quality metrics comparison
        quality_cols = ['clip_score', 'image_quality_score', 'architectural_realism']
        quality_means = df[quality_cols].mean()
        axes[1, 0].bar(range(len(quality_cols)), quality_means, 
                       color=['mediumpurple', 'orange', 'lightblue'])
        axes[1, 0].set_title('Quality Metrics Comparison')
        axes[1, 0].set_xticks(range(len(quality_cols)))
        axes[1, 0].set_xticklabels([col.replace('_', ' ').title() for col in quality_cols])
        axes[1, 0].set_ylabel('Mean Score')
        
        # 4. Score correlation heatmap
        correlation_cols = ['overall_score', 'semiotic_score', 'clip_score', 'architectural_realism']
        corr_matrix = df[correlation_cols].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Score Correlations')
        axes[1, 1].set_xticks(range(len(correlation_cols)))
        axes[1, 1].set_yticks(range(len(correlation_cols)))
        axes[1, 1].set_xticklabels([col.replace('_', ' ').title() for col in correlation_cols], rotation=45)
        axes[1, 1].set_yticklabels([col.replace('_', ' ').title() for col in correlation_cols])
        
        # Add correlation values to heatmap
        for i in range(len(correlation_cols)):
            for j in range(len(correlation_cols)):
                axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation visualizations saved to {output_dir}")

def main():
    """Main execution for testing evaluation system."""
    
    parser = argparse.ArgumentParser(description="Evaluate generated images")
    parser.add_argument("--images", nargs="+", help="Paths to images to evaluate")
    parser.add_argument("--prompts", nargs="+", help="Original prompts for the images")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    try:
        evaluator = SemioticImageEvaluator()
    except Exception as e:
        print(f"Could not initialize evaluator: {e}")
        return
    
    # Example evaluation if no args provided
    if not args.images or not args.prompts:
        print("No test images provided. Example usage:")
        print("python evaluation_pipeline.py --images image1.jpg image2.jpg --prompts 'prompt 1' 'prompt 2' --output report.json")
        return
    
    if len(args.images) != len(args.prompts):
        print("Error: Number of images must match number of prompts")
        return
    
    # Evaluate batch
    report = evaluator.evaluate_batch(
        args.images, 
        args.prompts, 
        output_path=args.output
    )
    
    print("Evaluation completed!")
    print(f"Overall mean score: {report['summary']['aggregate_metrics']['overall_score']['mean']:.3f}")
    print(f"Semiotic mean score: {report['summary']['aggregate_metrics']['semiotic_score']['mean']:.3f}")
    print(f"Report saved to: {args.output}")

if __name__ == "__main__":
    main()