"""
YOLO11 segmentation pipeline for extracting architectural and urban elements
with semiotic awareness for Flux.1d fine-tuning.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import fiftyone as fo
import fiftyone.utils.labels as foul
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemioticSegmentation:
    """Container for segmentation results with semiotic interpretation."""
    masks: np.ndarray
    classes: List[str]
    confidences: List[float]
    bboxes: List[Tuple[int, int, int, int]]
    semiotic_analysis: Dict[str, Any]

class UrbanYOLOSegmenter:
    """YOLO11-based segmentation system for urban architectural analysis."""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """Initialize YOLO11 segmentation model."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load YOLO11 segmentation model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom YOLO11 model from {model_path}")
        else:
            # Use pre-trained YOLO11 segmentation model from models folder
            default_model_path = Path(__file__).parent.parent / "models" / "YOLO" / "yolo11n-seg.pt"
            if default_model_path.exists():
                self.model = YOLO(str(default_model_path))
                logger.info(f"Loaded pre-trained YOLO11n-seg model from {default_model_path}")
            else:
                # Fallback to auto-download if not found in models folder
                self.model = YOLO('yolo11n-seg.pt')  # nano version for speed
                logger.info("Loaded pre-trained YOLO11n-seg model (auto-downloaded)")
        
        # Urban class mapping for semiotic analysis
        self.urban_class_mapping = {
            "building": "architectural_structure",
            "house": "residential_architecture", 
            "skyscraper": "high_rise_architecture",
            "tower": "vertical_landmark",
            "castle": "historical_architecture",
            "office building": "commercial_architecture",
            "convenience store": "commercial_structure",
            "lighthouse": "maritime_architecture",
            "fountain": "urban_amenity",
            "cart": "urban_furniture",
            "person": "human_presence",
            "car": "urban_mobility",
            "truck": "urban_logistics",
            "bus": "public_transport",
            "bicycle": "sustainable_mobility",
            "motorcycle": "personal_transport",
            "traffic light": "urban_infrastructure",
            "stop sign": "regulatory_signage",
            "bench": "public_furniture",
            "fire hydrant": "safety_infrastructure",
            "parking meter": "urban_management",
            "tree": "urban_greenery",
            "potted plant": "decorative_vegetation"
        }
        
        # Semiotic interpretation rules
        self.semiotic_rules = {
            "density_analysis": {
                "high_density": {"threshold": 10, "meaning": "urban_intensity"},
                "medium_density": {"threshold": 5, "meaning": "balanced_urbanism"},
                "low_density": {"threshold": 2, "meaning": "spacious_environment"}
            },
            "vertical_hierarchy": {
                "dominant_vertical": {"meaning": "monumentality"},
                "mixed_heights": {"meaning": "urban_diversity"},
                "horizontal_emphasis": {"meaning": "human_scale"}
            },
            "functional_mix": {
                "residential_dominant": {"meaning": "living_oriented"},
                "commercial_dominant": {"meaning": "activity_oriented"},
                "mixed_use": {"meaning": "urban_complexity"},
                "infrastructure_heavy": {"meaning": "utility_focused"}
            }
        }
    
    def segment_image(self, image_path: str, 
                     confidence_threshold: float = 0.5) -> SemioticSegmentation:
        """Perform segmentation with semiotic analysis."""
        
        # Run YOLO inference
        results = self.model(image_path, conf=confidence_threshold, device=self.device)
        
        if not results or not results[0].masks:
            return SemioticSegmentation(
                masks=np.array([]),
                classes=[],
                confidences=[],
                bboxes=[],
                semiotic_analysis={}
            )
        
        result = results[0]
        
        # Extract segmentation data
        masks = result.masks.data.cpu().numpy()  # (N, H, W)
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        classes = result.boxes.cls.cpu().numpy()  # (N,)
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        
        # Convert class indices to names
        class_names = [self.model.names[int(cls)] for cls in classes]
        
        # Perform semiotic analysis
        semiotic_analysis = self._analyze_semiotic_content(
            masks, class_names, boxes, confidences
        )
        
        return SemioticSegmentation(
            masks=masks,
            classes=class_names,
            confidences=confidences.tolist(),
            bboxes=boxes.tolist(),
            semiotic_analysis=semiotic_analysis
        )
    
    def _analyze_semiotic_content(self, masks: np.ndarray, 
                                classes: List[str],
                                boxes: np.ndarray,
                                confidences: np.ndarray) -> Dict[str, Any]:
        """Analyze semiotic content from segmentation results."""
        
        analysis = {
            "object_count": len(classes),
            "detected_classes": classes,
            "urban_categories": self._categorize_urban_elements(classes),
            "spatial_analysis": self._analyze_spatial_relationships(boxes, classes),
            "density_analysis": self._analyze_density(masks, classes),
            "architectural_hierarchy": self._analyze_architectural_hierarchy(boxes, classes),
            "functional_composition": self._analyze_functional_composition(classes),
            "semiotic_interpretation": {}
        }
        
        # Generate semiotic interpretations
        analysis["semiotic_interpretation"] = self._generate_semiotic_interpretation(analysis)
        
        return analysis
    
    def _categorize_urban_elements(self, classes: List[str]) -> Dict[str, List[str]]:
        """Categorize detected elements by urban function."""
        
        categories = {
            "architectural_structures": [],
            "infrastructure": [],
            "mobility": [],
            "vegetation": [],
            "urban_furniture": [],
            "human_activity": []
        }
        
        architectural = ["building", "house", "skyscraper", "tower", "castle", "office building"]
        infrastructure = ["traffic light", "stop sign", "fire hydrant", "parking meter"]
        mobility = ["car", "truck", "bus", "bicycle", "motorcycle"]
        vegetation = ["tree", "potted plant"]
        furniture = ["bench", "cart", "fountain"]
        human = ["person"]
        
        for cls in classes:
            if cls in architectural:
                categories["architectural_structures"].append(cls)
            elif cls in infrastructure:
                categories["infrastructure"].append(cls)
            elif cls in mobility:
                categories["mobility"].append(cls)
            elif cls in vegetation:
                categories["vegetation"].append(cls)
            elif cls in furniture:
                categories["urban_furniture"].append(cls)
            elif cls in human:
                categories["human_activity"].append(cls)
        
        return categories
    
    def _analyze_spatial_relationships(self, boxes: np.ndarray, 
                                     classes: List[str]) -> Dict[str, Any]:
        """Analyze spatial relationships between urban elements."""
        
        if len(boxes) == 0:
            return {}
        
        # Calculate centroids
        centroids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            centroids.append((centroid_x, centroid_y))
        
        # Analyze vertical distribution
        y_coords = [c[1] for c in centroids]
        vertical_distribution = {
            "mean_height": np.mean(y_coords),
            "height_variance": np.var(y_coords),
            "vertical_spread": "high" if np.var(y_coords) > 1000 else "low"
        }
        
        # Analyze clustering
        clustering = self._analyze_spatial_clustering(centroids, classes)
        
        return {
            "vertical_distribution": vertical_distribution,
            "spatial_clustering": clustering,
            "element_proximity": self._calculate_proximity_matrix(centroids, classes)
        }
    
    def _analyze_spatial_clustering(self, centroids: List[Tuple[float, float]], 
                                  classes: List[str]) -> Dict[str, Any]:
        """Analyze spatial clustering of urban elements."""
        
        if len(centroids) < 2:
            return {"clustering_type": "isolated"}
        
        # Simple clustering analysis based on distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                             (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        mean_distance = np.mean(distances)
        
        if mean_distance < 100:
            clustering_type = "highly_clustered"
        elif mean_distance < 300:
            clustering_type = "moderately_clustered"
        else:
            clustering_type = "dispersed"
        
        return {
            "clustering_type": clustering_type,
            "mean_distance": mean_distance,
            "distance_variance": np.var(distances)
        }
    
    def _calculate_proximity_matrix(self, centroids: List[Tuple[float, float]], 
                                  classes: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate proximity relationships between different urban element types."""
        
        proximity_matrix = defaultdict(lambda: defaultdict(list))
        
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                if i != j:
                    dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                 (centroids[i][1] - centroids[j][1])**2)
                    proximity_matrix[cls1][cls2].append(dist)
        
        # Average distances between class types
        avg_proximity = {}
        for cls1, cls2_dict in proximity_matrix.items():
            avg_proximity[cls1] = {}
            for cls2, distances in cls2_dict.items():
                avg_proximity[cls1][cls2] = np.mean(distances) if distances else float('inf')
        
        return avg_proximity
    
    def _analyze_density(self, masks: np.ndarray, classes: List[str]) -> Dict[str, Any]:
        """Analyze density and coverage of urban elements."""
        
        if len(masks) == 0:
            return {"total_coverage": 0, "density_type": "empty"}
        
        # Calculate total coverage
        combined_mask = np.any(masks, axis=0)
        total_coverage = np.sum(combined_mask) / combined_mask.size
        
        # Analyze per-class coverage
        class_coverage = {}
        for i, cls in enumerate(classes):
            coverage = np.sum(masks[i]) / masks[i].size
            class_coverage[cls] = coverage
        
        # Determine density type
        if total_coverage > 0.6:
            density_type = "high_density"
        elif total_coverage > 0.3:
            density_type = "medium_density"
        else:
            density_type = "low_density"
        
        return {
            "total_coverage": total_coverage,
            "class_coverage": class_coverage,
            "density_type": density_type
        }
    
    def _analyze_architectural_hierarchy(self, boxes: np.ndarray, 
                                       classes: List[str]) -> Dict[str, Any]:
        """Analyze architectural hierarchy and scale relationships."""
        
        if len(boxes) == 0:
            return {}
        
        # Calculate object sizes
        sizes = []
        size_by_class = defaultdict(list)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            sizes.append(area)
            size_by_class[classes[i]].append(area)
        
        # Analyze size hierarchy
        size_variance = np.var(sizes)
        
        if size_variance > 10000:
            hierarchy_type = "strong_hierarchy"
        elif size_variance > 1000:
            hierarchy_type = "moderate_hierarchy"
        else:
            hierarchy_type = "uniform_scale"
        
        # Find dominant elements
        dominant_classes = []
        for cls, class_sizes in size_by_class.items():
            avg_size = np.mean(class_sizes)
            if avg_size > np.mean(sizes) * 1.5:
                dominant_classes.append(cls)
        
        return {
            "hierarchy_type": hierarchy_type,
            "size_variance": size_variance,
            "dominant_elements": dominant_classes,
            "scale_relationships": dict(size_by_class)
        }
    
    def _analyze_functional_composition(self, classes: List[str]) -> Dict[str, Any]:
        """Analyze the functional composition of the urban scene."""
        
        # Count by function type
        functional_counts = defaultdict(int)
        
        for cls in classes:
            if cls in ["building", "house", "skyscraper", "tower"]:
                functional_counts["residential_commercial"] += 1
            elif cls in ["car", "truck", "bus", "bicycle"]:
                functional_counts["mobility"] += 1
            elif cls in ["tree", "potted plant"]:
                functional_counts["vegetation"] += 1
            elif cls in ["person"]:
                functional_counts["human_activity"] += 1
            else:
                functional_counts["infrastructure"] += 1
        
        # Determine dominant function
        if functional_counts:
            dominant_function = max(functional_counts, key=functional_counts.get)
        else:
            dominant_function = "undefined"
        
        return {
            "functional_counts": dict(functional_counts),
            "dominant_function": dominant_function,
            "functional_diversity": len(functional_counts)
        }
    
    def _generate_semiotic_interpretation(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate semiotic interpretation from analysis."""
        
        interpretations = {}
        
        # Density interpretation
        density_type = analysis.get("density_analysis", {}).get("density_type", "unknown")
        if density_type in self.semiotic_rules["density_analysis"]:
            interpretations["spatial_character"] = self.semiotic_rules["density_analysis"][density_type]["meaning"]
        
        # Hierarchy interpretation
        hierarchy = analysis.get("architectural_hierarchy", {}).get("hierarchy_type", "unknown")
        if hierarchy == "strong_hierarchy":
            interpretations["visual_power"] = "monumentality"
        elif hierarchy == "uniform_scale":
            interpretations["visual_power"] = "egalitarian"
        else:
            interpretations["visual_power"] = "balanced"
        
        # Functional interpretation
        dominant_function = analysis.get("functional_composition", {}).get("dominant_function", "undefined")
        interpretations["urban_character"] = dominant_function
        
        # Presence of humans
        human_count = analysis.get("urban_categories", {}).get("human_activity", [])
        if human_count:
            interpretations["social_vitality"] = "inhabited"
        else:
            interpretations["social_vitality"] = "uninhabited"
        
        return interpretations
    
    def process_dataset(self, dataset: fo.Dataset, 
                       segmentation_field: str = "yolo_segmentation",
                       batch_size: int = 1) -> None:
        """Process FiftyOne dataset to add YOLO segmentation analysis."""
        
        logger.info(f"Processing {len(dataset)} samples for YOLO segmentation")
        
        processed = 0
        for sample in dataset.iter_samples(progress=True):
            try:
                # Perform segmentation
                seg_result = self.segment_image(sample.filepath)
                
                # Get image dimensions for FiftyOne format conversion
                img = Image.open(sample.filepath)
                img_width, img_height = img.size
                
                # Convert to FiftyOne format
                fo_detections = self._convert_to_fiftyone_format(seg_result, img_width, img_height)
                
                # Add to sample
                sample[segmentation_field] = fo_detections
                sample["semiotic_segmentation_analysis"] = seg_result.semiotic_analysis
                sample.save()
                
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing {sample.filepath}: {e}")
                continue
        
        logger.info(f"Completed segmentation processing for {processed} samples")
    
    def _convert_to_fiftyone_format(self, seg_result: SemioticSegmentation, 
                                   image_width: int, image_height: int) -> fo.Detections:
        """Convert segmentation result to FiftyOne format."""
        
        detections = []
        
        for i, (cls, conf, bbox) in enumerate(zip(seg_result.classes, 
                                                 seg_result.confidences, 
                                                 seg_result.bboxes)):
            
            x1, y1, x2, y2 = bbox
            
            # Convert to relative coordinates (FiftyOne format)
            rel_bbox = [x1/image_width, y1/image_height, 
                       (x2-x1)/image_width, (y2-y1)/image_height]
            
            detection = fo.Detection(
                label=cls,
                bounding_box=rel_bbox,
                confidence=conf,
                mask=seg_result.masks[i] if len(seg_result.masks) > i else None
            )
            
            detections.append(detection)
        
        return fo.Detections(detections=detections)
    
    def visualize_segmentation(self, image_path: str, 
                             seg_result: SemioticSegmentation,
                             output_path: str = None) -> np.ndarray:
        """Visualize segmentation results with semiotic annotations."""
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = image.copy()
        
        # Draw masks
        colors = plt.cm.Set3(np.linspace(0, 1, len(seg_result.classes)))
        
        for i, (mask, cls, conf) in enumerate(zip(seg_result.masks, 
                                                 seg_result.classes, 
                                                 seg_result.confidences)):
            
            if len(mask.shape) == 2:
                mask_resized = cv2.resize(mask.astype(np.uint8), 
                                        (image.shape[1], image.shape[0]))
                
                color = (colors[i][:3] * 255).astype(np.uint8)
                overlay[mask_resized > 0.5] = overlay[mask_resized > 0.5] * 0.7 + color * 0.3
        
        # Add semiotic interpretation text
        y_offset = 30
        interpretation = seg_result.semiotic_analysis.get("semiotic_interpretation", {})
        
        for key, value in interpretation.items():
            text = f"{key}: {value}"
            cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 25
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        return overlay

    def segment_directory(self, input_dir: str, output_dir: str, 
                         confidence_threshold: float = 0.5,
                         save_visualizations: bool = False) -> None:
        """Segment all images in a directory and save results as JSON."""
        
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        base = Path(input_dir)
        images = [p for p in base.rglob("*") if p.suffix.lower() in img_exts]
        
        logger.info(f"Found {len(images)} images under {base}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        masks_dir = output_path / "masks"
        analysis_dir = output_path / "analysis"
        viz_dir = output_path / "visualizations"
        
        masks_dir.mkdir(exist_ok=True)
        analysis_dir.mkdir(exist_ok=True)
        if save_visualizations:
            viz_dir.mkdir(exist_ok=True)
        
        # Process all images
        all_results = {}
        
        for idx, img_path in enumerate(images, 1):
            try:
                # Get relative path for consistent naming
                rel_path = str(img_path.relative_to(base))
                
                # Perform segmentation
                seg_result = self.segment_image(str(img_path), confidence_threshold)
                
                # Save individual analysis
                analysis_file = analysis_dir / f"{img_path.stem}_analysis.json"
                with analysis_file.open("w", encoding="utf-8") as f:
                    json.dump({
                        "image_path": rel_path,
                        "classes": seg_result.classes,
                        "confidences": seg_result.confidences,
                        "bboxes": seg_result.bboxes,
                        "semiotic_analysis": seg_result.semiotic_analysis
                    }, f, indent=2, ensure_ascii=False)
                
                # Save masks as numpy arrays
                if len(seg_result.masks) > 0:
                    masks_file = masks_dir / f"{img_path.stem}_masks.npz"
                    np.savez_compressed(masks_file, 
                                      masks=seg_result.masks,
                                      classes=seg_result.classes,
                                      confidences=seg_result.confidences,
                                      bboxes=seg_result.bboxes)
                
                # Save visualization if requested
                if save_visualizations and len(seg_result.masks) > 0:
                    viz_file = viz_dir / f"{img_path.stem}_segmented.jpg"
                    self.visualize_segmentation(str(img_path), seg_result, str(viz_file))
                
                # Add to combined results (compatible with Phase 04 expectations)
                all_results[rel_path] = {
                    "classes": seg_result.classes,
                    "confidences": seg_result.confidences,
                    "bboxes": seg_result.bboxes,
                    "num_objects": len(seg_result.classes),
                    "semiotic_segmentation_analysis": seg_result.semiotic_analysis,  # Expected by Phase 04
                    "semiotic_interpretation": seg_result.semiotic_analysis.get("semiotic_interpretation", {})
                }
                
                if idx % 25 == 0:
                    logger.info(f"Segmented {idx}/{len(images)} images")
                    
            except Exception as e:
                logger.error(f"Failed to segment {img_path}: {e}")
                continue
        
        # Save combined results
        summary_file = output_path / "segmentation_summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Completed segmentation: {len(all_results)} images processed")
        logger.info(f"Results saved to: {output_path}")

def main():
    """CLI: segment an image directory and save results; fallback to demo."""
    parser = argparse.ArgumentParser(description="YOLO11 Urban Segmenter (03)")
    parser.add_argument("--input_dir", default=None, help="Directory with images to segment")
    parser.add_argument("--output_dir", default=None, help="Output directory for segmentation results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--visualizations", action="store_true", help="Save visualization images")
    parser.add_argument("--model_path", default=None, help="Path to custom YOLO model")
    args = parser.parse_args()
    
    # If CLI paths provided, run directory mode
    if args.input_dir:
        default_out = Path(__file__).parent.parent / "data" / "outputs" / "03_yolo_segmentation"
        output_dir = args.output_dir if args.output_dir else str(default_out)
        
        segmenter = UrbanYOLOSegmenter(model_path=args.model_path, device=args.device)
        segmenter.segment_directory(
            args.input_dir, 
            output_dir, 
            confidence_threshold=args.confidence,
            save_visualizations=args.visualizations
        )
        return
    
    # Fallback: FiftyOne demo
    segmenter = UrbanYOLOSegmenter(device=args.device)
    dataset_name = "semiotic_urban_combined"
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        
        # Process subset for testing
        test_view = dataset.take(3)
        segmenter.process_dataset(test_view)
        
        # Launch FiftyOne to view results
        session = fo.launch_app(dataset, port=5151)
        print("YOLO segmentation completed. View results at http://localhost:5151")
        
        try:
            session.wait()
        except KeyboardInterrupt:
            print("Shutting down...")
    else:
        print(f"Dataset {dataset_name} not found. Provide --input_dir to segment a folder.")

if __name__ == "__main__":
    main()