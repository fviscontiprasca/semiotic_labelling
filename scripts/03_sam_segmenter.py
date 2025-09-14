"""
SAM-based Architectural Segmentation Pipeline

This script replaces YOLO with Segment Anything Model (SAM) for comprehensive 
architectural element segmentation suitable for semiotic analysis.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMArchitecturalSegmenter:
    """SAM-based segmentation specifically designed for architectural scene understanding."""
    
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None):
        """
        Initialize SAM for architectural segmentation.
        
        Args:
            model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to SAM checkpoint file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        try:
            # Import SAM components
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            
            # Load SAM model
            if checkpoint_path and Path(checkpoint_path).exists():
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                logger.info(f"Loaded SAM from specified checkpoint: {checkpoint_path}")
            else:
                # Try to load from models/SAM directory
                default_checkpoint = Path(__file__).parent.parent / "models" / "SAM" / "sam_vit_h_4b8939.pth"
                if default_checkpoint.exists():
                    sam = sam_model_registry[model_type](checkpoint=str(default_checkpoint))
                    logger.info(f"Loaded SAM from local models directory: {default_checkpoint}")
                else:
                    logger.error("SAM checkpoint not found. Please ensure sam_vit_h_4b8939.pth is in models/SAM/")
                    sam = None
            
            if sam:
                sam.to(device=self.device)
                
                # Configure mask generator for architectural scenes
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,  # Higher density for architectural details
                    pred_iou_thresh=0.88,  # High quality masks
                    stability_score_thresh=0.95,  # Stable segmentation
                    crop_n_layers=1,  # Process at multiple scales
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,  # Filter tiny segments
                )
                
                logger.info(f"SAM {model_type} initialized successfully on {self.device}")
            else:
                self.mask_generator = None
                
        except ImportError:
            logger.error("SAM not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            self.mask_generator = None
    
    def segment_image(self, image_path: str) -> Dict[str, Any]:
        """
        Segment architectural image using SAM.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing segmentation results and analysis
        """
        if not self.mask_generator:
            return {"error": "SAM not properly initialized"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            logger.info(f"Generating masks for {image_path}")
            masks = self.mask_generator.generate(image_rgb)
            
            # Analyze masks for architectural relevance
            analysis_results = self.analyze_architectural_segments(masks, image_rgb)
            
            return {
                "image_path": image_path,
                "total_segments": len(masks),
                "architectural_segments": analysis_results["architectural_count"],
                "segment_analysis": analysis_results,
                "masks_data": self.process_masks_for_storage(masks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def analyze_architectural_segments(self, masks: List[Dict], image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze segments for architectural relevance and semiotic significance.
        
        Args:
            masks: List of SAM mask dictionaries
            image: Original RGB image
            
        Returns:
            Analysis results with architectural interpretation
        """
        architectural_segments = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, w, h]
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            
            # Extract segment characteristics
            segment_analysis = self.analyze_segment_properties(mask, bbox, area, image)
            
            # Classify architectural relevance
            architectural_type = self.classify_architectural_element(segment_analysis)
            
            if architectural_type != "non_architectural":
                architectural_segments.append({
                    "segment_id": i,
                    "architectural_type": architectural_type,
                    "properties": segment_analysis,
                    "bbox": bbox,
                    "area": area,
                    "stability_score": stability_score,
                    "semiotic_analysis": self.perform_semiotic_analysis(segment_analysis, architectural_type)
                })
        
        # Create pipeline-compatible analysis format
        analysis_result = {
            "total_segments": len(masks),
            "architectural_count": len(architectural_segments),
            "architectural_segments": architectural_segments,
            "coverage_ratio": len(architectural_segments) / len(masks) if masks else 0,
            "dominant_elements": self.identify_dominant_elements(architectural_segments),
            
            # Add fields expected by Phase 04 semiotic extractor
            "density_analysis": {
                "total_coverage": len(architectural_segments) / len(masks) if masks else 0,
                "density_type": "high_density" if len(architectural_segments) > 10 else 
                               "medium_density" if len(architectural_segments) > 5 else "low_density"
            },
            "architectural_hierarchy": {
                "hierarchy_type": "strong_hierarchy" if len(architectural_segments) > 0 else "uniform_scale",
                "dominant_elements": self.identify_dominant_elements(architectural_segments)
            },
            "functional_composition": {
                "architectural_elements": len(architectural_segments),
                "dominant_function": "architectural" if len(architectural_segments) > 0 else "undefined"
            }
        }
        
        return analysis_result
    
    def analyze_segment_properties(self, mask: np.ndarray, bbox: List[float], 
                                 area: int, image: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric and visual properties of a segment."""
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 0
        
        # Extract masked region
        masked_region = image * np.expand_dims(mask, axis=2)
        
        # Color analysis
        mask_pixels = image[mask]
        if len(mask_pixels) > 0:
            mean_color = np.mean(mask_pixels, axis=0)
            color_std = np.std(mask_pixels, axis=0)
        else:
            mean_color = [0, 0, 0]
            color_std = [0, 0, 0]
        
        # Shape analysis
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            compactness = 0
        
        return {
            "bbox": bbox,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
            "mean_color": mean_color.tolist() if hasattr(mean_color, 'tolist') else list(mean_color),
            "color_std": color_std.tolist() if hasattr(color_std, 'tolist') else list(color_std),
            "position": {"center_x": x + w/2, "center_y": y + h/2},
            "relative_size": area / (image.shape[0] * image.shape[1])
        }
    
    def classify_architectural_element(self, properties: Dict[str, Any]) -> str:
        """
        Classify segment as architectural element type based on properties.
        
        This is a heuristic-based classifier that can be enhanced with ML models.
        """
        aspect_ratio = properties["aspect_ratio"]
        relative_size = properties["relative_size"]
        compactness = properties["compactness"]
        
        # Heuristic classification rules
        if relative_size > 0.3:
            return "facade_or_building"
        elif aspect_ratio > 2.0 and relative_size > 0.05:
            return "horizontal_element"  # balconies, cornices
        elif aspect_ratio < 0.5 and relative_size > 0.02:
            return "vertical_element"  # columns, pillars
        elif 0.8 < aspect_ratio < 1.2 and 0.001 < relative_size < 0.1:
            return "window_or_opening"
        elif compactness > 0.7 and relative_size < 0.05:
            return "decorative_element"
        elif relative_size > 0.1:
            return "structural_element"
        else:
            return "detail_or_texture"
    
    def perform_semiotic_analysis(self, properties: Dict[str, Any], 
                                architectural_type: str) -> Dict[str, Any]:
        """
        Perform semiotic analysis on architectural elements.
        """
        analysis = {
            "architectural_type": architectural_type,
            "semiotic_indicators": []
        }
        
        # Add semiotic interpretations based on element type and properties
        if architectural_type == "facade_or_building":
            analysis["semiotic_indicators"].extend([
                "primary_architectural_identity",
                "cultural_architectural_style",
                "power_and_authority_symbolism"
            ])
        elif architectural_type == "window_or_opening":
            analysis["semiotic_indicators"].extend([
                "transparency_and_openness",
                "interior_exterior_relationship",
                "privacy_and_accessibility"
            ])
        elif architectural_type == "decorative_element":
            analysis["semiotic_indicators"].extend([
                "aesthetic_elaboration",
                "cultural_ornamentation",
                "status_and_craftsmanship"
            ])
        elif architectural_type == "vertical_element":
            analysis["semiotic_indicators"].extend([
                "structural_support_symbolism",
                "vertical_aspiration",
                "classical_architectural_reference"
            ])
        
        # Add contextual analysis based on properties
        if properties["relative_size"] > 0.2:
            analysis["semiotic_indicators"].append("dominant_visual_element")
        
        return analysis
    
    def identify_dominant_elements(self, architectural_segments: List[Dict]) -> List[str]:
        """Identify the most prominent architectural elements in the scene."""
        element_counts = {}
        for segment in architectural_segments:
            arch_type = segment["architectural_type"]
            element_counts[arch_type] = element_counts.get(arch_type, 0) + 1
        
        # Sort by frequency
        dominant = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
        return [elem[0] for elem in dominant[:3]]  # Top 3
    
    def process_masks_for_storage(self, masks: List[Dict]) -> List[Dict]:
        """Process masks for JSON serialization."""
        processed = []
        for i, mask_data in enumerate(masks):
            # Don't store the actual mask pixels (too large), just metadata
            processed.append({
                "mask_id": i,
                "bbox": mask_data['bbox'],
                "area": mask_data['area'],
                "stability_score": float(mask_data['stability_score']),
                "predicted_iou": float(mask_data.get('predicted_iou', 0))
            })
        return processed
    
    def segment_directory(self, input_dir: str, output_dir: str, max_images: Optional[int] = None):
        """Segment all images in a directory - main pipeline integration method."""
        """
        Segment all images in a directory using SAM.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output results
            max_images: Maximum number of images to process (None for all)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        analysis_dir = output_path / "analysis"
        masks_dir = output_path / "masks"
        analysis_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Find image files (avoiding duplicates from case variations)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        # Remove duplicates and sort for consistent processing
        image_files = sorted(list(set(image_files)))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "processed_images": 0,
            "failed_images": 0,
            "total_segments": 0,
            "total_architectural_segments": 0,
            "model_info": {
                "type": "SAM",
                "variant": self.model_type,
                "device": str(self.device)
            },
            "image_results": []
        }
        
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
                
                # Segment image
                result = self.segment_image(str(image_file))
                
                if "error" in result:
                    logger.error(f"Failed to process {image_file.name}: {result['error']}")
                    results_summary["failed_images"] += 1
                    continue
                
                # Save individual analysis
                analysis_file = analysis_dir / f"{image_file.stem}_sam_analysis.json"
                with open(analysis_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Update summary
                results_summary["processed_images"] += 1
                results_summary["total_segments"] += result["total_segments"]
                results_summary["total_architectural_segments"] += result["architectural_segments"]
                
                results_summary["image_results"].append({
                    "image_name": image_file.name,
                    "total_segments": result["total_segments"],
                    "architectural_segments": result["architectural_segments"],
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Unexpected error processing {image_file.name}: {str(e)}")
                results_summary["failed_images"] += 1
                results_summary["image_results"].append({
                    "image_name": image_file.name,
                    "error": str(e),
                    "success": False
                })
        
        # Save summary in format compatible with pipeline expectations
        summary_file = output_path / "segmentation_summary.json"
        
        # Create pipeline-compatible summary format
        pipeline_compatible_summary = {}
        for result in results_summary["image_results"]:
            if result["success"]:
                image_name = result["image_name"]
                # Read the detailed analysis for this image
                analysis_file = analysis_dir / f"{Path(image_name).stem}_sam_analysis.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        detailed_analysis = json.load(f)
                    
                    # Format compatible with Phase 04 expectations
                    pipeline_compatible_summary[image_name] = {
                        "total_segments": result["total_segments"],
                        "architectural_segments": result["architectural_segments"],
                        "semiotic_segmentation_analysis": detailed_analysis["segment_analysis"],
                        "model_type": "SAM",
                        "timestamp": detailed_analysis["timestamp"]
                    }
        
        with open(summary_file, 'w') as f:
            json.dump(pipeline_compatible_summary, f, indent=2)
        
        # Also save the detailed summary
        detailed_summary_file = output_path / "sam_segmentation_summary.json"
        with open(detailed_summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"✅ SAM segmentation complete!")
        logger.info(f"Processed: {results_summary['processed_images']}/{results_summary['total_images']}")
        logger.info(f"Total segments: {results_summary['total_segments']}")
        logger.info(f"Architectural segments: {results_summary['total_architectural_segments']}")
        logger.info(f"Results saved to: {output_path}")
        
        return results_summary
    
    def segment_full_dataset(self, base_dir: str, output_dir: str, max_images: Optional[int] = None):
        """
        Segment complete dataset including both train and val directories.
        
        Args:
            base_dir: Base directory containing train/ and val/ subdirectories
            output_dir: Directory for output results
            max_images: Maximum number of images to process (None for all)
        """
        base_path = Path(base_dir)
        train_dir = base_path / "train"
        val_dir = base_path / "val"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        analysis_dir = output_path / "analysis"
        masks_dir = output_path / "masks"
        analysis_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Collect all image files from both directories
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_image_files = []
        
        for directory in [train_dir, val_dir]:
            if directory.exists():
                logger.info(f"Scanning {directory}")
                for ext in image_extensions:
                    image_files = list(directory.glob(f"*{ext}"))
                    all_image_files.extend([(f, directory.name) for f in image_files])
                    logger.info(f"Found {len(image_files)} {ext} files in {directory.name}")
        
        # Remove duplicates and sort for consistent processing
        all_image_files = sorted(list(set(all_image_files)), key=lambda x: x[0].name)
        
        if max_images:
            all_image_files = all_image_files[:max_images]
        
        logger.info(f"Total images to process: {len(all_image_files)}")
        
        results_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(all_image_files),
            "processed_images": 0,
            "failed_images": 0,
            "total_segments": 0,
            "total_architectural_segments": 0,
            "model_info": {
                "type": "SAM",
                "variant": self.model_type,
                "device": str(self.device)
            },
            "train_images": len([f for f, split in all_image_files if split == "train"]),
            "val_images": len([f for f, split in all_image_files if split == "val"]),
            "image_results": []
        }
        
        for i, (image_file, split) in enumerate(all_image_files):
            try:
                logger.info(f"Processing {i+1}/{len(all_image_files)}: {split}/{image_file.name}")
                
                # Segment image
                result = self.segment_image(str(image_file))
                
                if "error" in result:
                    logger.error(f"Failed to process {image_file.name}: {result['error']}")
                    results_summary["failed_images"] += 1
                    continue
                
                # Save individual analysis with split prefix
                analysis_file = analysis_dir / f"{split}_{image_file.stem}_sam_analysis.json"
                with open(analysis_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Update summary
                results_summary["processed_images"] += 1
                results_summary["total_segments"] += result["total_segments"]
                results_summary["total_architectural_segments"] += result["architectural_segments"]
                
                results_summary["image_results"].append({
                    "image_name": image_file.name,
                    "split": split,
                    "total_segments": result["total_segments"],
                    "architectural_segments": result["architectural_segments"],
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Unexpected error processing {image_file.name}: {str(e)}")
                results_summary["failed_images"] += 1
                results_summary["image_results"].append({
                    "image_name": image_file.name,
                    "split": split,
                    "error": str(e),
                    "success": False
                })
        
        # Save summary in format compatible with pipeline expectations
        summary_file = output_path / "segmentation_summary.json"
        
        # Create pipeline-compatible summary format
        pipeline_compatible_summary = {}
        for result in results_summary["image_results"]:
            if result["success"]:
                image_name = result["image_name"]
                split = result["split"]
                # Read the detailed analysis for this image
                analysis_file = analysis_dir / f"{split}_{Path(image_name).stem}_sam_analysis.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        detailed_analysis = json.load(f)
                    
                    # Format compatible with Phase 04 expectations
                    pipeline_compatible_summary[f"{split}/{image_name}"] = {
                        "total_segments": result["total_segments"],
                        "architectural_segments": result["architectural_segments"],
                        "semiotic_segmentation_analysis": detailed_analysis["segment_analysis"],
                        "model_type": "SAM",
                        "split": split,
                        "timestamp": detailed_analysis["timestamp"]
                    }
        
        with open(summary_file, 'w') as f:
            json.dump(pipeline_compatible_summary, f, indent=2)
        
        # Also save the detailed summary
        detailed_summary_file = output_path / "sam_full_dataset_summary.json"
        with open(detailed_summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"✅ SAM full dataset segmentation complete!")
        logger.info(f"Processed: {results_summary['processed_images']}/{results_summary['total_images']}")
        logger.info(f"Train images: {results_summary['train_images']}, Val images: {results_summary['val_images']}")
        logger.info(f"Total segments: {results_summary['total_segments']}")
        logger.info(f"Architectural segments: {results_summary['total_architectural_segments']}")
        logger.info(f"Results saved to: {output_path}")
        
        return results_summary

def main():
    parser = argparse.ArgumentParser(description="SAM-based architectural segmentation")
    parser.add_argument("--input_dir", help="Input directory with images (single directory)")
    parser.add_argument("--dataset_dir", help="Dataset base directory containing train/ and val/ subdirectories")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model variant")
    parser.add_argument("--checkpoint", help="Path to SAM checkpoint file")
    parser.add_argument("--max_images", type=int, help="Maximum images to process")
    
    args = parser.parse_args()
    
    # Initialize SAM segmenter
    segmenter = SAMArchitecturalSegmenter(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint
    )
    
    if not segmenter.mask_generator:
        print("❌ SAM not properly initialized. Please install SAM and download checkpoints.")
        print("Installation: pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("Checkpoints: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        return
    
    # Validate arguments
    if not args.input_dir and not args.dataset_dir:
        print("❌ Please specify either --input_dir for single directory or --dataset_dir for full dataset processing")
        return
    
    if args.input_dir and args.dataset_dir:
        print("❌ Please specify either --input_dir OR --dataset_dir, not both")
        return
    
    # Process directory or full dataset
    if args.dataset_dir:
        print("🔄 Processing full dataset (train + val directories)")
        results = segmenter.segment_full_dataset(
            base_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_images=args.max_images
        )
    else:
        print("🔄 Processing single directory")
        results = segmenter.segment_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_images=args.max_images
        )
    
    print(f"\n✅ SAM segmentation completed successfully!")
    print(f"Success rate: {results['processed_images']}/{results['total_images']} images")
    print(f"Architectural elements found: {results['total_architectural_segments']}")

if __name__ == "__main__":
    main()