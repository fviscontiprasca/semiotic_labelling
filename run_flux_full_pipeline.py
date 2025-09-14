#!/usr/bin/env python3
"""
Full Fine-tuning Pipeline Orchestrator
======================================

This script orchestrates the complete full fine-tuning variant of the semiotic labelling pipeline.
It integrates the standard pipeline phases (01-04) with the specialized full fine-tuning phases (05-08)
to provide a complete alternative to the LoRA-based approach.

The orchestrator manages:
- Data preparation and unification (phases 01-04)
- Enhanced data preparation for full fine-tuning (phase 05)
- Full model fine-tuning (phase 06)
- Model inference with fine-tuned models (phase 08)

Author: [Your Name]
Project: Semiocity - Semiotic Labelling Pipeline
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import fiftyone as fo

# Add scripts directory to path
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

@dataclass
class PipelineConfig:
    """Configuration for the full fine-tuning pipeline"""
    # Data paths
    oid_data_path: str = "data/oid_urban"
    synthetic_data_path: str = "data/imaginary_synthetic"
    export_path: str = "data/export"
    
    # Model configuration
    model_output_path: str = "models/flux_full_finetuned"
    base_model: str = "black-forest-labs/FLUX.1-dev"
    
    # Training configuration
    num_train_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    mixed_precision: str = "fp16"
    
    # Data preparation
    min_semiotic_score: float = 0.5
    max_samples_per_class: int = 1000
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Inference configuration
    inference_output_dir: str = "outputs/flux_full_inference"
    num_inference_samples: int = 4
    inference_steps: int = 30
    guidance_scale: float = 7.5
    
    # Pipeline control
    skip_phases: List[str] = field(default_factory=list)
    phases_to_run: List[str] = field(default_factory=lambda: ["01", "02", "03", "04", "05", "06", "08"])
    
    # Advanced settings
    enable_fiftyone_app: bool = False
    save_intermediate_results: bool = True
    cleanup_temp_files: bool = False

class FullFinetuningOrchestrator:
    """Orchestrates the complete full fine-tuning pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        self.start_time = time.time()
        
        # Define phase mappings
        self.phase_scripts = {
            "01": "01_data_pipeline.py",
            "02": "02_blip2_captioner.py", 
            "03": "03_yolo_segmenter.py",
            "04": "04_semiotic_extractor.py",
            "05": "flux_full_finetune/05_flux_full_data_prep.py",
            "06": "flux_full_finetune/06_flux_full_trainer.py",
            "08": "flux_full_finetune/08_flux_full_inference.py"
        }
        
        self.phase_descriptions = {
            "01": "Data Pipeline - Unify OID and synthetic datasets",
            "02": "BLIP-2 Captioning - Generate image captions",
            "03": "YOLO Segmentation - Create segmentation masks",
            "04": "Semiotic Extraction - Extract semiotic features",
            "05": "Full Fine-tuning Data Prep - Prepare enhanced training data",
            "06": "Full Fine-tuning - Train complete Flux model",
            "08": "Full Model Inference - Generate images with fine-tuned model"
        }
        
        self._validate_environment()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "full_pipeline.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _validate_environment(self):
        """Validate that all required components are available"""
        self.logger.info("Validating environment...")
        
        # Check if required scripts exist
        missing_scripts = []
        for phase, script_name in self.phase_scripts.items():
            script_path = script_dir / script_name
            if not script_path.exists():
                missing_scripts.append(f"Phase {phase}: {script_name}")
        
        if missing_scripts:
            error_msg = f"Missing required scripts: {', '.join(missing_scripts)}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check data directories
        required_dirs = [self.config.oid_data_path, self.config.synthetic_data_path]
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                self.logger.warning(f"Data directory not found: {dir_path}")
        
        # Create output directories
        Path(self.config.export_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_output_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.inference_output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Environment validation complete")
    
    def run_phase(self, phase: str) -> bool:
        """Run a specific pipeline phase"""
        if phase in self.config.skip_phases:
            self.logger.info(f"Skipping phase {phase} as requested")
            return True
            
        script_name = self.phase_scripts[phase]
        description = self.phase_descriptions[phase]
        
        self.logger.info(f"Running Phase {phase}: {description}")
        self.logger.info(f"Executing: {script_name}")
        
        # Build command arguments based on phase
        cmd = self._build_phase_command(phase, script_name)
        
        try:
            start_time = time.time()
            
            # Execute the phase script
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            execution_time = time.time() - start_time
            
            # Log results
            if result.returncode == 0:
                self.logger.info(f"Phase {phase} completed successfully in {execution_time:.2f}s")
                self.results[phase] = {
                    "status": "success",
                    "execution_time": execution_time,
                    "output": result.stdout[-1000:] if result.stdout else ""  # Last 1000 chars
                }
                return True
            else:
                self.logger.error(f"Phase {phase} failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                self.results[phase] = {
                    "status": "failed",
                    "execution_time": execution_time,
                    "error": result.stderr,
                    "output": result.stdout
                }
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Phase {phase} timed out after 2 hours")
            self.results[phase] = {"status": "timeout", "execution_time": 7200}
            return False
        except Exception as e:
            self.logger.error(f"Phase {phase} failed with exception: {e}")
            self.results[phase] = {"status": "error", "error": str(e)}
            return False
    
    def _build_phase_command(self, phase: str, script_name: str) -> List[str]:
        """Build command line arguments for each phase"""
        base_cmd = ["python", script_name]
        
        if phase == "01":  # Data Pipeline
            return base_cmd + [
                "--oid-path", self.config.oid_data_path,
                "--synthetic-path", self.config.synthetic_data_path,
                "--output-path", self.config.export_path
            ]
        
        elif phase == "02":  # BLIP-2 Captioning
            return base_cmd + [
                "--dataset-path", self.config.export_path,
                "--batch-size", "4"
            ]
        
        elif phase == "03":  # YOLO Segmentation
            return base_cmd + [
                "--dataset-path", self.config.export_path,
                "--model-size", "n"
            ]
        
        elif phase == "04":  # Semiotic Extraction
            return base_cmd + [
                "--dataset-path", self.config.export_path
            ]
        
        elif phase == "05":  # Full Fine-tuning Data Prep
            return base_cmd + [
                "--dataset-path", self.config.export_path,
                "--output-dir", f"{self.config.model_output_path}/training_data",
                "--min-semiotic-score", str(self.config.min_semiotic_score),
                "--max-samples", str(self.config.max_samples_per_class),
                "--train-split", str(self.config.train_split),
                "--val-split", str(self.config.val_split)
            ]
        
        elif phase == "06":  # Full Fine-tuning
            return base_cmd + [
                "--training-data", f"{self.config.model_output_path}/training_data",
                "--output-dir", self.config.model_output_path,
                "--base-model", self.config.base_model,
                "--epochs", str(self.config.num_train_epochs),
                "--learning-rate", str(self.config.learning_rate),
                "--batch-size", str(self.config.batch_size),
                "--gradient-accumulation-steps", str(self.config.gradient_accumulation_steps),
                "--mixed-precision", self.config.mixed_precision
            ]
        
        elif phase == "08":  # Full Model Inference
            return base_cmd + [
                "--model-path", self.config.model_output_path,
                "--output-dir", self.config.inference_output_dir,
                "--num-samples", str(self.config.num_inference_samples),
                "--steps", str(self.config.inference_steps),
                "--guidance-scale", str(self.config.guidance_scale)
            ]
        
        return base_cmd
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete full fine-tuning pipeline"""
        self.logger.info("Starting Full Fine-tuning Pipeline")
        self.logger.info(f"Phases to run: {', '.join(self.config.phases_to_run)}")
        
        success_count = 0
        total_phases = len(self.config.phases_to_run)
        
        for phase in self.config.phases_to_run:
            if phase not in self.phase_scripts:
                self.logger.warning(f"Unknown phase: {phase}")
                continue
                
            success = self.run_phase(phase)
            if success:
                success_count += 1
            else:
                self.logger.error(f"Phase {phase} failed. Consider running with --continue-on-error")
                break
        
        # Generate final report
        self._generate_pipeline_report()
        
        total_time = time.time() - self.start_time
        self.logger.info(f"Pipeline completed: {success_count}/{total_phases} phases successful")
        self.logger.info(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        return success_count == total_phases
    
    def run_partial_pipeline(self, start_phase: str, end_phase: Optional[str] = None) -> bool:
        """Run a partial pipeline from start_phase to end_phase"""
        all_phases = list(self.phase_scripts.keys())
        
        if start_phase not in all_phases:
            self.logger.error(f"Invalid start phase: {start_phase}")
            return False
        
        start_idx = all_phases.index(start_phase)
        end_idx = len(all_phases) if end_phase is None else all_phases.index(end_phase) + 1
        
        phases_to_run = all_phases[start_idx:end_idx]
        self.config.phases_to_run = phases_to_run
        
        return self.run_complete_pipeline()
    
    def _generate_pipeline_report(self):
        """Generate a comprehensive pipeline execution report"""
        report_path = Path("logs") / "pipeline_report.json"
        
        report = {
            "pipeline_type": "full_fine_tuning",
            "execution_summary": {
                "start_time": self.start_time,
                "end_time": time.time(),
                "total_duration": time.time() - self.start_time,
                "phases_attempted": len(self.results),
                "phases_successful": len([r for r in self.results.values() if r.get("status") == "success"]),
                "phases_failed": len([r for r in self.results.values() if r.get("status") != "success"])
            },
            "configuration": {
                "model_output_path": self.config.model_output_path,
                "base_model": self.config.base_model,
                "training_epochs": self.config.num_train_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "min_semiotic_score": self.config.min_semiotic_score
            },
            "phase_results": self.results,
            "final_outputs": {
                "model_path": self.config.model_output_path,
                "inference_output": self.config.inference_output_dir,
                "training_data": f"{self.config.model_output_path}/training_data"
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Pipeline report saved to: {report_path}")
    
    def setup_fiftyone(self):
        """Setup FiftyOne for dataset visualization"""
        if self.config.enable_fiftyone_app:
            try:
                # Launch FiftyOne app
                subprocess.Popen(["python", "99_fiftyone_setup.py"], cwd=script_dir)
                self.logger.info("FiftyOne app launched for dataset visualization")
            except Exception as e:
                self.logger.warning(f"Failed to launch FiftyOne app: {e}")

def main():
    parser = argparse.ArgumentParser(description="Full Fine-tuning Pipeline Orchestrator")
    
    # Pipeline control
    parser.add_argument("--phases", type=str, nargs="+", 
                        default=["01", "02", "03", "04", "05", "06", "08"],
                        help="Phases to run")
    parser.add_argument("--skip-phases", type=str, nargs="+", default=[],
                        help="Phases to skip")
    parser.add_argument("--start-from", type=str,
                        help="Start pipeline from specific phase")
    parser.add_argument("--end-at", type=str,
                        help="End pipeline at specific phase")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue pipeline even if a phase fails")
    
    # Data configuration
    parser.add_argument("--oid-data-path", type=str, default="data/oid_urban",
                        help="Path to OID urban data")
    parser.add_argument("--synthetic-data-path", type=str, default="data/imaginary_synthetic",
                        help="Path to synthetic data")
    parser.add_argument("--export-path", type=str, default="data/export",
                        help="Path for unified dataset export")
    
    # Model configuration
    parser.add_argument("--model-output-path", type=str, default="models/flux_full_finetuned",
                        help="Output path for fine-tuned model")
    parser.add_argument("--base-model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Base Flux model to fine-tune")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    
    # Data filtering
    parser.add_argument("--min-semiotic-score", type=float, default=0.5,
                        help="Minimum semiotic score for training data")
    parser.add_argument("--max-samples-per-class", type=int, default=1000,
                        help="Maximum samples per class")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Training data split ratio")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation data split ratio")
    
    # Inference configuration
    parser.add_argument("--inference-output-dir", type=str, default="outputs/flux_full_inference",
                        help="Output directory for inference results")
    parser.add_argument("--inference-samples", type=int, default=4,
                        help="Number of samples to generate during inference")
    parser.add_argument("--inference-steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale for inference")
    
    # Additional options
    parser.add_argument("--enable-fiftyone", action="store_true",
                        help="Enable FiftyOne app for dataset visualization")
    parser.add_argument("--cleanup", action="store_true",
                        help="Cleanup temporary files after completion")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        oid_data_path=args.oid_data_path,
        synthetic_data_path=args.synthetic_data_path,
        export_path=args.export_path,
        model_output_path=args.model_output_path,
        base_model=args.base_model,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        min_semiotic_score=args.min_semiotic_score,
        max_samples_per_class=args.max_samples_per_class,
        train_split=args.train_split,
        val_split=args.val_split,
        inference_output_dir=args.inference_output_dir,
        num_inference_samples=args.inference_samples,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        skip_phases=args.skip_phases,
        phases_to_run=args.phases,
        enable_fiftyone_app=args.enable_fiftyone,
        cleanup_temp_files=args.cleanup
    )
    
    # Initialize orchestrator
    orchestrator = FullFinetuningOrchestrator(config)
    
    # Setup FiftyOne if requested
    if args.enable_fiftyone:
        orchestrator.setup_fiftyone()
    
    # Run pipeline
    if args.start_from:
        success = orchestrator.run_partial_pipeline(args.start_from, args.end_at)
    else:
        success = orchestrator.run_complete_pipeline()
    
    if success:
        print("\n✅ Full fine-tuning pipeline completed successfully!")
        print(f"📁 Fine-tuned model saved to: {config.model_output_path}")
        print(f"🖼️  Inference results saved to: {config.inference_output_dir}")
        print(f"📊 Pipeline report: logs/pipeline_report.json")
    else:
        print("\n❌ Pipeline completed with errors. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()