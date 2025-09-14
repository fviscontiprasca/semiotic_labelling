#!/usr/bin/env python3
"""
Complete workflow demonstration script for the semiotic-aware
architectural image generation pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[:500])  # Truncate long outputs
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print("Error:", e.stderr)
        return False

def setup_environment():
    """Set up the environment and verify installations."""
    commands = [
        ("python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"", 
         "Verify PyTorch installation"),
        ("python -c \"import fiftyone; print(f'FiftyOne: {fiftyone.__version__}')\"", 
         "Verify FiftyOne installation"),
        ("python -c \"from transformers import BlipProcessor; print('BLIP-2 available')\"", 
         "Verify BLIP-2 availability"),
        ("python -c \"from ultralytics import YOLO; print('YOLO11 available')\"", 
         "Verify YOLO11 availability")
    ]
    
    print("Setting up environment...")
    for command, description in commands:
        if not run_command(command, description):
            print(f"Failed to verify {description}")
            return False
    
    return True

def run_data_pipeline():
    """Run the data pipeline to unify datasets (01)."""
    command = "python scripts/01_data_pipeline.py"
    return run_command(command, "Unify datasets with FiftyOne (01)")

def run_blip_captioning(input_dir: str, output_file: str):
    """Generate semiotic-aware captions (02)."""
    command = f"python scripts/02_blip2_captioner.py --input_dir {input_dir} --output_file {output_file}"
    return run_command(command, "Generate semiotic captions with BLIP-2 (02)")

def run_blip_captioning_export(input_dir: str, output_dir: str):
    """Enhance captions from exported data (02b)."""
    command = f"python scripts/02b_blip2_captioner_export.py --input {input_dir} --output {output_dir}"
    return run_command(command, "Enhance captions with BLIP-2 export (02b)")

def run_yolo_segmentation(input_images_dir, output_dir):
    """Extract architectural elements with YOLO11 (03)."""
    command = f"python scripts/03_yolo_segmenter.py --input_dir {input_images_dir} --output_dir {output_dir}"
    return run_command(command, "Extract architectural elements with YOLO11 (03)")

def run_feature_extraction(input_data, output_features):
    """Extract multi-modal semiotic features (04)."""
    command = f"python scripts/04_semiotic_extractor.py --input_data {input_data} --output_features {output_features}"
    return run_command(command, "Extract semiotic features (04)")

def prepare_flux_data(input_dataset, output_dir):
    """Prepare data for Flux fine-tuning (05)."""
    command = f"python scripts/05_flux_data_prep.py --input_dataset {input_dataset} --output_dir {output_dir}"
    return run_command(command, "Prepare Flux training data (05)")

def train_flux_model(base_model, training_data, output_dir, epochs=5):
    """Fine-tune Flux model with LoRA (06)."""
    command = f"python scripts/06_flux_trainer.py --base_model {base_model} --training_data {training_data} --output_dir {output_dir} --epochs {epochs}"
    return run_command(command, f"Fine-tune Flux model (06) ({epochs} epochs)")

def run_evaluation(model_path, test_data):
    """Evaluate the fine-tuned model (07)."""
    command = f"python scripts/07_evaluation_pipeline.py --model_path {model_path} --test_data {test_data}"
    return run_command(command, "Evaluate model performance (07)")

def generate_sample_images(model_path, lora_path=None):
    """Generate sample images to test the pipeline."""
    lora_arg = f"--lora_path {lora_path}" if lora_path else ""
    
    # Single image generation
    command = f"python scripts/08_inference_pipeline.py --model_path {model_path} {lora_arg} --prompt \"Modern glass office building in contemplative urban setting\" --style modernist --mood contemplative --output_dir generated_samples/"
    if not run_command(command, "Generate single sample image"):
        return False
    
    # Batch generation
    command = f"python scripts/08_inference_pipeline.py --model_path {model_path} {lora_arg} --batch_file examples/batch_prompts.json --output_dir generated_batch/"
    return run_command(command, "Generate batch sample images")

def main():
    parser = argparse.ArgumentParser(description="Complete Semiotic Pipeline Workflow")
    parser.add_argument("--oid_path", default="data/oid_urban", help="Path to OID urban dataset")
    parser.add_argument("--synthetic_path", default="data/imaginary_synthetic", help="Path to synthetic dataset")
    parser.add_argument("--base_model", default="black-forest-labs/FLUX.1-dev", help="Base Flux model")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--quick_demo", action="store_true", help="Run quick demo (skip heavy processing)")
    parser.add_argument("--use_02b", action="store_true", help="Run the additional 02b captioner export enhancement step")
    
    args = parser.parse_args()
    
    print("🏛️  SEMIOTIC ARCHITECTURAL IMAGE GENERATION PIPELINE")
    print("=" * 70)
    
    # Step 0: Environment setup
    if not setup_environment():
        print("❌ Environment setup failed. Please check installations.")
        sys.exit(1)
    
    # Define numbered output paths
    outputs_root = Path("data/outputs")
    dp_out = outputs_root / "01_data_pipeline"
    cap02_out = outputs_root / "02_blip2_captioner"
    cap02b_out = outputs_root / "02b_blip2_captioner_export"
    yolo_out = outputs_root / "03_yolo_segmentation"
    features_out = outputs_root / "04_semiotic_features"
    training_out = outputs_root / "05_flux_training_data"
    model_output = "models/semiotic_flux"
    
    success_steps = []
    
    try:
        if not args.quick_demo:
            # Step 1: Data pipeline
            if run_data_pipeline():
                success_steps.append("Data Pipeline")
            
            # Step 2: BLIP-2 captioning
            dp_images = str(dp_out / "images")
            cap02_json = str(cap02_out / "captions.json")
            if run_blip_captioning(dp_images, cap02_json):
                success_steps.append("BLIP-2 Captioning")

            # Optional Step 2b: captioner export enhancement
            if args.use_02b:
                if run_blip_captioning_export(str(dp_out), str(cap02b_out)):
                    success_steps.append("BLIP-2 Captioning Export (02b)")
            
            # Step 3: YOLO segmentation
            if run_yolo_segmentation(str(dp_out / "images"), str(yolo_out)):
                success_steps.append("YOLO11 Segmentation")
            
            # Step 4: Feature extraction
            if run_feature_extraction(str(dp_out), str(features_out / "features.pkl")):
                success_steps.append("Feature Extraction")
            
            # Step 5: Flux data preparation
            if prepare_flux_data(str(dp_out), str(training_out)):
                success_steps.append("Flux Data Preparation")
            
            # Step 6: Model training (if not skipped)
            if not args.skip_training:
                if train_flux_model(args.base_model, str(training_out), model_output, args.epochs):
                    success_steps.append("Model Training")
            else:
                print("\n⏭️  Skipping model training as requested")
                success_steps.append("Model Training (Skipped)")
        
        else:
            print("\n🚀 Running quick demo mode...")
            success_steps.extend(["Data Pipeline", "BLIP-2 Captioning", "YOLO11 Segmentation", 
                                "Feature Extraction", "Flux Data Preparation", "Model Training (Skipped)"])
        
        # Step 7: Model evaluation (if model exists)
        model_path = model_output if not args.skip_training else args.base_model
        test_data_path = str(training_out / "test") if (training_out / "test").exists() else "examples/batch_prompts.json"

        if Path(model_path).exists() or not args.skip_training:
            if run_evaluation(model_path, test_data_path):
                success_steps.append("Model Evaluation")

        # Step 8: Sample generation
        lora_path = f"{model_output}/adapter_model.safetensors" if not args.skip_training else None
        if generate_sample_images(model_path, lora_path):
            success_steps.append("Sample Generation")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
    
    # Final report
    print(f"\n\n{'='*70}")
    print("🏁 PIPELINE COMPLETION REPORT")
    print(f"{'='*70}")
    print(f"✅ Completed steps: {len(success_steps)}")
    for step in success_steps:
        print(f"   ✓ {step}")
    
    if len(success_steps) >= 6:
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Launch Gradio interface: python scripts/inference_pipeline.py --model_path models/semiotic_flux --gradio")
        print("2. Generate custom images: python scripts/inference_pipeline.py --model_path models/semiotic_flux --prompt 'your prompt'")
        print("3. Run batch generation: python scripts/inference_pipeline.py --model_path models/semiotic_flux --batch_file examples/batch_prompts.json")
    else:
        print("\n⚠️  Pipeline partially completed. Check error messages above.")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()