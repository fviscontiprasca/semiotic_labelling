#!/usr/bin/env python3
"""
Environment Validation Script for Semiotic Labelling Pipeline
=============================================================

This script validates the environment setup for both LoRA and Full Fine-tuning
pipeline variants, checking dependencies, hardware compatibility, and providing
setup recommendations.

Usage:
    python check_environment.py
    python check_environment.py --full-finetune  # Check full fine-tuning requirements
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_package(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a package is installed and importable"""
    try:
        if import_name is None:
            import_name = package_name
        
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✓ {package_name} ({version})"
    except ImportError:
        return False, f"✗ {package_name} (not found)"

def check_gpu_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """Check GPU availability and CUDA support"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_memory': [],
        'cuda_version': None
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['gpu_count'] = torch.cuda.device_count()
        gpu_info['cuda_version'] = getattr(torch, 'version', {}).get('cuda', 'unknown')
        
        if gpu_info['cuda_available']:
            for i in range(gpu_info['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                gpu_info['gpu_memory'].append({
                    'device': i,
                    'name': props.name,
                    'memory_gb': memory_gb
                })
            
            memory_str = ", ".join([f"GPU{info['device']}: {info['name']} ({info['memory_gb']:.1f}GB)" 
                                  for info in gpu_info['gpu_memory']])
            return True, f"✓ CUDA {gpu_info['cuda_version']} - {memory_str}", gpu_info
        else:
            return False, "✗ CUDA not available", gpu_info
            
    except ImportError:
        return False, "✗ PyTorch not installed", gpu_info

def get_memory_recommendations(gpu_info: Dict[str, Any], full_finetune: bool = False) -> List[str]:
    """Provide memory usage recommendations"""
    recommendations = []
    
    if not gpu_info['cuda_available']:
        recommendations.append("⚠️  No GPU detected. CPU-only training will be very slow.")
        return recommendations
    
    max_memory = max([info['memory_gb'] for info in gpu_info['gpu_memory']], default=0)
    
    if full_finetune:
        if max_memory >= 24:
            recommendations.append("✓ Excellent GPU memory for full fine-tuning")
        elif max_memory >= 16:
            recommendations.append("✓ Good GPU memory for full fine-tuning with optimization")
        elif max_memory >= 12:
            recommendations.append("⚠️  Limited GPU memory. Use gradient accumulation and CPU offload")
        else:
            recommendations.append("✗ Insufficient GPU memory for full fine-tuning (16GB+ recommended)")
    else:
        if max_memory >= 12:
            recommendations.append("✓ Excellent GPU memory for LoRA training")
        elif max_memory >= 8:
            recommendations.append("✓ Good GPU memory for LoRA training")
        elif max_memory >= 6:
            recommendations.append("⚠️  Limited GPU memory. Use smaller batch sizes")
        else:
            recommendations.append("✗ Insufficient GPU memory for LoRA training (8GB+ recommended)")
    
    return recommendations

def check_core_dependencies() -> List[Tuple[bool, str]]:
    """Check core ML dependencies"""
    dependencies = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('accelerate', 'accelerate'),
        ('datasets', 'datasets'),
        ('PIL', 'PIL'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
    ]
    
    results = []
    for package, import_name in dependencies:
        results.append(check_package(package, import_name))
    
    return results

def check_pipeline_dependencies() -> List[Tuple[bool, str]]:
    """Check pipeline-specific dependencies"""
    dependencies = [
        ('sentence-transformers', 'sentence_transformers'),
        ('fiftyone', 'fiftyone'),
        ('opencv-python', 'cv2'),
        ('tqdm', 'tqdm'),
        ('jsonlines', 'jsonlines'),
    ]
    
    results = []
    for package, import_name in dependencies:
        results.append(check_package(package, import_name))
    
    return results

def check_full_finetune_dependencies() -> List[Tuple[bool, str]]:
    """Check full fine-tuning specific dependencies"""
    dependencies = [
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
        ('psutil', 'psutil'),
    ]
    
    results = []
    for package, import_name in dependencies:
        results.append(check_package(package, import_name))
    
    return results

def check_optional_dependencies() -> List[Tuple[bool, str]]:
    """Check optional performance dependencies"""
    dependencies = [
        ('wandb', 'wandb'),
        ('xformers', 'xformers'),
    ]
    
    results = []
    for package, import_name in dependencies:
        results.append(check_package(package, import_name))
    
    return results

def check_segment_anything() -> Tuple[bool, str]:
    """Check Segment Anything Model availability"""
    try:
        from segment_anything import sam_model_registry
        return True, "✓ Segment Anything Model"
    except ImportError:
        return False, "✗ Segment Anything Model (install from GitHub)"

def print_section(title: str, items: List[Tuple[bool, str]]):
    """Print a formatted section of check results"""
    print(f"\n{title}")
    print("=" * len(title))
    for success, message in items:
        print(f"  {message}")

def main():
    parser = argparse.ArgumentParser(description="Validate environment for semiotic labelling pipeline")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Check requirements for full fine-tuning pipeline")
    args = parser.parse_args()
    
    print("🔍 Semiotic Labelling Pipeline - Environment Check")
    print("=" * 50)
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    print(f"\nPython Version: {python_msg}")
    
    # Check GPU availability
    gpu_ok, gpu_msg, gpu_info = check_gpu_availability()
    print(f"GPU Support: {gpu_msg}")
    
    # Memory recommendations
    recommendations = get_memory_recommendations(gpu_info, args.full_finetune)
    if recommendations:
        print("\nMemory Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    
    # Check core dependencies
    core_deps = check_core_dependencies()
    print_section("Core ML Dependencies", core_deps)
    
    # Check pipeline dependencies
    pipeline_deps = check_pipeline_dependencies()
    print_section("Pipeline Dependencies", pipeline_deps)
    
    # Check SAM
    sam_ok, sam_msg = check_segment_anything()
    print_section("Computer Vision", [(sam_ok, sam_msg)])
    
    # Check full fine-tuning dependencies if requested
    if args.full_finetune:
        full_deps = check_full_finetune_dependencies()
        print_section("Full Fine-tuning Dependencies", full_deps)
    
    # Check optional dependencies
    optional_deps = check_optional_dependencies()
    print_section("Optional Dependencies", optional_deps)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_core_ok = all(ok for ok, _ in core_deps)
    all_pipeline_ok = all(ok for ok, _ in pipeline_deps)
    
    if args.full_finetune:
        full_deps_ok = all(ok for ok, _ in check_full_finetune_dependencies())
        if all_core_ok and all_pipeline_ok and sam_ok and full_deps_ok and gpu_ok:
            print("✅ Environment ready for FULL FINE-TUNING pipeline!")
        else:
            print("❌ Environment needs setup for full fine-tuning pipeline")
            print("\nInstallation command:")
            print("pip install -r requirements.txt")
    else:
        if all_core_ok and all_pipeline_ok and sam_ok:
            print("✅ Environment ready for LORA pipeline!")
            if gpu_ok:
                print("🚀 GPU acceleration available")
            else:
                print("⚠️  CPU-only mode (training will be slow)")
        else:
            print("❌ Environment needs setup for LoRA pipeline")
            print("\nInstallation command:")
            print("pip install -r requirements.txt")
    
    # Pipeline selection guidance
    print("\n📋 Pipeline Selection:")
    print("  • Standard LoRA: python run_pipeline.py")
    print("  • Full Fine-tuning: python run_flux_full_pipeline.py")
    
    if not gpu_ok:
        print("\n⚠️  Consider using a cloud GPU service (Colab, Paperspace, etc.) for training")

if __name__ == "__main__":
    main()