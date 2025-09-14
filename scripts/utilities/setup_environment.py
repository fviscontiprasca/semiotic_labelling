#!/usr/bin/env python3
"""
Quick Setup Script for Semiotic Labelling Pipeline
==================================================

This script provides guided setup for both LoRA and Full Fine-tuning pipeline variants.

Usage:
    python setup_environment.py
    python setup_environment.py --full-finetune
    python setup_environment.py --check-only
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def install_requirements(full_finetune: bool = False) -> bool:
    """Install requirements for the selected pipeline variant"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing dependencies...")
    print(f"   Pipeline: {'Full Fine-tuning' if full_finetune else 'LoRA'}")
    
    # Install main requirements
    success = run_command("pip install -r requirements.txt", "Installing main requirements")
    
    if success:
        print("✅ Installation completed successfully!")
        print("\n🔍 Running environment check...")
        
        # Run environment check
        check_cmd = f"python check_environment.py{'--full-finetune' if full_finetune else ''}"
        run_command(check_cmd, "Environment validation")
        
        return True
    else:
        print("❌ Installation failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup environment for semiotic labelling pipeline")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Setup for full fine-tuning pipeline")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run environment check, don't install")
    
    args = parser.parse_args()
    
    print("🚀 Semiotic Labelling Pipeline - Environment Setup")
    print("=" * 52)
    
    if args.check_only:
        print("🔍 Running environment check only...")
        check_cmd = f"python check_environment.py{' --full-finetune' if args.full_finetune else ''}"
        run_command(check_cmd, "Environment validation")
        return
    
    print(f"📋 Setting up for: {'Full Fine-tuning Pipeline' if args.full_finetune else 'LoRA Pipeline'}")
    print()
    
    # Show requirements
    if args.full_finetune:
        print("📋 Full Fine-tuning Requirements:")
        print("   • 16GB+ GPU VRAM (24GB+ optimal)")
        print("   • Additional memory optimization packages")
        print("   • All LoRA pipeline dependencies")
    else:
        print("📋 LoRA Pipeline Requirements:")
        print("   • 8GB+ GPU VRAM recommended")
        print("   • Standard ML dependencies")
    
    print()
    
    # Confirm installation
    confirm = input("Proceed with installation? [y/N]: ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Install dependencies
    success = install_requirements(args.full_finetune)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next Steps:")
        print("   1. Prepare your data in data/oid_urban/ and data/imaginary_synthetic/")
        print("   2. Run the appropriate pipeline:")
        if args.full_finetune:
            print("      python run_flux_full_pipeline.py")
        else:
            print("      python run_pipeline.py")
        print("   3. Check outputs in models/ and outputs/ directories")
        
        print("\n📖 Documentation:")
        print("   • README.md - Complete pipeline documentation")
        print("   • check_environment.py - Validate your setup anytime")
    else:
        print("\n❌ Setup failed. Please check error messages above.")
        print("   💡 Try manual installation: pip install -r requirements.txt")

if __name__ == "__main__":
    main()