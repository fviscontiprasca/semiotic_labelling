
# Semiotic-Aware Urban Image Generation Pipeline

---

**This project is part of the Master Thesis "Semiocity" for the Master in Advanced Computation for Architecture and Design (MaCAD) at the Institute of Advanced Architecture of Catalonia (IaaC).**

**Thesis Title:** Semiocity: Generating Semiotic-Enriched 3D Models from Text Inputs

**Authors:**
- Francesco Visconti Prasca
- Jose Lazokafatty ([DrArc](https://github.com/DrArc))
- Paul Suarez Fuenmayor

**Adviser:** David Andres Leon (MaCAD Director)

The Semiocity thesis aims to develop a pipeline for generating 3D models enriched with semiotic information, starting from text prompts. This repository contains the image-based pipeline and foundational work for the broader 3D generative system.

---


A comprehensive, modular pipeline for generating semiotic-aware architectural images using fine-tuned Flux.1d diffusion models, advanced prompt engineering, and multi-modal conditioning.

## Data Sources and Acknowledgements

This project leverages two primary datasets:

- **OpenImages v7 (OIDv7) Urban Classes**: A large-scale, publicly available dataset of real-world urban and architectural images. OIDv7 provides the foundation for real image analysis, segmentation, and semiotic feature extraction. See [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html) for more information and licensing.

- **Synthetic Imaginary Cities Dataset**: A unique collection of synthetic architectural images depicting imaginary cities. This dataset was authored and curated by **Paul Suarez Fuenmayor**, who generated the images using the Flux 1 Dev model in ComfyUI. The generation workflow, including prompt engineering and model configuration, is fully documented in `data/imaginary_synthetic/Imaginary Cities Generation Workflow.json`. This synthetic dataset enables advanced experimentation in semiotic-aware generative modeling and style transfer.

We gratefully acknowledge the contributions of both the OpenImages project and Paul Suarez Fuenmayor for enabling this research.


## Pipeline Overview


This project implements a multi-phase pipeline for semiotic-aware urban image generation. The pipeline processes both real (OIDv7) and synthetic (Imaginary Cities) urban image datasets, extracts rich semantic and architectural features, and enables advanced generative modeling and evaluation. Each phase is encapsulated in a dedicated script, allowing for flexible execution, debugging, and extension.

### About the Synthetic Dataset

The synthetic dataset, "Imaginary Cities," was created by Paul Suarez Fuenmayor using the Flux 1 Dev model in ComfyUI. The images were generated through a custom workflow that leverages prompt engineering, CLIP-based conditioning, and advanced diffusion techniques. The full workflow, including all nodes and parameters, is available in [`data/imaginary_synthetic/Imaginary Cities Generation Workflow.json`]. This dataset is intended for research and creative exploration in the field of semiotic-aware generative design.

**Pipeline Phases:**
1. Data Preparation
2. Caption Generation (BLIP-2)
3. Caption Export (BLIP-2 Export)
4. Segmentation (SAM)
5. Semiotic Feature Extraction
6. Flux Data Preparation
7. Flux Training
8. Evaluation
9. Inference

## Architecture Diagram

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ OID Dataset   │    │ Synthetic Data│    │ FiftyOne      │
│ Urban Classes │───▶│ Imaginary City│───▶│ Integration   │
└───────────────┘    └───────────────┘    └───────────────┘
                                              │
                  ┌───────────────────────────┘
                  ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   BLIP-2      │    │   SAM         │    │ Semiotic      │
│ Captioning    │───▶│ Segmentation  │───▶│ Feature       │
│               │    │               │    │ Extraction    │
└───────────────┘    └───────────────┘    └───────────────┘
                                              │
                  ┌───────────────────────────┘
                  ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Flux.1d     │    │ Evaluation    │    │ Inference     │
│ Fine-tuning   │───▶│ Pipeline      │───▶│ Pipeline      │
│ (LoRA)        │    │               │    │ (Gradio UI)   │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Requirements

To run the pipeline, you need:
- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- Transformers (HuggingFace)
- FiftyOne
- Diffusers, PEFT, Accelerate (for Flux training/inference)
- scikit-learn, pandas, numpy, Pillow, tqdm, sentence-transformers
- BLIP-2, SAM, and Flux model weights (see scripts for download locations)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage: Full Pipeline

To run the entire pipeline in sequence:

```bash
python run_pipeline.py --base_path .
```

This will execute all phases, printing progress and output for each step. You can also run each script individually as described below.

---

## 01_data_pipeline.py
**Scope:** Combines OpenImages v7 urban classes and synthetic datasets into a unified FiftyOne dataset for downstream processing.

**How it works:**
- Loads urban class definitions and image metadata
- Merges real and synthetic images, ensuring class consistency
- Outputs a standardized dataset for captioning and segmentation
- Handles dataset deduplication and cleaning

**Requirements:** FiftyOne, pandas, OpenImages/Synthetic data

**Usage Example:**
```bash
python scripts/01_data_pipeline.py --base_path .
```

---

## 02_blip2_captioner.py
**Scope:** Generates rich, semiotic-aware captions for each image using BLIP-2. Prompts are designed to elicit architectural, spatial, and cultural details.

**How it works:**
- Loads images from the unified dataset
- Uses BLIP-2 with custom prompts for style, mood, materials, etc.
- Stores captions and metadata for each image
- Supports GPU acceleration

**Requirements:** transformers, BLIP-2 weights, torch, PIL

**Usage Example:**
```bash
python scripts/02_blip2_captioner.py --input_dir data/outputs/01_data_pipeline --output_file data/outputs/blip2_captions.json
```

---

## 02b_blip2_captioner_export.py
**Scope:** Processes exported data pipeline files and generates enhanced semiotic-aware captions, with advanced GPU and precision controls.

**How it works:**
- Loads exported image data
- Runs BLIP-2 captioning with flexible device/dtype options
- Outputs captions in a format ready for segmentation and feature extraction

**Requirements:** transformers, BLIP-2 weights, torch, PIL

**Usage Example:**
```bash
python scripts/02b_blip2_captioner_export.py --input data/outputs/01_data_pipeline --output data/outputs/blip2_captioner_export
```

---

## 03_sam_segmenter.py
**Scope:** Segments architectural elements in images using the Segment Anything Model (SAM), replacing YOLO for more precise and flexible segmentation.

**How it works:**
- Loads images and runs SAM segmentation
- Extracts masks and bounding boxes for architectural elements
- Outputs segmentation data for use in semiotic feature extraction
- Supports multiple SAM model variants and checkpoints

**Requirements:** segment-anything, torch, cv2, PIL, numpy

**Usage Example:**
```bash
python scripts/03_sam_segmenter.py --input_dir data/outputs/01_data_pipeline --output_dir data/outputs/sam_segments
```

---

## 04_semiotic_extractor.py
**Scope:** Extracts comprehensive semiotic features by combining BLIP-2 captions, SAM segmentation, and architectural analysis. Produces embeddings and metadata for Flux training.

**How it works:**
- Loads captions and segmentation data
- Computes CLIP and sentence-transformer embeddings
- Analyzes style, mood, materials, spatial hierarchy, and more
- Outputs feature-rich JSON or dataset for Flux

**Requirements:** transformers, sentence-transformers, scikit-learn, torch, numpy, PIL

**Usage Example:**
```bash
python scripts/04_semiotic_extractor.py --input_data data/outputs/blip2_captioner_export --segmentation_dir data/outputs/sam_segments --output_features data/outputs/semiotic_features.json
```

---

## 05_flux_data_prep.py
**Scope:** Prepares the dataset for Flux.1d fine-tuning, formatting images, captions, and metadata into the required structure.

**How it works:**
- Loads semiotic features and images
- Applies prompt templates for enhanced captioning
- Organizes data into images, captions, and metadata folders
- Ensures compatibility with Flux training scripts

**Requirements:** pandas, numpy, PIL, fiftyone, datasets

**Usage Example:**
```bash
python scripts/05_flux_data_prep.py --input_dataset data/outputs/semiotic_features.json --output_dir data/outputs/flux_training_data
```

---

## 06_flux_trainer.py
**Scope:** Fine-tunes the Flux.1d model using LoRA and the prepared semiotic-aware dataset. Supports advanced training options and experiment tracking.

**How it works:**
- Loads training data and configuration
- Sets up LoRA adapters and training loop
- Supports mixed precision, distributed training, and experiment tracking (wandb)
- Saves trained model checkpoints

**Requirements:** torch, diffusers, peft, accelerate, transformers, wandb (optional)

**Usage Example:**
```bash
python scripts/06_flux_trainer.py --base_model models/flux1d --training_data data/outputs/flux_training_data --output_dir models/flux_finetuned --epochs 10
```

---

## 07_evaluation_pipeline.py
**Scope:** Evaluates generated images for semiotic coherence, architectural accuracy, and visual quality. Supports both quantitative and qualitative metrics.

**How it works:**
- Loads generated images and reference data
- Computes similarity metrics, style/mood accuracy, and other scores
- Supports custom evaluation modules and fallback logic
- Outputs detailed evaluation reports

**Requirements:** torch, numpy, pandas, PIL, scikit-learn, cv2 (optional)

**Usage Example:**
```bash
python scripts/07_evaluation_pipeline.py --model_path models/flux_finetuned --test_data data/outputs/flux_test_data
```

---

## 08_inference_pipeline.py
**Scope:** Generates new images using the fine-tuned Flux.1d model, with advanced prompt engineering and optional Gradio interface for interactive use.

**How it works:**
- Loads trained model and LoRA adapters
- Accepts prompts (single or batch)
- Generates images and saves outputs
- Optionally launches a Gradio web UI for interactive generation

**Requirements:** torch, diffusers, peft, gradio (optional), PIL, numpy

**Usage Example:**
```bash
python scripts/08_inference_pipeline.py --model_path models/flux_finetuned --prompt "Modern glass office building in contemplative urban setting" --output_dir generated_samples/
```

---

## 99_fiftyone_setup.py
**Scope:** Utility script to quickly set up and launch a FiftyOne dataset/app for visualizing and exploring your image data.

**How it works:**
- Loads or creates a FiftyOne dataset
- Adds images from a specified directory
- Launches the FiftyOne web app for interactive exploration

**Requirements:** fiftyone

**Usage Example:**
```bash
python scripts/99_fiftyone_setup.py
```

---

## Utility Scripts

### 00_check_classes.py
**Scope:** Checks your custom class list (`my_classes_final.txt`) against the actual available classes in OpenImages. Suggests similar matches for missing classes.

**Usage Example:**
```bash
python scripts/utilities/00_check_classes.py
```

### 00_find_urban_classes.py
**Scope:** Searches for additional urban/architectural classes in OpenImages, reporting which are present or missing. Useful for expanding or debugging your class list.

**Usage Example:**
```bash
python scripts/utilities/00_find_urban_classes.py
```

---

## Components

### 1. Data Pipeline (`scripts/data_pipeline.py`)
- **Purpose**: Unified dataset creation and management
- **Features**:
  - Loads OpenImages v7 urban architectural classes
  - Integrates imaginary synthetic images with CSV mappings
  - Extracts semiotic features (style, mood, materials)
  - Exports training-ready datasets
- **Usage**: `python data_pipeline.py --config config.yaml`

### 2. BLIP-2 Captioning (`scripts/blip2_captioner.py`)
- **Purpose**: Generate semiotic-aware captions for architectural images
- **Features**:
  - Style-aware caption generation (modernist, brutalist, etc.)
  - Mood detection and description (contemplative, vibrant, etc.)
  - Material and temporal context integration
  - Comprehensive architectural analysis
- **Usage**: `python blip2_captioner.py --input_dir images/ --output_file captions.json`

### 3. YOLO11 Segmentation (`scripts/yolo_segmenter.py`)
- **Purpose**: Extract architectural elements and spatial relationships
- **Features**:
  - Urban element detection (buildings, towers, houses, etc.)
  - Spatial composition analysis
  - Semiotic spatial interpretation
  - FiftyOne integration for visualization
- **Usage**: `python yolo_segmenter.py --dataset_name urban_dataset --export_dir segments/`

### 4. Semiotic Feature Extraction (`scripts/semiotic_extractor.py`)
- **Purpose**: Multi-modal feature extraction and embedding creation
- **Features**:
  - CLIP-based visual embeddings
  - Sentence transformer text embeddings
  - Architectural style classification
  - Unified embedding space for conditioning
- **Usage**: `python semiotic_extractor.py --input_data data.json --output_features features.pkl`

### 5. Flux Training Data Preparation (`scripts/flux_data_prep.py`)
- **Purpose**: Format dataset for Flux.1d fine-tuning
- **Features**:
  - Enhanced caption generation with templates
  - Quality filtering and validation
  - Train/validation/test splits
  - Metadata preservation
- **Usage**: `python flux_data_prep.py --input_dataset combined_data --output_dir training_data/`

### 6. Flux.1d Fine-tuning (`scripts/flux_trainer.py`)
- **Purpose**: LoRA fine-tuning of Flux.1d for semiotic conditioning
- **Features**:
  - Efficient LoRA parameter training
  - Semiotic token integration
  - Distributed training support
  - Automatic checkpointing and logging
- **Usage**: `python flux_trainer.py --config training_config.yaml --output_dir models/`

### 7. Evaluation Pipeline (`scripts/evaluation_pipeline.py`)
- **Purpose**: Comprehensive evaluation of generated images
- **Features**:
  - Semiotic coherence assessment
  - Architectural style accuracy
  - Visual quality metrics (CLIP score, etc.)
  - Batch evaluation with detailed reports
- **Usage**: `python evaluation_pipeline.py --model_path model/ --test_data test.json`

### 8. Inference Pipeline (`scripts/inference_pipeline.py`)
- **Purpose**: End-to-end image generation with advanced interfaces
- **Features**:
  - Advanced prompt engineering with semiotic tokens
  - Gradio web interface for interactive generation
  - Batch processing capabilities
  - Real-time evaluation integration
- **Usage**: 
  - CLI: `python inference_pipeline.py --model_path model/ --prompt "Modern glass office building"`
  - Web UI: `python inference_pipeline.py --model_path model/ --gradio`

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository_url>
cd semiotic_labelling
```

2. **Create virtual environment**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Models

The pipeline requires several pre-trained models:

1. **Flux.1d Base Model**:
```bash
# Download from Hugging Face
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.1-dev
```

2. **BLIP-2 Model**:
```bash
# Downloaded automatically via transformers
```

3. **YOLO11 Model**:
```bash
# Downloaded automatically via ultralytics
```

## Quick Start

### 1. Prepare Data
```bash
python scripts/data_pipeline.py --oid_path data/oid_urban/ --synthetic_path data/imaginary_synthetic/ --output_dir data/unified/
```

### 2. Generate Captions
```bash
python scripts/blip2_captioner.py --input_dir data/unified/images/ --output_file data/unified/captions.json
```

### 3. Extract Features
```bash
python scripts/semiotic_extractor.py --input_data data/unified/ --output_features data/features.pkl
```

### 4. Prepare Training Data
```bash
python scripts/flux_data_prep.py --input_dataset data/unified/ --output_dir training_data/
```

### 5. Fine-tune Model
```bash
python scripts/flux_trainer.py --base_model black-forest-labs/FLUX.1-dev --training_data training_data/ --output_dir models/semiotic_flux/
```

### 6. Generate Images
```bash
# Command line
python scripts/inference_pipeline.py --model_path models/semiotic_flux/ --prompt "Brutalist concrete housing complex" --style brutalist --mood dramatic

# Web interface
python scripts/inference_pipeline.py --model_path models/semiotic_flux/ --gradio
```

## Configuration

### Semiotic Tokens

The pipeline uses specialized tokens for architectural conditioning:

**Architectural Styles**:
- `<modernist>`: Clean geometric lines, minimal ornamentation
- `<brutalist>`: Raw concrete, massive geometric forms
- `<postmodern>`: Eclectic mix, colorful facades
- `<minimalist>`: Pure geometric forms, essential elements
- `<baroque>`: Ornate decoration, curved forms
- `<gothic>`: Pointed arches, vertical emphasis
- `<industrial>`: Exposed structural elements
- `<art_deco>`: Streamlined forms, geometric patterns

**Urban Moods**:
- `<contemplative>`: Peaceful atmosphere, soft lighting
- `<vibrant>`: Energetic colors, dynamic composition
- `<tense>`: Dramatic contrasts, sharp angles
- `<serene>`: Harmonious proportions, balanced
- `<dramatic>`: Bold contrasts, striking lighting
- `<peaceful>`: Gentle transitions, natural integration
- `<energetic>`: Dynamic lines, bright colors
- `<melancholic>`: Muted colors, weathered surfaces

**Materials**: `<concrete>`, `<glass>`, `<steel>`, `<stone>`, `<wood>`, `<brick>`

**Time Periods**: `<dawn>`, `<morning>`, `<afternoon>`, `<dusk>`, `<night>`, `<golden_hour>`, `<blue_hour>`

### Training Configuration

Key parameters for fine-tuning:

```yaml
# training_config.yaml
model:
  base_model: "black-forest-labs/FLUX.1-dev"
  lora_rank: 64
  lora_alpha: 64
  lora_dropout: 0.1

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 500
  mixed_precision: "fp16"

generation:
  resolution: 1024
  guidance_scale: 7.5
  num_inference_steps: 20
```

## Dataset Structure

```
data/
├── oid_urban/                    # OpenImages v7 urban classes
│   ├── images/
│   ├── annots/
│   └── metadata/
├── imaginary_synthetic/          # Synthetic architectural images
│   ├── images/
│   ├── annotations/
│   └── 02 Imaginary Cities - Mapping.csv
├── export/                       # Processed training data
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── unified/                      # Combined dataset
    ├── metadata.json
    ├── captions.json
    └── features.pkl
```

## Evaluation Metrics

The pipeline provides comprehensive evaluation:

### Semiotic Coherence
- **Style Accuracy**: Measures alignment between requested and generated architectural styles
- **Mood Accuracy**: Evaluates atmospheric and emotional consistency
- **Material Recognition**: Assesses correct material representation

### Visual Quality
- **CLIP Score**: Semantic similarity between text and image
- **Architectural Realism**: Structural and proportional accuracy
- **Technical Quality**: Resolution, sharpness, composition

### Batch Metrics
- **Consistency**: Variation across multiple generations
- **Diversity**: Creative range within style constraints
- **Error Analysis**: Common failure modes and improvements

## Performance

### Hardware Requirements

**Minimum**:
- GPU: GTX 1080 (8GB VRAM)
- RAM: 16GB
- Storage: 50GB

**Recommended**:
- GPU: RTX 4090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### Timing Benchmarks

| Operation | Time (RTX 4090) | Time (RTX 3080) |
|-----------|-----------------|-----------------|
| Data Processing | 2-3 hours | 4-6 hours |
| BLIP-2 Captioning | 1-2 hours | 3-4 hours |
| Feature Extraction | 30-60 min | 1-2 hours |
| LoRA Fine-tuning | 6-12 hours | 12-24 hours |
| Single Image Gen | 3-5 seconds | 8-12 seconds |
| Batch Evaluation | 2-5 min/100 images | 5-10 min/100 images |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training config
   - Use gradient checkpointing
   - Enable CPU offloading

2. **Model Loading Errors**:
   - Verify model paths are correct
   - Check Hugging Face authentication
   - Ensure sufficient disk space

3. **Slow Generation**:
   - Enable xformers for memory efficiency
   - Use mixed precision (fp16)
   - Optimize inference steps

4. **Poor Quality Outputs**:
   - Increase training epochs
   - Improve caption quality
   - Adjust LoRA parameters

### Performance Optimization

```python
# Enable optimizations
pipeline.enable_xformers_memory_efficient_attention()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{semiotic_architectural_generation,
  title={Semiotic-Aware Architectural Image Generation with Fine-tuned Flux.1d},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/semiotic_labelling}
}
```

## Acknowledgments

- **Black Forest Labs** for Flux.1d diffusion model
- **Salesforce** for BLIP-2 captioning model
- **Ultralytics** for YOLO11 segmentation
- **Voxel51** for FiftyOne dataset management
- **Hugging Face** for transformers and diffusers libraries

## Support

For questions and issues:
- Open an issue on GitHub
- Check the documentation
- Review existing discussions

---

**Note**: This pipeline is designed for research and educational purposes. Commercial use may require additional licensing for some model components.