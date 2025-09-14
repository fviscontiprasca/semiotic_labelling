# Semiotic-Aware Architectural Image Generation Pipeline

A comprehensive pipeline for generating semiotic-aware architectural images using fine-tuned Flux.1d diffusion models with advanced prompt engineering and multi-modal conditioning.

## Overview

This pipeline combines multiple state-of-the-art AI models to create a sophisticated system for generating architectural images that are contextually and semiotically aware. The system integrates:

- **OpenImages v7** urban architectural dataset
- **Imaginary synthetic** architectural images 
- **FiftyOne** for dataset management and visualization
- **YOLO11** for architectural element segmentation
- **BLIP-2** for semiotic-aware captioning
- **Flux.1d** diffusion model with LoRA fine-tuning
- **CLIP & Sentence Transformers** for multi-modal embeddings

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OID Dataset   │    │  Synthetic Data │    │   FiftyOne      │
│  Urban Classes  │───▶│  Imaginary City │───▶│  Integration    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                    ┌─────────────────────────────────┘
                    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BLIP-2        │    │   YOLO11        │    │   Semiotic      │
│  Captioning     │───▶│  Segmentation   │───▶│  Feature        │
│                 │    │                 │    │  Extraction     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                    ┌─────────────────────────────────┘
                    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flux.1d       │    │   Evaluation    │    │   Inference     │
│  Fine-tuning    │───▶│   Pipeline      │───▶│   Pipeline      │
│  (LoRA)         │    │                 │    │   (Gradio UI)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

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