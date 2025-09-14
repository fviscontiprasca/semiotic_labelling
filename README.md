# Semiocity — Semiotic-Aware Urban Image Generation Pipeline

**Repository for the Master Thesis "Semiocity: Generating Semiotic‑Enriched 3D Models from Text Inputs"**

**Degree:** Master in Advanced Computation for Architecture and Design (MaCAD), Institute for Advanced Architecture of Catalonia (IaaC)

**Primary authors:** Francesco Visconti Prasca, Jose Lazokafatty ([DrArc](https://github.com/DrArc)), Paul Suarez Fuenmayor

**Adviser:** David Andrés León (MaCAD Director)

---

## Abstract

This repository implements a modular, reproducible pipeline for producing semiotic‑aware architectural images from textual prompts and for preparing semiotic conditioning information for downstream 3D generation. The approach combines multi‑modal feature extraction (captioning, segmentation, semantic embeddings), curated real and synthetic datasets, and fine‑tuning of diffusion models (Flux.1d) through both LoRA adapters and full model fine-tuning approaches.

The codebase supports: dataset preparation and unification, BLIP‑2 captioning, segmentation with SAM (and optional YOLO variants), multi‑modal semiotic feature extraction, Flux.1d data preparation with both LoRA and full fine‑tuning variants, evaluation, and inference (CLI and optional Gradio UI).

---

## Repository layout (high level)

```
├── data/                      # raw and processed datasets
├── models/                    # downloaded and trained checkpoints
├── scripts/                   # processing, training, evaluation, inference scripts
│   └── flux_full_finetune/    # full fine-tuning variant scripts
├── notebooks/                 # experiments and visualizations
├── docs/                      # supplementary documentation / diagrams
├── requirements.txt
├── run_pipeline.py            # LoRA-based pipeline orchestrator
├── run_flux_full_pipeline.py  # Full fine-tuning pipeline orchestrator
├── LICENSE
└── README.md                  # this document
```

---

## Contributions and external resources (acknowledgements)

This research builds on multiple third‑party datasets, model families and tools. All are acknowledged and must be respected under their respective licences:

- **OpenImages v7 (Urban classes)** — primary source of real urban/architectural images.
- **Imaginary Cities (synthetic dataset)** — authored and curated by Paul Suarez Fuenmayor; generation workflow and prompt/configuration exported at `data/imaginary_synthetic/Imaginary Cities Generation Workflow.json`.
- **Flux.1d** (Black Forest Labs) — diffusion backbone used for fine‑tuning and inference.
- **BLIP‑2** (Salesforce) — used for caption generation and semiotic prompt extraction.
- **Segment Anything Model (SAM)** (Meta AI) — primary segmentation component; alternative detectors (e.g. Ultralytics/YOLO) are supported where noted.
- **FiftyOne (Voxel51)** — dataset management and visual inspection tools.
- **Hugging Face ecosystem** — transformers, diffusers and model hosting.

Please consult the original projects for licensing and citation details before any commercial use. See the `LICENSE` file for this repository's licence (MIT).

---

## Quick technical prerequisites

- Python 3.8+
- CUDA‑capable GPU recommended (see `Performance` for hardware guidance)
- Install required packages:

```bash
pip install -r requirements.txt
```

Model weights (Flux.1d, BLIP‑2, SAM/YOLO variants) are not included in the repository and must be obtained separately — scripts assume standard Hugging Face identifiers or local paths (see `scripts/*` headers for expected locations).

---

## Pipeline overview (phases)

### Standard LoRA Pipeline
1. **Data ingestion & unification** — combine OpenImages urban classes and the synthetic Imaginary Cities set into a standardized FiftyOne dataset, deduplicate and normalize metadata.
2. **Captioning** — generate descriptive, semiotic‑oriented captions using BLIP‑2 with structured prompts.
3. **Segmentation** — extract masks and regions of architectural relevance using SAM (or YOLO as an alternative).
4. **Semiotic feature extraction** — compute CLIP and sentence‑transformer embeddings, derive style/mood/material tags and other conditioning vectors.
5. **Flux data preparation** — format images, captions and metadata for LoRA fine‑tuning of Flux.1d.
6. **Training (LoRA)** — fine‑tune Flux.1d adapters with the prepared dataset, supporting mixed precision and distributed modes.
7. **Evaluation** — quantitative (CLIP, style/mood accuracy) and qualitative reporting.
8. **Inference** — conditioned image generation via CLI or optional Gradio UI.

### Full Fine-tuning Pipeline Alternative
For users requiring more comprehensive model adaptation, a full fine-tuning variant is available:
1. **Phases 01-04** — Same as LoRA pipeline (data preparation, captioning, segmentation, semiotic extraction)
2. **Enhanced data preparation** — Stricter quality filtering and enhanced semiotic conditioning optimized for full model training
3. **Full model fine-tuning** — Complete Flux.1d parameter adaptation with advanced memory optimization techniques
4. **Full model inference** — Enhanced inference pipeline optimized for fully fine-tuned models

Each phase is implemented as a discrete script under `scripts/` and can be executed independently or via the master orchestrators (`run_pipeline.py` for LoRA, `run_flux_full_pipeline.py` for full fine-tuning).

---

## Pipeline at a glance (diagram)

```mermaid
flowchart LR
	A[Real dataset: OpenImages v7\nUrban classes] --> C[01 Data Ingestion\nFiftyOne dataset]
	B[Synthetic dataset: Imaginary Cities\n(ComfyUI Flux 1 Dev)] --> C

	C --> D[02 BLIP-2 Captioning\nsemiotic captions]
	D --> E[03 SAM Segmentation\narchitectural masks]
	E --> F[04 Semiotic Feature Extraction\nembeddings + tokens]
	
	F --> G1[05 Flux Data Prep\nLoRA training data]
	G1 --> H1[06 Flux Training LoRA]
	H1 --> I[07 Evaluation]
	H1 --> J1[08 Inference CLI/Gradio]
	
	F --> G2[05 Full Data Prep\nEnhanced training data]
	G2 --> H2[06 Full Fine-tuning\nComplete model adaptation]
	H2 --> J2[08 Full Model Inference\nEnhanced generation]
```

---

## Selected scripts (concise reference)

- `scripts/01_data_pipeline.py` — unify datasets into a FiftyOne collection.
- `scripts/02_blip2_captioner.py` and `02b_blip2_captioner_export.py` — caption generation with advanced device/dtype options.
- `scripts/03_sam_segmenter.py` — segmentation, mask extraction and export.
- `scripts/04_semiotic_extractor.py` — embeddings, token extraction and feature export.
- `scripts/05_flux_data_prep.py` — build Flux.1d training corpus (images + prompts + metadata).
- `scripts/06_flux_trainer.py` — LoRA adapter training and checkpointing.
- `scripts/07_evaluation_pipeline.py` — evaluation metrics and report generation.
- `scripts/08_inference_pipeline.py` — generation CLI and optional Gradio UI.
- `scripts/99_fiftyone_setup.py` — helper to launch FiftyOne for inspection.

Each script includes argument parsing and a header documenting required model files, expected input paths and primary outputs.

---

## Scripts — detailed reference

Below is a practical, per‑script guide: the scope, typical inputs/outputs, and the most relevant arguments. Commands are PowerShell‑friendly.

### 01_data_pipeline.py — unify datasets
- Scope: Build a unified FiftyOne dataset from OIDv7 urban classes and the synthetic Imaginary Cities set; export a clean split for downstream steps.
- Inputs: `data/oid_urban/` (OIDv7 assets managed via FiftyOne), `data/imaginary_synthetic/` (CSV + images + annotations)
- Outputs: FiftyOne dataset named like `semiotic_urban_combined`; export under `data/outputs/01_data_pipeline/` with `images/{train,val}` and `labels/{train,val}`
- Key args: `--base_path`, `--dataset_name`, `--max_oid_samples`
- Example:
```powershell
python scripts/01_data_pipeline.py --base_path . --dataset_name semiotic_urban_combined --max_oid_samples 1000
```

### 02_blip2_captioner.py — semiotic BLIP‑2 captions
- Scope: Generate rich, structured captions (architecture, mood, materials, lighting, culture) and a unified caption per image.
- Inputs: directory of images or a FiftyOne dataset
- Outputs: JSON mapping image path → captions dict (includes `unified_caption`)
- Key args: `--input_dir`, `--output_file`, `--device auto|cpu|cuda`
- Example:
```powershell
python scripts/02_blip2_captioner.py --input_dir data/unified/images --output_file data/unified/captions.json --device auto
```

### 02b_blip2_captioner_export.py — BLIP‑2 with device/dtype controls
- Scope: Enhance/export captions for the training split created by 01; supports GPU selection and quantization flags.
- Inputs: `data/outputs/01_data_pipeline/{images,labels}`
- Outputs: enhanced captions per split under your chosen output dir
- Key args: `--input`, `--output`, `--device`, `--gpu_id`, `--dtype auto|fp16|bf16|fp32`, `--load_in_8bit`, `--load_in_4bit`
- Example:
```powershell
python scripts/02b_blip2_captioner_export.py --input data/outputs/01_data_pipeline --output data/outputs/blip2_captioner_export --device cuda --gpu_id 0 --dtype fp16
```

### 03_sam_segmenter.py — architectural segmentation (SAM)
- Scope: Replace YOLO with SAM for dense architectural masks; produce per‑image analysis (dominant elements, hierarchy, density).
- Inputs: directory of images
- Outputs: segmentation JSON (masks + analysis), optional visualizations
- Key args: `--model_type vit_h|vit_l|vit_b`, `--checkpoint_path models/SAM/sam_vit_h_4b8939.pth`, `--input_dir`, `--output_dir`
- Example:
```powershell
python scripts/03_sam_segmenter.py --input_dir data/unified/images --output_dir data/outputs/sam_segments --model_type vit_h --checkpoint_path models/SAM/sam_vit_h_4b8939.pth
```

### 04_semiotic_extractor.py — multi‑modal features
- Scope: Fuse captions, SAM analysis, and visual embeddings (CLIP, sentence transformers) into a single semiotic feature payload.
- Inputs: unified dataset metadata + SAM outputs
- Outputs: features JSON (per image: textual, visual, spatial, interpretations, unified embedding, score)
- Key args: `--input_data`, `--segmentation_dir`, `--output_features`
- Example:
```powershell
python scripts/04_semiotic_extractor.py --input_data data/unified --segmentation_dir data/outputs/sam_segments --output_features data/outputs/semiotic_features.json
```

### 05_flux_data_prep.py — training corpus builder
- Scope: Filter high‑quality samples; generate enhanced prompts; standardize images; split into train/val/test; export metadata.
- Inputs: FiftyOne dataset enriched with semiotic fields
- Outputs: `data/outputs/05_flux_training_data/{images, captions, metadata}` + training config JSON
- Key args: `--base_path`, `--output_path`, split ratios, quality thresholds (inside script)
- Example:
```powershell
python scripts/05_flux_data_prep.py --base_path . --output_path data/outputs/05_flux_training_data
```

### 06_flux_trainer.py — LoRA fine‑tuning
- Scope: Train Flux.1d adapters using semiotic prompts; mixed precision + Accelerator supported.
- Inputs: prepared training data (`train/`, `val/`) and config
- Outputs: `models/semiotic_flux/{checkpoints, samples}` with periodic samples and checkpoints
- Key args: via CLI or a config dict (see dataclass `TrainingConfig`); typical: `--dataset_path`, `--output_dir`, `--model_name`, `--num_epochs`, `--lora_rank`
- Example:
```powershell
python scripts/06_flux_trainer.py --dataset_path data/outputs/05_flux_training_data --output_dir models/semiotic_flux --model_name black-forest-labs/FLUX.1-dev --num_epochs 10 --lora_rank 64
```

### 07_evaluation_pipeline.py — quality and coherence
- Scope: Quantitative and qualitative evaluation of generated images across semiotic, architectural, and visual metrics.
- Inputs: generated images + prompts (and optionally pipeline components)
- Outputs: CSV/JSON reports and plots
- Key args: vary by evaluation mode; typical: `--images_dir`, `--prompts_file`, thresholds within the script
- Example:
```powershell
python scripts/07_evaluation_pipeline.py --images_dir generated_samples --prompts_file prompts.json
```

### 08_inference_pipeline.py — generation (CLI or Gradio)
- Scope: Load base model + LoRA; apply semiotic tokens; generate images; optional live UI.
- Inputs: prompt(s) + conditioning tokens
- Outputs: images and optional evaluation summaries
- Key args: `--model_path`, `--lora_path` (optional), `--prompt`, `--height`, `--width`, `--num_inference_steps`, `--guidance_scale`, `--num_images`, `--seed`, `--gradio`
- Example:
```powershell
python scripts/08_inference_pipeline.py --model_path models/semiotic_flux --prompt "Modern glass office building in contemplative urban setting" --num_images 2 --seed 42
```

### 99_fiftyone_setup.py — dataset viewer
- Scope: Quickly open a FiftyOne session to browse your dataset.
- Example:
```powershell
python scripts/99_fiftyone_setup.py
```

---

## Full Fine-tuning Pipeline Alternative

For users requiring more comprehensive model adaptation than LoRA provides, a complete full fine-tuning variant is available in the `scripts/flux_full_finetune/` subdirectory:

### flux_full_finetune/05_flux_full_data_prep.py — Enhanced data preparation
- Scope: Specialized data preparation optimized for full model fine-tuning with stricter quality filtering and enhanced semiotic conditioning.
- Key features: Multiple caption generation templates, WebP compression, semiotic token augmentation, quality score filtering (min_semiotic_score=0.5)
- Example:
```powershell
python scripts/flux_full_finetune/05_flux_full_data_prep.py --dataset-path data/export --output-dir models/flux_full_finetuned/training_data --min-semiotic-score 0.5
```

### flux_full_finetune/06_flux_full_trainer.py — Complete model fine-tuning
- Scope: Full parameter fine-tuning of Flux.1d with advanced memory optimization techniques including gradient checkpointing, EMA, layer-wise learning rates.
- Key features: Enhanced semiotic conditioning, 8-bit Adam optimization, model component freezing options, comprehensive logging
- Example:
```powershell
python scripts/flux_full_finetune/06_flux_full_trainer.py --training-data models/flux_full_finetuned/training_data --output-dir models/flux_full_finetuned --epochs 3 --learning-rate 1e-5
```

### flux_full_finetune/08_flux_full_inference.py — Enhanced inference
- Scope: Optimized inference pipeline for fully fine-tuned models with enhanced prompt engineering and semiotic conditioning.
- Key features: Advanced prompt templates, semiotic feature analysis, memory-efficient generation, comprehensive metadata tracking
- Example:
```powershell
python scripts/flux_full_finetune/08_flux_full_inference.py --model-path models/flux_full_finetuned --prompt "A brutalist residential complex showcasing hierarchy and monumentality" --num-samples 4
```

### run_flux_full_pipeline.py — Full fine-tuning orchestrator
- Scope: Complete pipeline orchestrator that integrates standard phases (01-04) with full fine-tuning phases (05-08).
- Key features: Phase control, comprehensive logging, memory optimization, pipeline reporting
- Example:
```powershell
python run_flux_full_pipeline.py --phases 01 02 03 04 05 06 08 --epochs 3 --learning-rate 1e-5 --min-semiotic-score 0.5
```

---

## Quick start (example)

### Standard LoRA Pipeline

1. Prepare data

```bash
python scripts/01_data_pipeline.py --oid_path data/oid_urban/ --synthetic_path data/imaginary_synthetic/ --output_dir data/unified/
```

2. Generate captions

```bash
python scripts/02_blip2_captioner.py --input_dir data/unified/images/ --output_file data/unified/captions.json
```

3. Run segmentation and extract semiotic features

```bash
python scripts/03_sam_segmenter.py --input_dir data/unified/images/ --output_dir data/outputs/sam_segments
python scripts/04_semiotic_extractor.py --input_data data/unified/ --segmentation_dir data/outputs/sam_segments --output_features data/outputs/semiotic_features.json
```

4. Prepare training data and fine‑tune

```bash
python scripts/05_flux_data_prep.py --input_dataset data/outputs/semiotic_features.json --output_dir data/outputs/flux_training_data
python scripts/06_flux_trainer.py --base_model black-forest-labs/FLUX.1-dev --training_data data/outputs/flux_training_data --output_dir models/semiotic_flux/ --epochs 10
```

5. Inference

```bash
python scripts/08_inference_pipeline.py --model_path models/semiotic_flux/ --prompt "Modern glass office building in contemplative urban setting" --output_dir generated_samples/
```

### Full Fine-tuning Pipeline

For comprehensive model adaptation, use the full fine-tuning orchestrator:

```bash
# Run complete full fine-tuning pipeline
python run_flux_full_pipeline.py --epochs 3 --learning-rate 1e-5 --min-semiotic-score 0.5

# Or run specific phases
python run_flux_full_pipeline.py --phases 05 06 08 --start-from 05
```

---

## Configuration notes

- Training hyperparameters and generation settings are specified in `training_config.yaml` (example included).
- Semiotic tokens (styles, moods, materials, time periods) are used to structure conditioning prompts; these tokens are defined in `scripts/config_tokens.py` or in the documentation directory.

---

## Evaluation metrics (summary)

- **Semiotic coherence**: style and mood alignment, material recognition.
- **Visual quality**: CLIP‑based semantic similarity, image sharpness and composition.
- **Dataset metrics**: diversity and intra‑style consistency.

Reports are exported as CSV/JSON and can be visualized via the included notebooks.

---

## Performance and hardware guidance

Minimum: GTX 1080 (8 GB), 16 GB RAM, 50 GB storage.
Recommended: RTX 4090 (24 GB), 32 GB RAM, SSD storage.

Timings depend strongly on batch size, resolution and model variants; see the `Performance` table in `docs/` for measured baselines.

---

## Troubleshooting (common remedies)

- **OOM errors**: reduce batch size, enable gradient checkpointing, use mixed precision or CPU offload.
- **Model download/auth failures**: verify Hugging Face authentication and model identifiers.
- **Low quality outputs**: refine captions, increase training iterations, review dataset balance.

---

## Contribution guidelines

Contributions are welcome. Please fork, create a feature branch, include tests or example outputs, update documentation and submit a pull request. For substantial contributions (new datasets, model variants) please contact the maintainers to coordinate licensing and reproducibility checks.

---

## License & citation

This repository is provided under the MIT License. See `LICENSE` for details.

If you use the pipeline or datasets in academic work, please cite the thesis and relevant third‑party projects. Example BibTeX for the thesis is provided in `CITATION.bib`.

---

## Contact

For questions, issues or collaboration requests, open an issue or contact the primary authors via their GitHub profiles.

---

**Note**: This pipeline is provided for research and educational purposes. Some model components may require separate licensing for commercial use.
