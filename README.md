# Semiocity Pipeline

A local pipeline to fine-tune **FLUX.1D** for architectural semiotics using FiftyOne for annotation, SAM/Detectron2 for segmentation, and GIT/CoCa for captioning.

---

## Workflow

Download OID subset
python scripts/download_oid.py

Merge with your images
python scripts/split_and_merge.py

Load into FiftyOne
python scripts/fiftyone_setup.py
Annotate ~10% manually with segmentation.

Use hierarchical labels.

Auto-segmentation
python scripts/auto_segment_sam.py

Generate captions
python scripts/generate_captions_git.py

Export for training
python scripts/export_for_training.py

Convert YOLO labels (if any)
python scripts/labels_to_masks.py


Train FLUX.1D LoRA
./scripts/train_flux_lora.sh
