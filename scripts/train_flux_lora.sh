#!/bin/bash
# scripts/train_flux_lora.sh

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="outputs/flux1_lora"

accelerate launch --mixed_precision=fp16 \
  train_controlnet.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --controlnet_model_name_or_path lllyasviel/sd-controlnet-seg \
  --train_data_dir data/fiftyone_export/images \
  --train_mask_dir data/fiftyone_export/masks \
  --train_data_file data/fiftyone_export/captions.csv \
  --output_dir $OUTPUT_DIR \
  --resolution 512 \
  --train_batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --save_every_n_epochs 1 \
  --lora_rank 16
