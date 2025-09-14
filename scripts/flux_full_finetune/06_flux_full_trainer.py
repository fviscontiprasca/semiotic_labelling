"""
Full Flux.1d Fine-tuning Pipeline for Semiotic-Aware Image Generation
Alternative to LoRA training - performs full model fine-tuning with gradient checkpointing
and memory optimization for semiotic conditioning.
"""

import os
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# ML libraries with error handling
try:
    from datasets import Dataset
    from transformers import (
        CLIPTextModel, CLIPTokenizer,
        get_scheduler, set_seed
    )
    from diffusers import FluxPipeline, FluxTransformer2DModel, DDPMScheduler
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML libraries not available: {e}")

# Optional tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class FullTrainingConfig:
    """Configuration for full Flux fine-tuning."""
    
    # Dataset
    dataset_path: str
    output_dir: str
    
    # Model
    model_name: str = "black-forest-labs/FLUX.1-dev"
    revision: str = "main"
    
    # Training parameters
    resolution: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6  # Lower for full fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_train_steps: Optional[int] = None
    num_epochs: int = 5  # Fewer epochs for full fine-tuning
    warmup_steps: int = 500
    lr_scheduler: str = "cosine_with_restarts"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    dataloader_num_workers: int = 2
    
    # Model fine-tuning specific
    freeze_text_encoder: bool = True
    freeze_vae: bool = True
    train_transformer_layers: Optional[List[int]] = None  # None = all layers
    layer_wise_lr_decay: float = 0.9  # Decay factor for earlier layers
    
    # Conditioning
    use_semiotic_conditioning: bool = True
    max_caption_length: int = 77
    guidance_scale: float = 7.5
    prediction_type: str = "epsilon"
    
    # Regularization
    noise_offset: float = 0.1  # Add noise for better training
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Logging and checkpointing
    logging_steps: int = 10
    validation_steps: int = 250
    save_steps: int = 1000
    sample_steps: int = 2000
    max_checkpoints_to_keep: int = 3
    
    # System
    mixed_precision: str = "fp16"
    num_workers: int = 2
    seed: int = 42
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "semiotic-flux-full"

class SemioticFluxFullTrainer:
    """Full fine-tuner for Flux.1d with semiotic conditioning."""
    
    def __init__(self, config: Union[Dict[str, Any], FullTrainingConfig]):
        """Initialize the full fine-tuning trainer."""
        
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available. Please install transformers, diffusers, accelerate.")
        
        # Convert config to FullTrainingConfig if needed
        if isinstance(config, dict):
            self.config = FullTrainingConfig(**config)
        else:
            self.config = config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb" if self.config.use_wandb and WANDB_AVAILABLE else None,
            project_dir=self.config.output_dir
        )
        
        # Set up output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        
        # Initialize models (will be loaded in load_models)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.ema_transformer = None
        
        # Training state
        self.optimizer = None
        self.lr_scheduler = None
        self.max_train_steps = None
        
        logger.info(f"Full fine-tuning trainer initialized. Output dir: {self.output_dir}")
    
    def load_models(self):
        """Load and setup Flux models for full fine-tuning."""
        
        logger.info(f"Loading Flux models from {self.config.model_name}")
        
        # Load the full pipeline first
        pipeline = FluxPipeline.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # Extract components
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        
        # Setup trainable parameters
        self.setup_trainable_parameters()
        
        # Setup EMA if requested
        if self.config.use_ema:
            self.setup_ema()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing")
        
        # Freeze components as specified
        if self.config.freeze_vae:
            self.vae.requires_grad_(False)
            logger.info("VAE frozen")
        
        if self.config.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            logger.info("Text encoder frozen")
        
        # Enable training mode for trainable components
        self.transformer.train()
        
        logger.info("Models loaded and configured for full fine-tuning")
    
    def setup_trainable_parameters(self):
        """Setup which transformer layers to train with layer-wise learning rates."""
        
        # Get all transformer layers
        transformer_layers = list(self.transformer.named_parameters())
        
        # Determine which layers to train
        if self.config.train_transformer_layers is not None:
            # Train only specified layers
            trainable_params = []
            for name, param in transformer_layers:
                layer_num = self._extract_layer_number(name)
                if layer_num is None or layer_num in self.config.train_transformer_layers:
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        else:
            # Train all transformer parameters
            trainable_params = [param for name, param in transformer_layers]
        
        # Count parameters
        trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
        total_count = sum(p.numel() for p in self.transformer.parameters())
        
        logger.info(f"Full fine-tuning configured: {trainable_count:,} trainable / {total_count:,} total parameters "
                   f"({100 * trainable_count / total_count:.2f}%)")
        
        return trainable_params
    
    def _extract_layer_number(self, param_name: str) -> Optional[int]:
        """Extract layer number from parameter name for layer-wise learning rates."""
        
        # This is a simplified implementation - adjust based on actual Flux architecture
        import re
        match = re.search(r'layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None
    
    def setup_ema(self):
        """Setup Exponential Moving Average for model weights."""
        
        try:
            from copy import deepcopy
            self.ema_transformer = deepcopy(self.transformer)
            self.ema_transformer.requires_grad_(False)
            logger.info("EMA model initialized")
        except Exception as e:
            logger.warning(f"Could not setup EMA: {e}")
            self.ema_transformer = None
    
    def update_ema(self):
        """Update EMA weights."""
        
        if self.ema_transformer is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_transformer.parameters(), self.transformer.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(param.data, alpha=1 - self.config.ema_decay)
    
    def load_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        """Load training dataset optimized for full fine-tuning."""
        
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Load training data
        train_samples = []
        val_samples = []
        
        # Load from directory structure
        if dataset_path.is_dir():
            # Look for train/val structure
            train_dir = dataset_path / "train" 
            val_dir = dataset_path / "val"
            
            if train_dir.exists():
                train_samples = self._load_samples_from_dir(train_dir)
                logger.info(f"Loaded {len(train_samples)} training samples")
            
            if val_dir.exists():
                val_samples = self._load_samples_from_dir(val_dir)
                logger.info(f"Loaded {len(val_samples)} validation samples")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_samples)
        val_dataset = Dataset.from_list(val_samples) if val_samples else None
        
        return train_dataset, val_dataset
    
    def _load_samples_from_dir(self, data_dir: Path) -> List[Dict[str, str]]:
        """Load samples from directory structure."""
        
        samples = []
        
        # Look for images and metadata
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Try to find metadata file
        metadata_file = None
        for meta_name in ['metadata.json', 'captions.json', 'data.json']:
            if (data_dir / meta_name).exists():
                metadata_file = data_dir / meta_name
                break
        
        metadata = {}
        if metadata_file:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Collect image files
        for image_path in data_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                
                # Get caption and semiotic features
                caption = ""
                semiotic_features = {}
                
                # Try to get from metadata
                rel_path = str(image_path.relative_to(data_dir))
                if rel_path in metadata:
                    item_data = metadata[rel_path]
                    caption = item_data.get('caption', '')
                    semiotic_features = item_data.get('semiotic_features', {})
                
                # Enhanced caption generation for full fine-tuning
                if not caption:
                    caption = self._generate_enhanced_caption_from_filename(image_path.name)
                
                samples.append({
                    "image_path": str(image_path),
                    "caption": caption,
                    "semiotic_features": semiotic_features,
                    "filename": image_path.name
                })
        
        return samples
    
    def _generate_enhanced_caption_from_filename(self, filename: str) -> str:
        """Generate enhanced semiotic caption from filename for full fine-tuning."""
        
        # Remove extension and replace underscores/hyphens with spaces
        base_name = Path(filename).stem
        caption = base_name.replace('_', ' ').replace('-', ' ')
        
        # Add semiotic architectural context for full fine-tuning
        semiotic_prefixes = [
            "semiotic architectural photography of",
            "architectural scene showing",
            "urban environment featuring",
            "built environment with"
        ]
        
        import random
        prefix = random.choice(semiotic_prefixes)
        caption = f"{prefix} {caption}, photorealistic architectural design"
        
        return caption
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Create optimized dataloaders for full fine-tuning."""
        
        def collate_fn(examples):
            """Custom collate function for full fine-tuning."""
            
            pixel_values = []
            input_ids = []
            
            for example in examples:
                # Load and process image
                image = Image.open(example["image_path"]).convert("RGB")
                
                # Resize to training resolution
                image = image.resize((self.config.resolution, self.config.resolution), Image.Resampling.LANCZOS)
                
                # Convert to tensor and normalize
                image_array = np.array(image).astype(np.float32) / 255.0
                image_array = (image_array - 0.5) / 0.5  # Normalize to [-1, 1]
                pixel_values.append(torch.from_numpy(image_array).permute(2, 0, 1))
                
                # Tokenize caption
                caption = example["caption"]
                
                # Add semiotic conditioning tokens if available
                if self.config.use_semiotic_conditioning and "semiotic_features" in example:
                    caption = self._enhance_caption_with_semiotic_tokens(caption, example["semiotic_features"])
                
                tokenized = self.tokenizer(
                    caption,
                    max_length=self.config.max_caption_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids.append(tokenized.input_ids[0])
            
            return {
                "pixel_values": torch.stack(pixel_values),
                "input_ids": torch.stack(input_ids)
            }
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=True
            )
        
        return train_dataloader, val_dataloader
    
    def _enhance_caption_with_semiotic_tokens(self, caption: str, semiotic_features: Dict[str, Any]) -> str:
        """Enhance caption with semiotic conditioning tokens."""
        
        semiotic_tokens = []
        
        # Add architectural style token
        if "architectural_style" in semiotic_features and semiotic_features["architectural_style"]:
            style = semiotic_features["architectural_style"]
            semiotic_tokens.append(f"<{style}>")
        
        # Add mood token
        if "urban_mood" in semiotic_features and semiotic_features["urban_mood"]:
            mood = semiotic_features["urban_mood"]
            semiotic_tokens.append(f"<{mood}>")
        
        # Add material tokens
        if "materials" in semiotic_features and semiotic_features["materials"]:
            materials = semiotic_features["materials"]
            if isinstance(materials, list):
                for material in materials[:2]:  # Limit to 2 materials
                    semiotic_tokens.append(f"<{material}>")
        
        # Prepend tokens to caption
        if semiotic_tokens:
            enhanced_caption = " ".join(semiotic_tokens) + " " + caption
        else:
            enhanced_caption = caption
        
        return enhanced_caption
    
    def setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer with layer-wise learning rates and scheduler."""
        
        # Calculate training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_epochs * num_update_steps_per_epoch
        self.max_train_steps = self.config.max_train_steps
        
        # Setup optimizer with layer-wise learning rates
        if self.config.layer_wise_lr_decay < 1.0:
            param_groups = self._create_layer_wise_param_groups()
        else:
            param_groups = [{"params": [p for p in self.transformer.parameters() if p.requires_grad]}]
        
        # Use 8-bit Adam if available and requested
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                optimizer_cls = AdamW
                logger.warning("8-bit Adam not available, using standard AdamW")
        else:
            optimizer_cls = AdamW
        
        self.optimizer = optimizer_cls(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        
        logger.info(f"Optimizer and scheduler setup complete. Max training steps: {self.max_train_steps}")
    
    def _create_layer_wise_param_groups(self) -> List[Dict]:
        """Create parameter groups with layer-wise learning rate decay."""
        
        param_groups = []
        base_lr = self.config.learning_rate
        
        # Group parameters by layer
        layer_params = {}
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                layer_num = self._extract_layer_number(name)
                if layer_num not in layer_params:
                    layer_params[layer_num] = []
                layer_params[layer_num].append(param)
        
        # Create param groups with decaying learning rates
        max_layer = max(layer_params.keys()) if layer_params else 0
        
        for layer_num, params in layer_params.items():
            # Earlier layers get lower learning rates
            layer_lr = base_lr * (self.config.layer_wise_lr_decay ** (max_layer - layer_num))
            
            param_groups.append({
                "params": params,
                "lr": layer_lr,
                "layer": layer_num
            })
        
        logger.info(f"Created {len(param_groups)} layer-wise parameter groups")
        return param_groups
    
    def train(self):
        """Run the full fine-tuning training loop."""
        
        logger.info("Starting full Flux fine-tuning training")
        
        # Load models
        self.load_models()
        
        # Load dataset
        train_dataset, val_dataset = self.load_dataset()
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_dataloader)
        
        # Prepare with accelerator
        self.transformer, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, train_dataloader, self.lr_scheduler
        )
        
        if val_dataloader:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        
        # Training loop
        global_step = 0
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        
        for epoch in range(self.config.num_epochs):
            self.transformer.train()
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.transformer):
                    # Forward pass
                    loss = self._compute_loss(batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        if self.config.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.config.use_ema:
                        self.update_ema()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }
                        
                        if self.accelerator.is_local_main_process:
                            logger.info(f"Step {global_step}: {logs}")
                        
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            self.accelerator.log(logs, step=global_step)
                    
                    # Validation
                    if global_step % self.config.validation_steps == 0 and val_dataloader:
                        self._validate(val_dataloader, global_step)
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)
                    
                    # Generate samples
                    if global_step % self.config.sample_steps == 0:
                        self._generate_samples(global_step)
                    
                    if global_step >= self.max_train_steps:
                        break
            
            if global_step >= self.max_train_steps:
                break
        
        # Save final model
        self._save_final_model()
        logger.info("Full fine-tuning training completed!")
    
    def _compute_loss(self, batch):
        """Compute diffusion loss for full fine-tuning."""
        
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        
        # Encode images with VAE
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Add noise
        noise = torch.randn_like(latents)
        if self.config.noise_offset > 0:
            noise += self.config.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
        
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # Predict noise
        model_pred = self.transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        # Compute loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def _validate(self, val_dataloader, step):
        """Run validation loop."""
        
        self.transformer.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._compute_loss(batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        if self.accelerator.is_local_main_process:
            logger.info(f"Validation at step {step}: loss = {avg_val_loss:.4f}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            self.accelerator.log({"val_loss": avg_val_loss}, step=step)
        
        self.transformer.train()
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        
        if self.accelerator.is_local_main_process:
            checkpoint_dir = self.checkpoints_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save transformer
            unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
            unwrapped_transformer.save_pretrained(checkpoint_dir / "transformer")
            
            # Save EMA if available
            if self.ema_transformer:
                self.ema_transformer.save_pretrained(checkpoint_dir / "transformer_ema")
            
            # Save training state
            training_state = {
                "step": step,
                "config": asdict(self.config),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            }
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            
            logger.info(f"Checkpoint saved: {checkpoint_dir}")
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        
        checkpoints = sorted([d for d in self.checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
        
        if len(checkpoints) > self.config.max_checkpoints_to_keep:
            for checkpoint in checkpoints[:-self.config.max_checkpoints_to_keep]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
    
    def _generate_samples(self, step):
        """Generate sample images during training."""
        
        if self.accelerator.is_local_main_process:
            try:
                # Use EMA model if available
                model_for_inference = self.ema_transformer if self.ema_transformer else self.transformer
                
                # Sample prompts
                sample_prompts = [
                    "modernist glass office building in contemplative urban setting",
                    "brutalist concrete housing complex with dramatic lighting",
                    "minimalist wooden pavilion in serene natural environment"
                ]
                
                # Create pipeline for inference
                inference_pipeline = FluxPipeline(
                    transformer=model_for_inference,
                    scheduler=self.scheduler,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                )
                
                # Generate samples
                for i, prompt in enumerate(sample_prompts):
                    image = inference_pipeline(
                        prompt=prompt,
                        height=512,  # Smaller for sampling
                        width=512,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        num_images_per_prompt=1,
                        generator=torch.Generator().manual_seed(42)
                    ).images[0]
                    
                    sample_path = self.samples_dir / f"step_{step:06d}_sample_{i}.jpg"
                    image.save(sample_path)
                
                logger.info(f"Generated samples at step {step}")
                
            except Exception as e:
                logger.warning(f"Sample generation failed: {e}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        
        if self.accelerator.is_local_main_process:
            final_dir = self.output_dir / "final_model"
            final_dir.mkdir(exist_ok=True)
            
            # Use EMA model if available
            model_to_save = self.ema_transformer if self.ema_transformer else self.accelerator.unwrap_model(self.transformer)
            
            # Save full pipeline
            pipeline = FluxPipeline(
                transformer=model_to_save,
                scheduler=self.scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
            )
            
            pipeline.save_pretrained(final_dir)
            
            logger.info(f"Final model saved: {final_dir}")

def main():
    parser = argparse.ArgumentParser(description="Full Flux.1d Fine-tuning")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-dev", help="Base model name")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="semiotic-flux-full", help="W&B project name")
    
    args = parser.parse_args()
    
    # Create config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = FullTrainingConfig(**config_dict)
    else:
        config = FullTrainingConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
    
    # Initialize trainer
    trainer = SemioticFluxFullTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()