"""
Flux.1d Fine-tuning Pipeline with LoRA for Semiotic-Aware Image Generation
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
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from peft import LoraConfig, get_peft_model, TaskType
    from accelerate import Accelerator
    
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
class TrainingConfig:
    """Configuration for Flux training."""
    
    # Dataset
    dataset_path: str
    output_dir: str
    
    # Model
    model_name: str = "black-forest-labs/FLUX.1-dev"
    revision: str = "main"
    
    # Training parameters
    resolution: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_train_steps: Optional[int] = None
    num_epochs: int = 10
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    
    # LoRA parameters
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Conditioning
    use_semiotic_conditioning: bool = True
    max_caption_length: int = 77
    guidance_scale: float = 7.5
    prediction_type: str = "epsilon"
    
    # Logging and checkpointing
    logging_steps: int = 10
    validation_steps: int = 100
    save_steps: int = 500
    sample_steps: int = 1000
    
    # System
    mixed_precision: str = "fp16"
    num_workers: int = 2
    seed: int = 42
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "semiotic-flux"

class SemioticFluxTrainer:
    """LoRA trainer for Flux.1d with semiotic conditioning."""
    
    def __init__(self, config: Union[Dict[str, Any], TrainingConfig]):
        """Initialize the trainer."""
        
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available. Please install transformers, diffusers, peft, accelerate.")
        
        # Convert config to TrainingConfig if needed
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
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
        
        # Training state
        self.optimizer = None
        self.lr_scheduler = None
        self.max_train_steps = None
        
        logger.info(f"Trainer initialized. Output dir: {self.output_dir}")
    
    def load_models(self):
        """Load and setup Flux models with LoRA."""
        
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
        
        # Setup LoRA for transformer
        self.setup_lora()
        
        # Freeze all models except transformer
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Enable training mode for transformer
        self.transformer.train()
        
        logger.info("Models loaded and LoRA configured")
    
    def setup_lora(self):
        """Setup LoRA configuration for the transformer."""
        
        # Define LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.DIFFUSION,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            init_lora_weights="gaussian"
        )
        
        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        
        logger.info(f"LoRA configured: {trainable_params:,} trainable / {total_params:,} total parameters "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def load_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        """Load training dataset."""
        
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Load training data
        train_samples = []
        
        # Load from directory structure
        if dataset_path.is_dir():
            # Look for train/val structure
            train_dir = dataset_path / "train" 
            val_dir = dataset_path / "val"
            
            if train_dir.exists():
                train_samples = self._load_samples_from_dir(train_dir)
                logger.info(f"Loaded {len(train_samples)} training samples")
            else:
                # Load all samples and split later
                train_samples = self._load_samples_from_dir(dataset_path)
                logger.info(f"Loaded {len(train_samples)} samples (will split for validation)")
            
            val_samples = []
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
                
                # Get caption from metadata or filename
                caption = ""
                
                # Try to get from metadata
                rel_path = str(image_path.relative_to(data_dir))
                if rel_path in metadata:
                    caption = metadata[rel_path].get('caption', '')
                elif image_path.name in metadata:
                    caption = metadata[image_path.name].get('caption', '')
                
                # Fallback to filename-based caption
                if not caption:
                    caption = self._generate_caption_from_filename(image_path.name)
                
                samples.append({
                    "image_path": str(image_path),
                    "caption": caption,
                    "filename": image_path.name
                })
        
        return samples
    
    def _generate_caption_from_filename(self, filename: str) -> str:
        """Generate basic caption from filename."""
        
        # Remove extension and replace underscores/hyphens with spaces
        base_name = Path(filename).stem
        caption = base_name.replace('_', ' ').replace('-', ' ')
        
        # Add basic architectural context
        caption = f"architectural scene showing {caption}"
        
        return caption
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Create training and validation dataloaders."""
        
        def collate_fn(examples):
            """Custom collate function for batch processing."""
            
            # Load and process images
            pixel_values = []
            input_ids = []
            
            for example in examples:
                # Load image
                image = Image.open(example["image_path"]).convert("RGB")
                
                # Resize to training resolution
                resolution = self.config.resolution
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                
                # Convert to tensor and normalize
                image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
                pixel_values.append(image_tensor)
                
                # Tokenize caption
                caption = example["caption"]
                
                # Add semiotic conditioning if enabled
                if self.config.use_semiotic_conditioning:
                    caption = self._add_semiotic_conditioning(caption)
                
                tokens = self.tokenizer(
                    caption,
                    max_length=self.config.max_caption_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids.append(tokens.input_ids[0])
            
            return {
                "pixel_values": torch.stack(pixel_values),
                "input_ids": torch.stack(input_ids)
            }
        
        # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers
            )
        
        return train_dataloader, val_dataloader
    
    def _add_semiotic_conditioning(self, caption: str) -> str:
        """Add semiotic conditioning tokens to caption."""
        
        # Extract semiotic elements from caption
        semiotic_tokens = []
        
        # Check for architectural styles
        styles = ["brutalist", "modernist", "postmodern", "minimalist", "baroque"]
        for style in styles:
            if style in caption.lower():
                semiotic_tokens.append(f"<{style}>")
                break
        
        # Check for moods
        moods = ["contemplative", "vibrant", "tense", "serene", "dramatic"]
        for mood in moods:
            if mood in caption.lower():
                semiotic_tokens.append(f"<{mood}>")
                break
        
        # Add architectural context token
        semiotic_tokens.append("<architectural>")
        
        # Prepend tokens to caption
        if semiotic_tokens:
            conditioned_caption = " ".join(semiotic_tokens) + " " + caption
            return conditioned_caption
        
        return caption
    
    def setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and learning rate scheduler."""
        
        # Get trainable parameters
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        # Setup optimizer
    self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.config.gradient_accumulation_steps
        )
        max_train_steps = self.config.max_train_steps
        
        if max_train_steps is None:
            max_train_steps = self.config.num_epochs * num_update_steps_per_epoch
        
        # Setup scheduler
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max_train_steps
        )
        
        self.max_train_steps = max_train_steps
        
        logger.info(f"Total training steps: {max_train_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step."""
        
        with self.accelerator.accumulate(self.transformer):
            # Encode images to latent space
            pixel_values = batch["pixel_values"].to(self.accelerator.device)
            
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            # Add noise for denoising objective
            noise = torch.randn_like(latents)
            
            # Sample timesteps
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            
            # Add noise to latents
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            input_ids = batch["input_ids"].to(self.accelerator.device)
            
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(input_ids)[0]
            
            # Predict noise
            model_pred = self.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # Calculate loss
            if self.config.prediction_type == "epsilon":
                target = noise
            elif self.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.config.prediction_type}")
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.detach().item()}
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save LoRA weights
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        unwrapped_transformer.save_pretrained(checkpoint_path)
        
        # Save training state
        training_state = {
            "step": step,
            "config": asdict(self.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
        }
        
    torch.save(training_state, checkpoint_path / "training_state.pt")
        
        logger.info(f"Checkpoint saved at step {step}")
    
    def train(self):
        """Main training loop."""
        
        logger.info("Starting training...")
        
        # Load models and dataset
        self.load_models()
        train_dataset, val_dataset = self.load_dataset()
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_dataloader)
        
        # Prepare everything with accelerator
        self.transformer, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, train_dataloader, self.lr_scheduler
        )
        
        if val_dataloader:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        
        # Training loop
        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_main_process
        )
        
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    progress_bar.set_postfix(**metrics)
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        self.accelerator.log(metrics, step=global_step)
                
                # Save checkpoint
                if (global_step % self.config.save_steps == 0 and 
                    global_step > 0 and 
                    self.accelerator.is_main_process):
                    
                    self.save_checkpoint(global_step)
                
                global_step += 1
                progress_bar.update(1)
                
                if global_step >= self.max_train_steps:
                    break
            
            if global_step >= self.max_train_steps:
                break
        
        # Save final checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint(global_step)
            
        logger.info("Training completed!")

def create_default_config(dataset_path: str, output_dir: str) -> TrainingConfig:
    """Create default training configuration."""
    
    return TrainingConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name="black-forest-labs/FLUX.1-dev",
        resolution=1024,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_epochs=10,
        lora_rank=64,
        lora_alpha=64,
        lora_dropout=0.1,
        use_semiotic_conditioning=True,
        mixed_precision="fp16",
        seed=42
    )

def main():
    """Main training execution."""
    
    parser = argparse.ArgumentParser(description="Train Flux.1d with semiotic awareness")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to prepared dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--config", type=str, help="Path to training config JSON")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
    else:
        config = create_default_config(args.dataset_path, args.output_dir)
    
    # Override config with args
    config.dataset_path = args.dataset_path
    config.output_dir = args.output_dir
    config.use_wandb = args.wandb
    config.num_epochs = args.epochs
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize and start training
    trainer = SemioticFluxTrainer(config)
    trainer.train()
    
    print("Training completed successfully!")
    print(f"Checkpoints saved in: {config.output_dir}")

if __name__ == "__main__":
    main()