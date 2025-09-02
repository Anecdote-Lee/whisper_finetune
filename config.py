import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class WhisperConfig:
    """Configuration for Whisper LoRA fine-tuning"""
    
    # Model configuration
    model_name_or_path: str = "openai/whisper-small"
    language: str = "Korean"
    language_abbr: str = "ko"
    task: str = "transcribe"
    
    # Dataset configuration
    dataset_name: str = "mozilla-foundation/common_voice_13_0"
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training configuration
    output_dir: str = "temp"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    num_train_epochs: int = 3
    logging_steps: int = 25
    generation_max_length: int = 128
    max_new_tokens: int = 255
    
    # Data paths
    data_dir: str = "data"
    model_cache_dir: str = "models"
    
    # Device configuration
    cuda_visible_devices: str = "0"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


def get_config() -> WhisperConfig:
    """Get the default configuration"""
    return WhisperConfig() 