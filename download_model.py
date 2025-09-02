"""
Online script for downloading and caching Whisper base model
Run this script with internet connection to download model for offline training
"""

import os
from transformers import WhisperForConditionalGeneration
from config import get_config


def download_base_model(config):
    """Download and cache the base Whisper model"""
    print(f"Downloading base model: {config.model_name_or_path}")
    
    # Download model with 8-bit quantization disabled for caching
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir
    )
    
    # Save model locally
    model_path = os.path.join(config.model_cache_dir, "base_model")
    model.save_pretrained(model_path)
    
    print(f"Base model saved to: {model_path}")
    return model_path


def main():
    """Main function to download base model"""
    config = get_config()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    try:
        # Download base model
        model_path = download_base_model(config)
        
        print("\n✅ Model download completed successfully!")
        print(f"Model cached at: {model_path}")
        print("You can now run offline training using the cached model.")
        
    except Exception as e:
        print(f"❌ Error during model download: {e}")
        raise


if __name__ == "__main__":
    main() 