"""
Online script for downloading and preprocessing Common Voice dataset
Run this script with internet connection to prepare data for offline training
"""

import os
import pickle
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from config import get_config
from utils import prepare_dataset
import functools


def download_and_prepare_dataset(config):
    """Download and preprocess the Common Voice dataset"""
    print("Loading dataset...")
    
    # Load dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        config.dataset_name, 
        config.language_abbr, 
        split="train+validation",
        cache_dir=config.data_dir
    )
    common_voice["test"] = load_dataset(
        config.dataset_name, 
        config.language_abbr, 
        split="test",
        cache_dir=config.data_dir
    )
    
    print(f"Original dataset: {common_voice}")
    
    # Remove unnecessary columns
    common_voice = common_voice.remove_columns([
        "accent", "age", "client_id", "down_votes", 
        "gender", "locale", "path", "segment", "up_votes"
    ])
    
    print(f"Cleaned dataset: {common_voice}")
    
    # Cast audio to correct sampling rate
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    return common_voice


def preprocess_dataset(common_voice, config):
    """Preprocess dataset with feature extraction and tokenization"""
    print("Loading feature extractor and tokenizer...")
    
    # Load feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_name_or_path, 
        language=config.language, 
        task=config.task,
        cache_dir=config.model_cache_dir
    )
    processor = WhisperProcessor.from_pretrained(
        config.model_name_or_path, 
        language=config.language, 
        task=config.task,
        cache_dir=config.model_cache_dir
    )
    
    # Create preprocessing function with bound parameters
    prepare_fn = functools.partial(
        prepare_dataset, 
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    
    print("Preprocessing dataset...")
    # Apply preprocessing
    common_voice = common_voice.map(
        prepare_fn, 
        remove_columns=common_voice.column_names["train"], 
        num_proc=1
    )
    
    print(f"Preprocessed dataset: {common_voice}")
    
    return common_voice, feature_extractor, tokenizer, processor


def save_preprocessed_data(common_voice, feature_extractor, tokenizer, processor, config):
    """Save preprocessed data and components to disk"""
    print("Saving preprocessed data...")
    
    # Save dataset
    dataset_path = os.path.join(config.data_dir, "preprocessed_dataset")
    common_voice.save_to_disk(dataset_path)
    
    # Save tokenizer and feature extractor
    tokenizer_path = os.path.join(config.model_cache_dir, "tokenizer")
    feature_extractor_path = os.path.join(config.model_cache_dir, "feature_extractor")
    processor_path = os.path.join(config.model_cache_dir, "processor")
    
    tokenizer.save_pretrained(tokenizer_path)
    feature_extractor.save_pretrained(feature_extractor_path)
    processor.save_pretrained(processor_path)
    
    print(f"Dataset saved to: {dataset_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Feature extractor saved to: {feature_extractor_path}")
    print(f"Processor saved to: {processor_path}")


def main():
    """Main function to download and prepare data"""
    config = get_config()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    try:
        # Download and prepare dataset
        common_voice = download_and_prepare_dataset(config)
        
        # Preprocess dataset
        common_voice, feature_extractor, tokenizer, processor = preprocess_dataset(common_voice, config)
        
        # Save everything for offline use
        save_preprocessed_data(common_voice, feature_extractor, tokenizer, processor, config)
        
        print("\n✅ Data preparation completed successfully!")
        print("You can now run offline training using the saved data.")
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main() 