"""
Offline script for training Whisper with LoRA
Run this script without internet connection using preprocessed data
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import get_config
from utils import DataCollatorSpeechSeq2SeqWithPadding, load_metric, setup_model_for_training


def load_preprocessed_data(config):
    """Load preprocessed dataset and components from disk"""
    print("Loading preprocessed data...")
    
    # Load dataset
    dataset_path = os.path.join(config.data_dir, "preprocessed_dataset")
    common_voice = load_from_disk(dataset_path)
    
    # Load tokenizer and feature extractor
    tokenizer_path = os.path.join(config.model_cache_dir, "tokenizer")
    feature_extractor_path = os.path.join(config.model_cache_dir, "feature_extractor")
    processor_path = os.path.join(config.model_cache_dir, "processor")
    
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(feature_extractor_path)
    processor = WhisperProcessor.from_pretrained(processor_path)
    
    print(f"Loaded dataset: {common_voice}")
    print("Loaded tokenizer, feature extractor, and processor")
    
    return common_voice, tokenizer, feature_extractor, processor


def load_base_model(config):
    """Load the cached base model"""
    print("Loading base model...")
    
    model_path = os.path.join(config.model_cache_dir, "base_model")
    
    # Load model with 8-bit quantization for training
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto"
    )
    
    print("Base model loaded successfully")
    return model


def setup_lora_model(model, config):
    """Setup LoRA configuration and prepare model for training"""
    print("Setting up LoRA model...")
    
    # Setup model for training
    model = setup_model_for_training(model)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("LoRA model setup completed")
    return model


def setup_training_arguments(config):
    """Setup training arguments"""
    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        eval_strategy="epoch",
        report_to=["tensorboard"],
        fp16=True,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        generation_max_length=config.generation_max_length,
        logging_steps=config.logging_steps,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )


def train_model(model, common_voice, processor, config):
    """Train the model using Seq2SeqTrainer"""
    print("Starting training...")
    
    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        # Note: compute_metrics is commented out due to INT8 training limitations
        # compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    
    # Start training
    train_result = trainer.train()
    
    print("Training completed!")
    print(f"Training results: {train_result}")
    
    return trainer, train_result


def save_trained_model(trainer, config):
    """Save the trained LoRA adapter"""
    print("Saving trained model...")
    
    # Save the adapter
    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_path)
    
    print(f"LoRA adapter saved to: {adapter_path}")
    return adapter_path


def main():
    """Main function for offline training"""
    config = get_config()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    try:
        # Load preprocessed data
        common_voice, tokenizer, feature_extractor, processor = load_preprocessed_data(config)
        
        # Load base model
        model = load_base_model(config)
        
        # Setup LoRA model
        model = setup_lora_model(model, config)
        
        # Train model
        trainer, train_result = train_model(model, common_voice, processor, config)
        
        # Save trained model
        adapter_path = save_trained_model(trainer, config)
        
        print("\n✅ Offline training completed successfully!")
        print(f"LoRA adapter saved at: {adapter_path}")
        
    except Exception as e:
        print(f"❌ Error during offline training: {e}")
        raise


if __name__ == "__main__":
    main() 