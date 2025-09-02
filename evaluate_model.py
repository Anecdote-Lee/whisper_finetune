"""
Script for evaluating trained Whisper LoRA model
"""

import os
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
from peft import PeftModel, PeftConfig
from config import get_config
from utils import DataCollatorSpeechSeq2SeqWithPadding, load_metric


def load_trained_model(config):
    """Load the trained LoRA model"""
    print("Loading trained LoRA model...")
    
    # Load base model
    base_model_path = os.path.join(config.model_cache_dir, "base_model")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Load LoRA adapter
    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Trained model loaded successfully")
    return model


def evaluate_model(model, common_voice, tokenizer, processor, config):
    """Evaluate the model on test set"""
    print("Evaluating model...")
    
    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Create evaluation dataloader
    eval_dataloader = DataLoader(
        common_voice["test"], 
        batch_size=config.per_device_eval_batch_size, 
        collate_fn=data_collator
    )
    
    # Load metric
    metric = load_metric()
    
    model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=config.max_new_tokens,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
    
    wer = 100 * metric.compute()
    print(f"Word Error Rate (WER): {wer:.2f}%")
    
    return wer


def main():
    """Main function for model evaluation"""
    config = get_config()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    try:
        # Load preprocessed data
        dataset_path = os.path.join(config.data_dir, "preprocessed_dataset")
        common_voice = load_from_disk(dataset_path)
        
        # Load components
        tokenizer_path = os.path.join(config.model_cache_dir, "tokenizer")
        processor_path = os.path.join(config.model_cache_dir, "processor")
        
        tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
        processor = WhisperProcessor.from_pretrained(processor_path)
        
        # Load trained model
        model = load_trained_model(config)
        
        # Evaluate model
        wer = evaluate_model(model, common_voice, tokenizer, processor, config)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Final WER: {wer:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main() 