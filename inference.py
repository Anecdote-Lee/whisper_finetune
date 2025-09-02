"""
Script for running inference with trained Whisper LoRA model
"""

import os
import torch
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel
from config import get_config


def load_inference_model(config):
    """Load model for inference"""
    print("Loading model for inference...")
    
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
    
    # Load tokenizer and processor
    tokenizer_path = os.path.join(config.model_cache_dir, "tokenizer")
    processor_path = os.path.join(config.model_cache_dir, "processor")
    
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
    processor = WhisperProcessor.from_pretrained(processor_path)
    feature_extractor = processor.feature_extractor
    
    print("Model loaded successfully")
    return model, tokenizer, processor, feature_extractor


def create_transcription_pipeline(model, tokenizer, feature_extractor, processor, config):
    """Create transcription pipeline"""
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language, 
        task=config.task
    )
    
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model, 
        tokenizer=tokenizer, 
        feature_extractor=feature_extractor
    )
    
    def transcribe(audio):
        with torch.cuda.amp.autocast():
            text = pipe(
                audio, 
                generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, 
                max_new_tokens=config.max_new_tokens
            )["text"]
        return text
    
    return transcribe


def create_gradio_interface(transcribe_fn, config):
    """Create Gradio interface for real-time transcription"""
    iface = gr.Interface(
        fn=transcribe_fn,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs="text",
        title=f"PEFT LoRA + INT8 Whisper {config.language} ASR",
        description=f"Realtime demo for {config.language} speech recognition using PEFT-LoRA+INT8 fine-tuned Whisper model.",
    )
    return iface


def transcribe_audio_file(audio_file_path, transcribe_fn):
    """Transcribe a single audio file"""
    try:
        result = transcribe_fn(audio_file_path)
        print(f"Transcription: {result}")
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def main():
    """Main function for inference"""
    config = get_config()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    try:
        # Load model for inference
        model, tokenizer, processor, feature_extractor = load_inference_model(config)
        
        # Create transcription function
        transcribe_fn = create_transcription_pipeline(
            model, tokenizer, feature_extractor, processor, config
        )
        
        print("\n✅ Model loaded successfully!")
        print("Choose an option:")
        print("1. Start Gradio interface for real-time transcription")
        print("2. Transcribe a single audio file")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print("Starting Gradio interface...")
                iface = create_gradio_interface(transcribe_fn, config)
                iface.launch(share=True)
                break
                
            elif choice == "2":
                audio_path = input("Enter path to audio file: ").strip()
                if os.path.exists(audio_path):
                    result = transcribe_audio_file(audio_path, transcribe_fn)
                    if result:
                        print(f"Transcription result: {result}")
                else:
                    print("Audio file not found!")
                    
            elif choice == "3":
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main() 