"""
Master script for running all online setup tasks
Run this script with internet connection to prepare everything for offline training
"""

import os
import sys
from config import get_config


def run_data_download():
    """Run data download script"""
    print("=" * 50)
    print("STEP 1: Downloading and preprocessing dataset...")
    print("=" * 50)
    
    try:
        from download_data import main as download_data_main
        download_data_main()
        return True
    except Exception as e:
        print(f"❌ Error in data download: {e}")
        return False


def run_model_download():
    """Run model download script"""
    print("\n" + "=" * 50)
    print("STEP 2: Downloading and caching base model...")
    print("=" * 50)
    
    try:
        from download_model import main as download_model_main
        download_model_main()
        return True
    except Exception as e:
        print(f"❌ Error in model download: {e}")
        return False


def main():
    """Main function to run all online setup"""
    print("🚀 Starting online setup for Whisper LoRA training...")
    print("This will download and prepare all necessary data and models.")
    print("Make sure you have a stable internet connection.\n")
    
    config = get_config()
    success_count = 0
    total_steps = 2
    
    # Step 1: Download and preprocess data
    if run_data_download():
        success_count += 1
        print("✅ Data download completed")
    else:
        print("❌ Data download failed")
    
    # Step 2: Download and cache model
    if run_model_download():
        success_count += 1
        print("✅ Model download completed")
    else:
        print("❌ Model download failed")
    
    # Summary
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    if success_count == total_steps:
        print("🎉 All setup steps completed successfully!")
        print("\nYou can now run offline training with:")
        print("  python train_offline.py")
        print("\nAfter training, you can evaluate with:")
        print("  python evaluate_model.py")
        print("\nFor inference, use:")
        print("  python inference.py")
    else:
        print(f"⚠️  {success_count}/{total_steps} steps completed successfully")
        print("Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 