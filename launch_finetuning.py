#!/usr/bin/env python3
"""
S.I.Y.A Fine-tuning Launcher Script
Simple script to launch fine-tuning with various options
"""

import sys
import argparse
import os
from finetune_siYa import SiyaFineTuner

def main():
    parser = argparse.ArgumentParser(description="S.I.Y.A Enhanced Fine-tuning")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2-0.6B-Instruct",
        help="Base model to fine-tune (default: Qwen/Qwen2-0.6B-Instruct)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--lora-rank", 
        type=int, 
        default=16,
        help="LoRA rank parameter (default: 16)"
    )
    
    parser.add_argument(
        "--dataset-sources", 
        nargs="+", 
        default=["oasst1", "persona_chat", "daily_dialog"],
        help="Dataset sources to use (default: oasst1 persona_chat daily_dialog)"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Test the fine-tuned model without training"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🚀 S.I.Y.A Enhanced Fine-tuning Launcher")
    print("=" * 50)
    
    if args.verbose:
        print(f"Base Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"LoRA Rank: {args.lora_rank}")
        print(f"Dataset Sources: {args.dataset_sources}")
        print("-" * 50)
    
    # Initialize fine-tuner
    finetuner = SiyaFineTuner(base_model_name=args.model)
    
    try:
        if args.test_only:
            print("🔍 Testing existing fine-tuned model...")
            finetuner.load_model_and_tokenizer()
            finetuner.setup_lora(r=args.lora_rank)
            finetuner.test_model()
        else:
            print("🎯 Starting fine-tuning process...")
            
            # Modify training args if needed
            finetuner.create_training_args = lambda: {
                "output_dir": "./siya-finetuned",
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "per_device_eval_batch_size": args.batch_size,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "learning_rate": args.learning_rate,
                "fp16": True,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "report_to": None,
                "dataloader_pin_memory": False,
            }
            
            finetuner.run_fine_tuning_pipeline()
            
            print("\n✅ Fine-tuning completed successfully!")
            print("📁 Model saved to: ./siya-finetuned")
            print("📊 Training plot saved to: training_history.png")
            
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    requirements = [
        "torch",
        "transformers", 
        "accelerate",
        "datasets",
        "peft",
        "optimum",
        "bitsandbytes",
        "huggingface-hub",
        "tqdm",
        "matplotlib",
        "numpy"
    ]
    
    import subprocess
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    print("✅ Dependencies installation completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    else:
        main()