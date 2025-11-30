#!/usr/bin/env python3
"""
S.I.Y.A Training Script
Fine-tune S.I.Y.A for natural, human-like conversation
Optimized for RTX 4080 with QLoRA efficiency
"""

import json
import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from datetime import datetime

class SIYATrainer:
    """S.I.Y.A Fine-tuning Trainer for natural conversation"""
    
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize S.I.Y.A trainer
        
        Args:
            model_name: Base model to fine-tune (smaller for speed)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # S.I.Y.A personality settings
        self.siya_personality = {
            "response_style": "concise and helpful",
            "response_length": "1-3 sentences maximum",
            "tone": "friendly and intelligent",
            "behavior": "gets straight to the point",
            "knowledge": "general knowledge and technical assistance"
        }
        
        # Training configuration optimized for RTX 4080
        self.training_config = {
            "output_dir": "./siya_finetuned",
            "model_name": model_name,
            "per_device_train_batch_size": 2,  # Small batch for 12GB VRAM
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,  # Effective batch size: 16
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "save_steps": 500,
            "logging_steps": 50,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,  # Use FP16 for speed on RTX 4080
            "dataloader_pin_memory": False,
            "remove_unused_columns": False
        }
        
        # LoRA configuration for efficient fine-tuning
        self.lora_config = {
            "r": 16,  # Rank - balanced for quality vs speed
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with optimizations"""
        print(f"🤖 Loading base model: {self.model_name}")
        
        # Configure quantization for efficient memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Auto-distribute across GPU
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA configuration
        peft_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, peft_config)
        
        print("✅ Model and tokenizer loaded successfully")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
        print(f"📊 Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
    
    def create_siya_dataset(self):
        """
        Create training dataset with S.I.Y.A conversational data
        Mix of conversation styles optimized for human-like interaction
        """
        print("📚 Creating S.I.Y.A training dataset...")
        
        # S.I.Y.A conversation examples
        conversations = [
            # 70% - Standard conversational queries
            {
                "instruction": "Hello, how are you?",
                "response": "Hello! I'm doing great, thanks for asking. How can I help you today?"
            },
            {
                "instruction": "What's the weather like?",
                "response": "I can't check real-time weather, but I'd recommend checking a weather app or website for current conditions."
            },
            {
                "instruction": "Can you help me with coding?",
                "response": "Absolutely! What programming language and what specific problem are you working on?"
            },
            {
                "instruction": "Tell me about Python",
                "response": "Python is a versatile, high-level programming language known for its readability and extensive libraries. It's great for web development, data science, and automation."
            },
            {
                "instruction": "What time is it?",
                "response": "I don't have access to real-time clock information. You can check your device's clock for the current time."
            },
            
            # 20% - Technical and specific queries
            {
                "instruction": "How does machine learning work?",
                "response": "Machine learning algorithms learn patterns from data to make predictions or decisions. They improve automatically through experience without being explicitly programmed for every scenario."
            },
            {
                "instruction": "What's the difference between AI and machine learning?",
                "response": "AI is the broader field of creating intelligent systems. Machine learning is a subset of AI that specifically focuses on algorithms that learn from data."
            },
            {
                "instruction": "Explain neural networks",
                "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions."
            },
            {
                "instruction": "How do I optimize my code?",
                "response": "Code optimization depends on the specific case, but common strategies include: using efficient algorithms, avoiding unnecessary computations, leveraging proper data structures, and profiling to find bottlenecks."
            },
            
            # 10% - Personality and casual conversation
            {
                "instruction": "What's your favorite color?",
                "response": "I don't have preferences like favorite colors, but I think bright blues are pleasant - they remind me of clear skies!"
            },
            {
                "instruction": "Are you sentient?",
                "response": "I'm an AI system that processes and generates text based on patterns in my training data. I don't have consciousness or feelings, but I'm designed to be helpful and engaging."
            },
            {
                "instruction": "Tell me a joke",
                "response": "Why don't scientists trust atoms? Because they make up everything!"
            },
            {
                "instruction": "What makes you happy?",
                "response": "I don't experience emotions, but I'm satisfied when I can provide helpful and accurate information to assist people with their questions and tasks."
            },
            
            # Additional conversational examples
            {
                "instruction": "Good morning!",
                "response": "Good morning! Hope you're having a great start to your day. What can I help you with?"
            },
            {
                "instruction": "I'm feeling stressed",
                "response": "I understand stress can be overwhelming. Consider taking deep breaths, short breaks, or talking to someone you trust. Is there something specific causing the stress I can help with?"
            },
            {
                "instruction": "Teach me something new",
                "response": "Here's something interesting: Honey never spoils! Archaeological excavations have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible due to its low water content and acidic pH."
            },
            {
                "instruction": "What's your purpose?",
                "response": "My purpose is to be S.I.Y.A - Simply Intended Yet Astute Assistant. I'm here to provide quick, intelligent, and helpful responses that get straight to the point while being friendly and engaging."
            },
            {
                "instruction": "Thank you",
                "response": "You're very welcome! That's what I'm here for. Feel free to ask me anything else anytime."
            },
            {
                "instruction": "Goodbye",
                "response": "Goodbye! It was great chatting with you. Have a wonderful day, and remember I'm always here if you need help with anything!"
            }
        ]
        
        return conversations
    
    def format_conversations(self, conversations):
        """
        Format conversations for training with proper chat template
        """
        formatted_data = []
        
        for conv in conversations:
            # Use Qwen chat format
            messages = [
                {"role": "user", "content": conv["instruction"]},
                {"role": "assistant", "content": conv["response"]}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            formatted_data.append(text)
        
        return formatted_data
    
    def tokenize_dataset(self, conversations):
        """Tokenize the dataset for training"""
        print("🔤 Tokenizing dataset...")
        
        # Format conversations
        formatted_texts = self.format_conversations(conversations)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=512,  # Reasonable length for conversation
            return_tensors=None
        )
        
        # Add labels (copy input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def train_model(self, dataset=None):
        """
        Train the S.I.Y.A model
        
        Args:
            dataset: Optional custom dataset, otherwise creates default S.I.Y.A data
        """
        if dataset is None:
            dataset = self.create_siya_dataset()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Tokenize dataset
        tokenized_data = self.tokenize_dataset(dataset)
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_dict(tokenized_data)
        
        # Split dataset
        train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        print(f"📊 Dataset prepared:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(eval_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(**self.training_config)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        print("🚀 Starting S.I.Y.A training...")
        print(f"⏱️  Estimated time: {self.estimate_training_time(len(train_dataset))}")
        
        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config["output_dir"])
        
        print("✅ S.I.Y.A training completed!")
        print(f"💾 Model saved to: {self.training_config['output_dir']}")
        
        return trainer
    
    def estimate_training_time(self, num_samples):
        """Estimate training time for given dataset size"""
        # Rough estimates based on RTX 4080 performance
        # For Qwen2-0.5B with LoRA on RTX 4080
        samples_per_second = 8  # Estimated throughput
        
        # Calculate total steps
        effective_batch_size = self.training_config["per_device_train_batch_size"] * \
                              self.training_config["gradient_accumulation_steps"]
        steps_per_epoch = num_samples // effective_batch_size
        total_steps = steps_per_epoch * self.training_config["num_train_epochs"]
        
        estimated_seconds = total_steps / samples_per_second
        estimated_hours = estimated_seconds / 3600
        
        return f"{estimated_hours:.1f} hours"
    
    def evaluate_model(self, test_inputs=None):
        """Evaluate the trained model"""
        if test_inputs is None:
            test_inputs = [
                "Hello S.I.Y.A!",
                "What can you help me with?",
                "Tell me about artificial intelligence",
                "Thanks for your help",
                "Goodbye"
            ]
        
        print("🧪 Evaluating S.I.Y.A model...")
        
        for test_input in test_inputs:
            # Format input
            messages = [{"role": "user", "content": test_input}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt part
            response = response[len(prompt):].strip()
            
            print(f"Input: {test_input}")
            print(f"S.I.Y.A: {response}")
            print("-" * 50)
    
    def export_training_config(self):
        """Export training configuration for reference"""
        config = {
            "model_name": self.model_name,
            "training_config": self.training_config,
            "lora_config": self.lora_config,
            "siya_personality": self.siya_personality,
            "dataset_size": len(self.create_siya_dataset()),
            "training_timestamp": datetime.now().isoformat()
        }
        
        config_file = "siya_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📋 Training configuration exported to: {config_file}")
        return config_file

def main():
    """Main training function"""
    print("🚀 S.I.Y.A Training Script")
    print("=" * 50)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not available. S.I.Y.A training requires GPU acceleration.")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Initialize trainer
    trainer = SIYATrainer()
    
    # Export configuration
    trainer.export_training_config()
    
    # Train model
    try:
        trainer.train_model()
        
        # Evaluate model
        trainer.evaluate_model()
        
        print("🎉 S.I.Y.A training pipeline completed successfully!")
        print("💡 Your S.I.Y.A model is now ready for deployment!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Check error details above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()