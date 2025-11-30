#!/usr/bin/env python3
"""
S.I.Y.A Enhanced Fine-tuning Pipeline
Fine-tuning with LoRA for conversation improvement while preserving base knowledge
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import os
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    import warnings
    import seaborn as sns
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class SiyaFineTuner:
    def __init__(self, base_model_name="Qwen/Qwen2-0.6B-Instruct"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        print(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.model, self.tokenizer
    
    def setup_lora(self, r=16, alpha=32, dropout=0.1):
        """Setup LoRA configuration for efficient fine-tuning"""
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Add LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_datasets(self):
        """Prepare and combine multiple conversation datasets"""
        
        # Dataset configurations
        datasets_config = [
            {
                "name": "oasst1",
                "description": "OpenAssistant conversations",
                "load_func": self.load_oasst_dataset
            },
            {
                "name": "anthropic_h3", 
                "description": "Anthropic helpful conversations",
                "load_func": self.load_anthropic_dataset
            },
            {
                "name": "persona_chat",
                "description": "Persona-based conversations", 
                "load_func": self.load_persona_dataset
            },
            {
                "name": "daily_dialog",
                "description": "Daily conversation patterns",
                "load_func": self.load_daily_dialog
            },
            {
                "name": "wizard_wikipedia",
                "description": "Knowledge-based conversations",
                "load_func": self.load_wizard_wiki
            }
        ]
        
        all_datasets = []
        
        for dataset_config in datasets_config:
            try:
                print(f"Loading {dataset_config['description']}...")
                dataset = dataset_config["load_func"]()
                if dataset and len(dataset) > 0:
                    all_datasets.append(dataset)
                    print(f"✓ Loaded {len(dataset)} samples from {dataset_config['name']}")
                else:
                    print(f"✗ Failed to load {dataset_config['name']}")
            except Exception as e:
                print(f"✗ Error loading {dataset_config['name']}: {e}")
                continue
        
        if not all_datasets:
            print("No datasets loaded successfully. Creating sample data...")
            all_datasets = [self.create_sample_conversations()]
        
        # Combine all datasets
        combined_dataset = self.combine_datasets(all_datasets)
        return combined_dataset
    
    def load_oasst_dataset(self):
        """Load OpenAssistant dataset"""
        try:
            # Try to load from HuggingFace
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            return self.process_conversations(dataset)
        except Exception as e:
            print(f"Could not load OASST dataset: {e}")
            return None
    
    def load_anthropic_dataset(self):
        """Load Anthropic dataset (placeholder - requires access)"""
        # Placeholder for Anthropic dataset
        # In practice, you would need API access or use available subsets
        return None
    
    def load_persona_dataset(self):
        """Load Persona-Chat dataset"""
        try:
            dataset = load_dataset("personachat", split="train")
            return self.process_persona_conversations(dataset)
        except Exception as e:
            print(f"Could not load Persona-Chat dataset: {e}")
            return None
    
    def load_daily_dialog(self):
        """Load DailyDialog dataset"""
        try:
            dataset = load_dataset("daily_dialog", split="train")
            return self.process_daily_dialog(dataset)
        except Exception as e:
            print(f"Could not load DailyDialog dataset: {e}")
            return None
    
    def load_wizard_wiki(self):
        """Load Wizard of Wikipedia dataset"""
        try:
            dataset = load_dataset("wizard_of_wikipedia", split="train")
            return self.process_wizard_wiki(dataset)
        except Exception as e:
            print(f"Could not load Wizard of Wikipedia: {e}")
            return None
    
    def process_conversations(self, dataset):
        """Process general conversation dataset"""
        processed = []
        for item in dataset:
            if 'text' in item and len(item['text']) > 50:  # Filter short responses
                processed.append({
                    "input": item['text'][:100],  # First 100 chars as input
                    "output": item['text'][100:] if len(item['text']) > 100 else item['text'][:50]
                })
        return processed
    
    def process_persona_conversations(self, dataset):
        """Process Persona-Chat dataset"""
        processed = []
        for item in dataset:
            if 'history' in item and 'candidates' in item:
                history = item['history']
                candidates = item['candidates'][0] if item['candidates'] else ""
                
                if history and candidates:
                    processed.append({
                        "input": f"Persona: {', '.join(item.get('personality', ['friendly', 'helpful']))}",
                        "conversation": history,
                        "output": candidates
                    })
        return processed
    
    def process_daily_dialog(self, dataset):
        """Process DailyDialog dataset"""
        processed = []
        for item in dataset:
            if 'dialog' in item and len(item['dialog']) > 1:
                dialog = item['dialog']
                for i in range(len(dialog) - 1):
                    processed.append({
                        "input": dialog[i],
                        "output": dialog[i + 1]
                    })
        return processed
    
    def process_wizard_wiki(self, dataset):
        """Process Wizard of Wikipedia dataset"""
        processed = []
        for item in dataset:
            if 'dialog' in item and 'passage' in item:
                passage = item['passage']
                for turn in item['dialog']:
                    processed.append({
                        "input": f"Knowledge: {passage}",
                        "question": turn.get('query', ''),
                        "output": turn.get('answer', '')
                    })
        return processed
    
    def create_sample_conversations(self):
        """Create sample conversations for demonstration"""
        sample_conversations = [
            {
                "input": "What makes a good conversation?",
                "output": "A good conversation flows naturally, both parties listen actively, and it feels balanced. It's when you're genuinely interested in what the other person has to say, and they feel the same way about you!"
            },
            {
                "input": "Tell me a joke about programming",
                "output": "Why do programmers prefer dark mode? Because light attracts bugs! 😄 (Sorry, couldn't resist the classic programmer humor!)"
            },
            {
                "input": "How can I improve my communication skills?",
                "output": "Great question! Practice active listening - it's just as important as speaking well. Ask follow-up questions, paraphrase what you've heard, and try to understand the emotion behind words. Also, match your communication style to who you're talking with!"
            },
            {
                "input": "What are some natural ways to stay motivated?",
                "output": "Break big goals into smaller wins, celebrate progress (even tiny steps!), find an accountability buddy, and remember why you started. Oh, and sometimes motivation is about creating good habits rather than waiting to 'feel' motivated!"
            },
            {
                "input": "Can you help me think through this problem?",
                "output": "Absolutely! Let's break it down step by step. First, let's define exactly what the problem is, then list what we know, what we don't know, and possible solutions. Sometimes just articulating it clearly helps us see the answer!"
            }
        ]
        return sample_conversations
    
    def combine_datasets(self, datasets):
        """Combine multiple datasets with proper formatting"""
        all_samples = []
        
        for dataset in datasets:
            if dataset:
                all_samples.extend(dataset)
        
        # Create HF dataset
        combined_dataset = Dataset.from_list(all_samples)
        print(f"Combined dataset size: {len(combined_dataset)} samples")
        
        return combined_dataset
    
    def tokenize_function(self, examples):
        """Tokenize the dataset for training"""
        # For conversational format, combine input and output
        texts = []
        for example in examples["input"]:
            # Create conversational prompt
            prompt = f"Human: {example}\nAssistant:"
            texts.append(prompt)
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Add labels (for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def create_training_args(self):
        """Create training arguments for LoRA fine-tuning"""
        return TrainingArguments(
            output_dir="./siya-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb reporting
            dataloader_pin_memory=False,
        )
    
    def train_model(self, dataset):
        """Train the model with LoRA"""
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        print("Setting up training arguments...")
        training_args = self.create_training_args()
        
        print("Creating trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset,
            eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving fine-tuned model...")
        trainer.save_model("./siya-finetuned")
        self.tokenizer.save_pretrained("./siya-finetuned")
        
        return trainer
    
    def evaluate_model(self, trainer):
        """Evaluate the fine-tuned model"""
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print("Evaluation Results:", eval_results)
        return eval_results
    
    def test_model(self):
        """Test the fine-tuned model with sample conversations"""
        print("Testing fine-tuned model...")
        
        test_prompts = [
            "What's the best way to learn something new?",
            "Tell me about making conversations more engaging",
            "How do I handle difficult conversations?",
            "What's your take on work-life balance?",
            "Can you help me with creative problem solving?"
        ]
        
        for prompt in test_prompts:
            print(f"\n--- Testing: {prompt} ---")
            
            # Format input for the model
            input_text = f"Human: {prompt}\nAssistant:"
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            print(f"Response: {response}")
    
    def plot_training_history(self, trainer):
        """Plot training and evaluation history"""
        try:
            logs = trainer.state.log_history
            
            # Extract training and eval losses
            train_losses = []
            eval_losses = []
            steps = []
            
            for log in logs:
                if 'loss' in log and 'step' in log:
                    train_losses.append(log['loss'])
                    steps.append(log['step'])
                elif 'eval_loss' in log and 'step' in log:
                    eval_losses.append((log['eval_loss'], log['step']))
            
            setup_matplotlib_for_plotting()
            
            plt.figure(figsize=(12, 6))
            
            # Plot training loss
            if train_losses:
                plt.subplot(1, 2, 1)
                plt.plot(steps[:len(train_losses)], train_losses)
                plt.title('Training Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.grid(True)
            
            # Plot eval loss if available
            if eval_losses:
                plt.subplot(1, 2, 2)
                eval_steps = [step for _, step in eval_losses]
                eval_values = [loss for loss, _ in eval_losses]
                plt.plot(eval_steps, eval_values, 'r-', label='Eval Loss')
                plt.title('Evaluation Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Training history plotted to training_history.png")
            
        except Exception as e:
            print(f"Could not create training plot: {e}")
    
    def run_fine_tuning_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        print("=" * 60)
        print("S.I.Y.A Enhanced Fine-tuning Pipeline")
        print("=" * 60)
        
        # Step 1: Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Step 2: Setup LoRA
        self.setup_lora()
        
        # Step 3: Prepare datasets
        dataset = self.prepare_datasets()
        
        # Step 4: Train model
        trainer = self.train_model(dataset)
        
        # Step 5: Evaluate
        self.evaluate_model(trainer)
        
        # Step 6: Plot training history
        self.plot_training_history(trainer)
        
        # Step 7: Test model
        self.test_model()
        
        print("\n" + "=" * 60)
        print("Fine-tuning complete!")
        print("Model saved to: ./siya-finetuned")
        print("=" * 60)
        
        return trainer

if __name__ == "__main__":
    # Initialize and run fine-tuning
    finetuner = SiyaFineTuner()
    finetuner.run_fine_tuning_pipeline()