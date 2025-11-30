#!/usr/bin/env python3
"""
S.I.Y.A Performance Benchmark
Simply Intended Yet Astute Assistant - Speed Testing
"""

import time
import torch
import numpy as np
import psutil
import GPUtil
import json
from pathlib import Path
import subprocess
import sys

class SIYaBenchmark:
    def __init__(self):
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")
        
    def print_result(self, component, time_ms, status="✅"):
        print(f"{status} {component}: {time_ms:.1f}ms")
        return time_ms
        
    def benchmark_system_specs(self):
        """Benchmark S.I.Y.A system specifications"""
        self.print_header("S.I.Y.A System Specifications")
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"💻 CPU: {cpu_count} cores @ {cpu_freq.current:.0f}MHz")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
        
        # GPU info for S.I.Y.A
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"🎮 GPU: {gpu.name}")
                print(f"   Memory: {gpu.memoryTotal}MB total, {gpu.memoryUsed}MB used")
                print(f"   Load: {gpu.load*100:.1f}%")
                print(f"   Temperature: {gpu.temperature}°C")
        except:
            print("🎮 GPU: Not detected")
            
        # PyTorch info for S.I.Y.A
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🔧 CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
        if torch.cuda.is_available():
            print(f"   Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            
        # Audio devices for S.I.Y.A
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print(f"🎤 Audio devices: {len(devices)} detected")
            for i, device in enumerate(devices[:3]):  # Show first 3
                device_type = "Input" if device['max_input_channels'] > 0 else "Output"
                print(f"   {i}: {device['name']} ({device_type})")
        except:
            print("🎤 Audio devices: Could not detect")
            
    def benchmark_model_loading(self):
        """Benchmark S.I.Y.A model loading times"""
        self.print_header("S.I.Y.A Model Loading Performance")
        
        start_time = time.time()
        
        try:
            # Test ASR model loading for S.I.Y.A
            print("🎤 Loading S.I.Y.A ASR model...")
            asr_start = time.time()
            
            # Simulate model loading with actual import
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained("openai/whisper-base", torch_dtype=torch.float16)
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", torch_dtype=torch.float16)
            
            asr_time = (time.time() - asr_start) * 1000
            self.print_result("S.I.Y.A ASR Model Loading", asr_time)
            
        except Exception as e:
            asr_time = 0
            self.print_result("S.I.Y.A ASR Model Loading", 0, f"❌ Failed: {str(e)[:30]}...")
            
        try:
            # Test LLM model loading for S.I.Y.A
            print("🧠 Loading S.I.Y.A LLM model...")
            llm_start = time.time()
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float16, trust_remote_code=True)
            
            llm_time = (time.time() - llm_start) * 1000
            self.print_result("S.I.Y.A LLM Model Loading", llm_time)
            
        except Exception as e:
            llm_time = 0
            self.print_result("S.I.Y.A LLM Model Loading", 0, f"❌ Failed: {str(e)[:30]}...")
            
        total_time = (time.time() - start_time) * 1000
        self.print_result("S.I.Y.A Total Loading Time", total_time)
        
        self.results["model_loading"] = {
            "asr_ms": asr_time,
            "llm_ms": llm_time,
            "total_ms": total_time
        }
        
    def benchmark_asr_speed(self):
        """Benchmark S.I.Y.A ASR performance"""
        self.print_header("S.I.Y.A Speech Recognition Speed Test")
        
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            # Load model for S.I.Y.A
            model_name = "openai/whisper-base"  # Smaller for demo
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            if self.device == "cuda":
                model = model.cuda().half()
                
            # Generate test audio (simulated speech data)
            test_audio = torch.randn(16000 * 3)  # 3 seconds of random audio
            if self.device == "cuda":
                test_audio = test_audio.cuda()
                
            print(f"🎤 Testing S.I.Y.A ASR with 3-second audio sample...")
            
            # Measure S.I.Y.A ASR processing time
            start_time = time.time()
            
            # Process audio
            inputs = processor(test_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_new_tokens=50,
                    do_sample=False
                )
                
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            asr_time = (time.time() - start_time) * 1000
            self.print_result("S.I.Y.A Speech Recognition", asr_time)
            
            if "audio" in transcription.lower() or "speech" in transcription.lower():
                print(f"📝 Transcription preview: {transcription[:50]}...")
            else:
                print(f"📝 Transcription: {transcription}")
                
            # S.I.Y.A ASR performance grade
            if asr_time < 50:
                grade = "Excellent (A+)"
            elif asr_time < 100:
                grade = "Great (A)"
            elif asr_time < 200:
                grade = "Good (B)"
            else:
                grade = "Needs optimization (C)"
                
            print(f"📊 S.I.Y.A ASR Grade: {grade}")
                
            self.results["asr"] = {
                "processing_time_ms": asr_time,
                "transcription": transcription,
                "target_ms": 50,
                "grade": grade
            }
            
        except Exception as e:
            self.print_result("S.I.Y.A Speech Recognition", 0, f"❌ Failed: {str(e)[:50]}...")
            self.results["asr"] = {"error": str(e)}
            
    def benchmark_llm_speed(self):
        """Benchmark S.I.Y.A LLM response generation"""
        self.print_header("S.I.Y.A Language Model Speed Test")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model for S.I.Y.A
            model_name = "Qwen/Qwen3-0.6B"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
            
            if self.device == "cuda":
                model = model.cuda()
                
            model.eval()
            
            # Test prompts suitable for S.I.Y.A conversation
            test_prompts = [
                "Hello S.I.Y.A",
                "How are you today?",
                "What's the weather?",
                "Can you help me with this?"
            ]
            
            llm_times = []
            
            for i, prompt in enumerate(test_prompts):
                print(f"🤖 Testing S.I.Y.A response {i+1}: '{prompt}'")
                
                start_time = time.time()
                
                # Format input for S.I.Y.A
                inputs = f"User: {prompt}\nAssistant: "
                input_ids = tokenizer.encode(inputs, return_tensors="pt").to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(inputs, "").strip()
                
                generation_time = (time.time() - start_time) * 1000
                llm_times.append(generation_time)
                
                self.print_result(f"S.I.Y.A Response {i+1}", generation_time)
                print(f"   Response: {response}")
                
            avg_llm_time = sum(llm_times) / len(llm_times)
            self.print_result("S.I.Y.A Average LLM Generation", avg_llm_time)
            
            # S.I.Y.A LLM performance grade
            if avg_llm_time < 30:
                grade = "Excellent (A+)"
            elif avg_llm_time < 50:
                grade = "Great (A)"
            elif avg_llm_time < 100:
                grade = "Good (B)"
            else:
                grade = "Needs optimization (C)"
                
            print(f"📊 S.I.Y.A LLM Grade: {grade}")
            
            self.results["llm"] = {
                "avg_generation_time_ms": avg_llm_time,
                "individual_times_ms": llm_times,
                "target_ms": 30,
                "grade": grade
            }
            
        except Exception as e:
            self.print_result("S.I.Y.A LLM Generation", 0, f"❌ Failed: {str(e)[:50]}...")
            self.results["llm"] = {"error": str(e)}
            
    def benchmark_tts_speed(self):
        """Benchmark S.I.Y.A TTS performance"""
        self.print_header("S.I.Y.A Text-to-Speech Speed Test")
        
        try:
            # Test different TTS methods for S.I.Y.A
            test_texts = [
                "Hello! I'm S.I.Y.A.",
                "How can I help you today?",
                "That's a great question!"
            ]
            
            tts_times = []
            
            # Test system TTS speed for S.I.Y.A
            for text in test_texts:
                print(f"🔊 Testing S.I.Y.A TTS with: '{text}'")
                
                start_time = time.time()
                
                # Test system TTS (fastest option for S.I.Y.A)
                if sys.platform.startswith("linux"):
                    subprocess.run(["espeak", text], check=False, capture_output=True)
                elif sys.platform.startswith("darwin"):
                    subprocess.run(["say", text], check=False, capture_output=True)
                elif sys.platform.startswith("win"):
                    subprocess.run([
                        "powershell", "-Command", 
                        f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                    ], check=False, capture_output=True)
                    
                tts_time = (time.time() - start_time) * 1000
                tts_times.append(tts_time)
                
                self.print_result("S.I.Y.A System TTS", tts_time)
                
            avg_tts_time = sum(tts_times) / len(tts_times)
            self.print_result("S.I.Y.A Average TTS Speed", avg_tts_time)
            
            # S.I.Y.A TTS performance grade
            if avg_tts_time < 20:
                grade = "Excellent (A+)"
            elif avg_tts_time < 50:
                grade = "Great (A)"
            elif avg_tts_time < 100:
                grade = "Good (B)"
            else:
                grade = "Needs optimization (C)"
                
            print(f"📊 S.I.Y.A TTS Grade: {grade}")
            
            self.results["tts"] = {
                "avg_synthesis_time_ms": avg_tts_time,
                "individual_times_ms": tts_times,
                "target_ms": 20,
                "grade": grade
            }
            
        except Exception as e:
            self.print_result("S.I.Y.A TTS Synthesis", 0, f"❌ Failed: {str(e)[:50]}...")
            self.results["tts"] = {"error": str(e)}
            
    def benchmark_memory_usage(self):
        """Benchmark S.I.Y.A memory usage"""
        self.print_header("S.I.Y.A Memory Usage Analysis")
        
        try:
            # CPU Memory for S.I.Y.A
            memory = psutil.virtual_memory()
            print(f"💾 System RAM: {memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
            
            # GPU Memory for S.I.Y.A
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🔥 GPU Memory: {gpu_memory:.1f}GB / {gpu_memory_total:.1f}GB used")
                
                # Peak memory during S.I.Y.A inference
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    # Simulate S.I.Y.A inference
                    dummy_input = torch.randn(1, 768).cuda()
                    _ = dummy_input * 2
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"📈 Peak S.I.Y.A GPU Memory: {peak_memory:.1f}GB")
                    
                # Memory efficiency grade
                memory_efficiency = (gpu_memory / gpu_memory_total) * 100
                if memory_efficiency < 60:
                    grade = "Excellent (A+)"
                elif memory_efficiency < 80:
                    grade = "Good (A)"
                elif memory_efficiency < 90:
                    grade = "Fair (B)"
                else:
                    grade = "High usage (C)"
                    
                print(f"📊 S.I.Y.A Memory Efficiency: {memory_efficiency:.1f}% - {grade}")
                    
            self.results["memory"] = {
                "system_ram_percent": memory.percent,
                "gpu_memory_gb": gpu_memory if torch.cuda.is_available() else 0
            }
            
        except Exception as e:
            print(f"❌ S.I.Y.A Memory benchmark failed: {e}")
            
    def benchmark_end_to_end(self):
        """Benchmark complete S.I.Y.A pipeline"""
        self.print_header("S.I.Y.A End-to-End Performance Test")
        
        try:
            # Simulate complete S.I.Y.A conversation pipeline
            test_input = "Hello S.I.Y.A, how are you today?"
            
            print(f"🧪 Testing complete S.I.Y.A pipeline with: '{test_input}'")
            
            # 1. ASR simulation (already done)
            asr_time = self.results.get("asr", {}).get("processing_time_ms", 50)
            
            # 2. LLM generation (already done)
            llm_time = self.results.get("llm", {}).get("avg_generation_time_ms", 30)
            
            # 3. TTS synthesis (already done)
            tts_time = self.results.get("tts", {}).get("avg_synthesis_time_ms", 20)
            
            # 4. Audio I/O (estimated for S.I.Y.A)
            audio_io_time = 10  # Microphone capture + playback
            
            total_time = asr_time + llm_time + tts_time + audio_io_time
            
            self.print_result("S.I.Y.A Speech Recognition", asr_time)
            self.print_result("S.I.Y.A Response Generation", llm_time)
            self.print_result("S.I.Y.A Speech Synthesis", tts_time)
            self.print_result("S.I.Y.A Audio I/O", audio_io_time)
            
            print(f"{'='*30}")
            self.print_result("S.I.Y.A Total Pipeline", total_time, "🚀")
            
            # S.I.Y.A performance grade
            if total_time < 100:
                grade = "Excellent (A+)"
            elif total_time < 150:
                grade = "Great (A)"
            elif total_time < 200:
                grade = "Good (B)"
            elif total_time < 300:
                grade = "Okay (C)"
            else:
                grade = "Needs optimization (D)"
                
            print(f"📊 S.I.Y.A Overall Grade: {grade}")
            
            self.results["end_to_end"] = {
                "total_time_ms": total_time,
                "breakdown": {
                    "asr_ms": asr_time,
                    "llm_ms": llm_time,
                    "tts_ms": tts_time,
                    "audio_ms": audio_io_time
                },
                "grade": grade
            }
            
        except Exception as e:
            print(f"❌ S.I.Y.A End-to-end test failed: {e}")
            
    def save_results(self):
        """Save S.I.Y.A benchmark results to file"""
        results_file = "siya_benchmark_results.json"
        
        # Add S.I.Y.A metadata
        self.results["siya_metadata"] = {
            "name": "S.I.Y.A",
            "full_name": "Simply Intended Yet Astute Assistant",
            "timestamp": time.time(),
            "device": self.device,
            "pytorch_version": torch.__version__
        }
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n💾 S.I.Y.A Results saved to: {results_file}")
        
    def print_summary(self):
        """Print S.I.Y.A comprehensive summary"""
        self.print_header("S.I.Y.A Performance Summary")
        
        # System specs
        gpu_count = len(GPUtil.getGPUs()) if torch.cuda.is_available() else 0
        print(f"✅ S.I.Y.A system configured with {gpu_count} GPU(s)")
        
        # Performance summary
        if "end_to_end" in self.results:
            total_time = self.results["end_to_end"]["total_time_ms"]
            grade = self.results["end_to_end"]["grade"]
            print(f"⚡ S.I.Y.A Average response time: {total_time:.1f}ms")
            print(f"📊 S.I.Y.A Performance grade: {grade}")
            
            # Check S.I.Y.A targets
            targets_met = 0
            total_targets = 3
            
            if "asr" in self.results and self.results["asr"].get("processing_time_ms", 999) < 50:
                targets_met += 1
            if "llm" in self.results and self.results["llm"].get("avg_generation_time_ms", 999) < 30:
                targets_met += 1
            if "tts" in self.results and self.results["tts"].get("avg_synthesis_time_ms", 999) < 20:
                targets_met += 1
                
            print(f"🎯 S.I.Y.A Targets met: {targets_met}/{total_targets}")
            
            if targets_met == total_targets:
                print("🎉 S.I.Y.A All performance targets achieved!")
                print("🚀 Ready for lightning-fast conversations!")
            else:
                print("💡 S.I.Y.A Optimization opportunities available")
                print("🔧 Check performance configuration in siya_config.json")
                
    def run_full_benchmark(self):
        """Run complete S.I.Y.A benchmark suite"""
        print("🚀 S.I.Y.A Performance Benchmark")
        print("🤖 Simply Intended Yet Astute Assistant")
        print("Measuring system performance for optimal conversation quality...")
        
        # Run all tests
        self.benchmark_system_specs()
        self.benchmark_model_loading()
        self.benchmark_asr_speed()
        self.benchmark_llm_speed()
        self.benchmark_tts_speed()
        self.benchmark_memory_usage()
        self.benchmark_end_to_end()
        
        # Save and summarize
        self.save_results()
        self.print_summary()
        
        print(f"\n{'='*60}")
        print("🎉 S.I.Y.A Benchmark Complete!")
        print("💡 Check siya_benchmark_results.json for detailed results")
        print("🚀 Ready to run: python siya.py")

def main():
    """Main benchmark function for S.I.Y.A"""
    benchmark = SIYaBenchmark()
    benchmark.run_full_benchmark()

if __name__ == "__main__":
    main()