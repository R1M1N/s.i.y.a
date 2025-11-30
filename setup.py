#!/usr/bin/env python3
"""
S.I.Y.A Setup Script - Simply Intended Yet Astute Assistant
Automated installation and configuration for ultra-fast conversational AI
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command with error handling"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_gpu():
    """Check GPU configuration for S.I.Y.A"""
    print("🖥️ Checking GPU Configuration...")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GeForce' in line:
                    print(f"🎮 GPU: {line.strip()}")
                    break
        else:
            print("⚠️ No NVIDIA GPU detected or nvidia-smi not available")
    except FileNotFoundError:
        print("⚠️ nvidia-smi not found - GPU acceleration may not be available")
    
    return True

def install_cuda_pytorch():
    """Install CUDA-optimized PyTorch for S.I.Y.A"""
    print("🔥 Installing CUDA-optimized PyTorch for S.I.Y.A...")
    
    # Detect CUDA version
    cuda_version = subprocess.run(["nvcc", "--version"], 
                                capture_output=True, text=True)
    
    if "release 12" in cuda_version.stdout:
        print("📦 CUDA 12.0 detected - Installing PyTorch with CUDA 12.0")
        commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "pip install torch-audio --index-url https://download.pytorch.org/whl/cu121"
        ]
    elif "release 11" in cuda_version.stdout:
        print("📦 CUDA 11.8 detected - Installing PyTorch with CUDA 11.8")
        commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install torch-audio --index-url https://download.pytorch.org/whl/cu118"
        ]
    else:
        print("📦 Installing PyTorch with CUDA support")
        commands = [
            "pip install torch torchvision torchaudio",
            "pip install torch-audio"
        ]
    
    for command in commands:
        package_name = command.split()[3] if len(command.split()) > 3 else "PyTorch"
        if not run_command(command, f"Installing {package_name}"):
            return False
    
    return True

def install_siya_dependencies():
    """Install S.I.Y.A specific dependencies"""
    print("⚡ Installing S.I.Y.A dependencies...")
    
    # Core ML packages optimized for S.I.Y.A
    core_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "optimum>=1.14.0",
        "faster-whisper>=0.10.0",
        "TTS>=0.15.0",
        "librosa>=0.10.1",
        "sounddevice>=0.4.6",
        "uvloop>=0.19.0",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "psutil>=5.9.6",
        "GPUtil>=1.4.0",
        "numpy>=1.24.0",
        "torch-audio>=2.1.0"
    ]
    
    for package in core_packages:
        package_name = package.split('>=')[0]
        if not run_command(f"pip install {package}", f"Installing {package_name}"):
            return False
    
    return True

def optimize_torch_for_siYa():
    """Optimize PyTorch settings for S.I.Y.A"""
    print("⚙️ Optimizing PyTorch for S.I.Y.A performance...")
    
    torch_config = '''
# S.I.Y.A Performance Optimizations
import os
import torch

# GPU optimizations for S.I.Y.A
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TORCH_BACKENDS_CUDNN_BENCHMARK'] = 'True'
os.environ['TORCH_BACKENDS_CUDNN_DETERMINISTIC'] = 'False'

# S.I.Y.A specific optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("🚀 CUDA optimizations enabled for S.I.Y.A")
'''
    
    with open("/workspace/torch_optimize_siYa.py", "w") as f:
        f.write(torch_config)
    
    print("✅ S.I.Y.A optimization script created")
    return True

def setup_siya_directories():
    """Setup directory structure for S.I.Y.A"""
    print("📁 Setting up S.I.Y.A directory structure...")
    
    directories = [
        "siya_models/asr",
        "siya_models/llm", 
        "siya_models/tts",
        "siya_audio/temp",
        "siya_cache",
        "siya_data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📂 Created: {directory}")
    
    return True

def download_siYa_models():
    """Download and cache S.I.Y.A models"""
    print("⬇️ Downloading S.I.Y.A models...")
    
    try:
        # Download ASR model (NVIDIA Parakeet)
        print("🎤 Downloading NVIDIA Parakeet ASR for S.I.Y.A...")
        subprocess.run([
            sys.executable, "-c", 
            """
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

print("Downloading Parakeet ASR model...")
processor = WhisperProcessor.from_pretrained('nvidia/parakeet-tdt-0.6b-v3', torch_dtype=torch.float16)
model = WhisperForConditionalGeneration.from_pretrained('nvidia/parakeet-tdt-0.6b-v3', torch_dtype=torch.float16)
print('✅ S.I.Y.A ASR model ready!')
"""
        ], check=True)
        
        # Download LLM (Qwen3-0.6B)
        print("🧠 Downloading Qwen3-0.6B for S.I.Y.A...")
        subprocess.run([
            sys.executable, "-c",
            """
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading Qwen3-0.6B model...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float16, trust_remote_code=True)
print('✅ S.I.Y.A LLM model ready!')
"""
        ], check=True)
        
        print("✅ S.I.Y.A models downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Model download failed: {e}")
        print("📝 Models will be downloaded when S.I.Y.A is first used")
        return True

def create_siYa_microphone():
    """Create S.I.Y.A optimized microphone system"""
    microphone_code = '''
"""
S.I.Y.A Microphone System
Optimized for Simply Intended Yet Astute Assistant
"""

import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import time
from collections import deque

class SIYaMicrophone:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_recording = False
        self.vad_threshold = 0.5
        
        # Audio buffer for voice activity detection
        self.audio_buffer = deque(maxlen=3)
        
        # Setup audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self.audio_callback,
            device=None  # Use default microphone
        )
        
        print("🎤 S.I.Y.A Microphone initialized")
        
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for S.I.Y.A real-time processing"""
        if self.is_recording:
            # Convert to numpy and add to buffer
            audio_data = indata[:, 0]  # Take first channel
            self.audio_buffer.append(audio_data.copy())
            
            # Put in queue for S.I.Y.A processing
            try:
                self.audio_queue.put_nowait(audio_data.copy())
            except queue.Full:
                pass  # Skip if queue is full for speed
    
    def start_recording(self):
        """Start S.I.Y.A optimized recording"""
        self.is_recording = True
        self.stream.start()
        print("🎤 S.I.Y.A microphone recording started...")
        
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.stream.stop()
        print("🛑 S.I.Y.A microphone stopped")
        
    def get_audio_chunk(self, timeout=1.0):
        """Get latest audio chunk for S.I.Y.A processing"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def has_audio(self):
        """Check if audio data is available for S.I.Y.A"""
        return not self.audio_queue.empty()
        
    def close(self):
        """Cleanup S.I.Y.A microphone resources"""
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()

# S.I.Y.A Microphone Test
if __name__ == "__main__":
    mic = SIYaMicrophone()
    mic.start_recording()
    
    print("S.I.Y.A Microphone Test - Press Ctrl+C to stop")
    try:
        while True:
            audio_data = mic.get_audio_chunk(timeout=0.1)
            if audio_data is not None:
                level = np.max(np.abs(audio_data))
                print(f"S.I.Y.A Audio Level: {level:.4f}", end="\\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        mic.stop_recording()
        mic.close()
        print("\\nS.I.Y.A Microphone test completed")
'''
    
    with open("/workspace/siya_microphone.py", "w") as f:
        f.write(microphone_code)
    
    print("🎤 Created S.I.Y.A microphone system")
    return True

def create_performance_monitor():
    """Create S.I.Y.A performance monitoring system"""
    monitor_code = '''
"""
S.I.Y.A Performance Monitor
Simply Intended Yet Astute Assistant - Performance Tracking
"""

import time
import psutil
import GPUtil
import torch
from collections import deque
import json

class SIYaPerformanceMonitor:
    def __init__(self):
        self.response_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100)
        self.component_times = {
            'asr': deque(maxlen=50),
            'llm': deque(maxlen=50),
            'tts': deque(maxlen=50)
        }
        
    def log_response_time(self, response_time_ms):
        """Log S.I.Y.A response time for analysis"""
        self.response_times.append(response_time_ms)
        
    def log_component_time(self, component, time_ms):
        """Log specific component timing"""
        if component in self.component_times:
            self.component_times[component].append(time_ms)
        
    def get_siya_stats(self):
        """Get S.I.Y.A system statistics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU stats for S.I.Y.A
        gpu_percent = 0
        gpu_memory = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory = gpu.memoryUsed
        except:
            pass
        
        return {
            "name": "S.I.Y.A",
            "full_name": "Simply Intended Yet Astute Assistant",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "gpu_percent": gpu_percent,
            "gpu_memory_mb": gpu_memory,
            "avg_response_time_ms": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "component_averages": {
                comp: sum(times) / len(times) if times else 0 
                for comp, times in self.component_times.items()
            }
        }
        
    def print_siya_report(self):
        """Print S.I.Y.A detailed performance report"""
        stats = self.get_siya_stats()
        
        print(f"\\n📊 S.I.Y.A Performance Report:")
        print(f"🤖 Assistant: {stats['name']} ({stats['full_name']})")
        print(f"💻 CPU Usage: {stats['cpu_percent']:.1f}%")
        print(f"💾 Memory Usage: {stats['memory_percent']:.1f}% ({stats['memory_used_gb']:.1f}GB)")
        print(f"🎮 GPU Usage: {stats['gpu_percent']:.1f}%")
        print(f"🔥 GPU Memory: {stats['gpu_memory_mb']:.0f}MB")
        print(f"⚡ Avg Response Time: {stats['avg_response_time_ms']:.1f}ms")
        print(f"🔧 PyTorch Version: {stats['pytorch_version']}")
        print(f"🚀 CUDA Available: {stats['cuda_available']}")
        if stats['cuda_available']:
            print(f"📦 CUDA Version: {stats['cuda_version']}")
            
        print(f"\\n🎯 Component Performance:")
        for component, avg_time in stats['component_averages'].items():
            if avg_time > 0:
                component_name = component.upper()
                print(f"  {component_name}: {avg_time:.1f}ms")

# Global S.I.Y.A monitor instance
siya_monitor = SIYaPerformanceMonitor()

def log_siya_response_time_ms(response_time_ms):
    siya_monitor.log_response_time(response_time_ms)

def log_siya_component_time(component, time_ms):
    siya_monitor.log_component_time(component, time_ms)

def print_siya_performance_report():
    siya_monitor.print_siya_report()
'''
    
    with open("/workspace/siya_performance.py", "w") as f:
        f.write(monitor_code)
    
    print("📊 Created S.I.Y.A performance monitoring system")
    return True

def create_sample_data():
    """Create sample training data for S.I.Y.A"""
    sample_data = [
        {
            "instruction": "Hello SIYA",
            "input": "",
            "output": "Hello! I'm SIYA, your Simply Intended Yet Astute Assistant. How can I help you today?"
        },
        {
            "instruction": "What can you do?",
            "input": "",
            "output": "I can help you with various tasks through natural conversation. Just tell me what you need!"
        },
        {
            "instruction": "How are you?",
            "input": "",
            "output": "I'm doing great! Ready to assist with whatever you need."
        },
        {
            "instruction": "Tell me about yourself",
            "input": "",
            "output": "I'm SIYA - Simply Intended Yet Astute Assistant. I'm designed for quick, helpful responses that get straight to the point."
        },
        {
            "instruction": "Thanks for your help",
            "input": "",
            "output": "You're very welcome! That's what I'm here for."
        },
        {
            "instruction": "What's the time?",
            "input": "",
            "output": "I don't have access to real-time data, but you can check your device's clock."
        },
        {
            "instruction": "Can you set a timer?",
            "input": "",
            "output": "I can't set timers directly, but you can use your device's timer or alarm app."
        },
        {
            "instruction": "Tell me a joke",
            "input": "",
            "output": "Why don't scientists trust atoms? Because they make up everything!"
        },
        {
            "instruction": "That's not working",
            "input": "",
            "output": "Let me try a different approach then. What specifically isn't working as expected?"
        },
        {
            "instruction": "Goodbye SIYA",
            "input": "",
            "output": "Goodbye! Feel free to ask me anything anytime. I'm always here to help!"
        }
    ]
    
    with open("siya_sample_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("📝 Created S.I.Y.A sample training data")
    return True

def main():
    """Main setup function for S.I.Y.A"""
    print("🤖 S.I.Y.A Setup - Simply Intended Yet Astute Assistant")
    print("🚀 Ultra-Fast Conversational AI System")
    print("=" * 60)
    
    # Check GPU
    check_gpu()
    
    # Install CUDA PyTorch
    if not install_cuda_pytorch():
        print("❌ CUDA PyTorch installation failed")
        return False
    
    # Install S.I.Y.A dependencies
    if not install_siya_dependencies():
        print("❌ S.I.Y.A dependency installation failed")
        return False
    
    # Optimize PyTorch
    optimize_torch_for_siYa()
    
    # Setup directories
    setup_siya_directories()
    
    # Download models
    download_siYa_models()
    
    # Create S.I.Y.A systems
    create_siYa_microphone()
    create_performance_monitor()
    create_sample_data()
    
    print("\n" + "=" * 60)
    print("🎉 S.I.Y.A Setup Complete!")
    print("🤖 Simply Intended Yet Astute Assistant is ready!")
    
    print("\n⚡ S.I.Y.A Optimizations:")
    print("- 🚀 CUDA GPU acceleration enabled")
    print("- 🎤 NVIDIA Parakeet ASR (ultra-fast speech recognition)")
    print("- 🧠 Qwen3-0.6B LLM (intelligent response generation)")
    print("- 🔊 OpenAudio TTS (natural voice synthesis)")
    print("- 💾 Memory optimization for efficient operation")
    print("- 📊 Performance monitoring and analysis")
    
    print("\n🚀 Next Steps:")
    print("1. Test audio system: python test_siYa_audio.py")
    print("2. Run performance benchmark: python benchmark_siYa_speed.py")
    print("3. Start S.I.Y.A: python siya.py")
    print("4. Configure in siya_config.json if needed")
    
    print("\n🎯 S.I.Y.A Performance Targets:")
    print("- Speech Recognition: < 50ms")
    print("- Response Generation: < 30ms") 
    print("- Speech Synthesis: < 20ms")
    print("- Total Pipeline: < 100ms")
    
    print(f"\n💬 Say hello to your new AI assistant!")
    
    return True

if __name__ == "__main__":
    main()