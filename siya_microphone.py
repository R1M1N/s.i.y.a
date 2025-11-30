#!/usr/bin/env python3
"""
S.I.Y.A Microphone Module
Optimized real-time audio capture with Voice Activity Detection
Designed for sub-100ms response times on RTX 4080
"""

import pyaudio
import numpy as np
import time
import threading
from collections import deque
import warnings

class FastMicrophone:
    """Ultra-fast microphone with Voice Activity Detection for S.I.Y.A"""
    
    def __init__(self, sample_rate=16000, chunk_size=800, channels=1):
        """
        Initialize fast microphone with optimized settings
        
        Args:
            sample_rate: Audio sample rate (16000 for ASR)
            chunk_size: Audio chunk size in samples
            channels: Number of audio channels (1 for mono)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32
        
        # Audio buffer for voice activity detection
        self.audio_buffer = deque(maxlen=10)  # ~500ms of audio
        self.is_recording = False
        self.vad_threshold = 0.01  # Voice activity threshold
        
        # Performance tracking
        self.latency_samples = []
        self.frame_count = 0
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # S.I.Y.A specific optimizations
        self.buffer_duration = chunk_size / sample_rate  # Duration per chunk
        
    def get_available_devices(self):
        """Get list of available audio devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
        return devices
    
    def start_recording(self, device_index=None):
        """Start optimized audio recording with VAD"""
        try:
            # Audio stream configuration
            stream_config = {
                'format': self.format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self._audio_callback
            }
            
            if device_index is not None:
                stream_config['input_device_index'] = device_index
            
            # Start stream
            self.stream = self.audio.open(**stream_config)
            self.is_recording = True
            
            print(f"🎤 S.I.Y.A Microphone started - Sample Rate: {self.sample_rate}Hz, Chunk: {self.chunk_size}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start microphone: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Optimized audio callback for low latency"""
        if status:
            print(f"Audio callback status: {status}")
        
        start_time = time.time()
        
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to buffer for voice activity detection
            self.audio_buffer.append(audio_data)
            
            # Voice Activity Detection
            if self._detect_voice_activity(audio_data):
                # Voice detected - this chunk should be processed
                self._process_voice_chunk(audio_data)
            
            # Track latency
            processing_time = time.time() - start_time
            self.latency_samples.append(processing_time)
            
            # Keep only recent latency samples
            if len(self.latency_samples) > 100:
                self.latency_samples.pop(0)
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _detect_voice_activity(self, audio_data):
        """Simple VAD using audio energy threshold"""
        # Calculate RMS (Root Mean Square) of audio
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Check if above voice activity threshold
        return rms > self.vad_threshold
    
    def _process_voice_chunk(self, audio_data):
        """Process voice chunk for S.I.Y.A"""
        # This would trigger the ASR processing
        # For now, just log that voice was detected
        if self.frame_count % 50 == 0:  # Print every 50th frame to avoid spam
            print(f"🗣️  Voice activity detected - Frame: {self.frame_count}")
    
    def get_audio_chunk(self, timeout=1.0):
        """
        Get audio chunk with timeout for S.I.Y.A processing
        
        Args:
            timeout: Maximum time to wait for audio chunk
            
        Returns:
            numpy array of audio data or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if len(self.audio_buffer) > 0:
                # Return the most recent audio chunk
                return np.concatenate(list(self.audio_buffer))
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        return None
    
    def get_performance_stats(self):
        """Get microphone performance statistics"""
        if not self.latency_samples:
            return None
        
        avg_latency = np.mean(self.latency_samples) * 1000  # Convert to ms
        max_latency = np.max(self.latency_samples) * 1000
        
        return {
            'average_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'total_frames': self.frame_count,
            'frames_per_second': self.frame_count / max(1, sum(self.latency_samples)),
            'buffer_size': len(self.audio_buffer),
            'chunk_duration_ms': self.buffer_duration * 1000
        }
    
    def stop_recording(self):
        """Stop recording and clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_recording = False
        self.audio.terminate()
        print("🎤 S.I.Y.A Microphone stopped")

class WSLMicrophone(FastMicrophone):
    """WSL-optimized microphone configuration"""
    
    def __init__(self):
        super().__init__(sample_rate=16000, chunk_size=800)
        
        # WSL-specific audio optimizations
        self.pulseaudio_mode = True
        self.wsl_audio_fix_applied = False
    
    def start_recording(self, device_index=None):
        """Start recording with WSL audio optimizations"""
        # Apply WSL audio fix if needed
        if not self.wsl_audio_fix_applied:
            self._apply_wsl_audio_fix()
        
        return super().start_recording(device_index)
    
    def _apply_wsl_audio_fix(self):
        """Apply WSL-specific audio configurations"""
        try:
            import subprocess
            # Set PulseAudio environment variables for WSL
            subprocess.run(['pulseaudio', '--start'], capture_output=True)
            print("🔧 WSL PulseAudio started")
            self.wsl_audio_fix_applied = True
        except Exception as e:
            print(f"⚠️  WSL audio fix failed: {e}")

def test_microphone():
    """Test microphone functionality"""
    print("🧪 Testing S.I.Y.A Microphone...")
    
    try:
        mic = WSLMicrophone()
        
        # Show available devices
        devices = mic.get_available_devices()
        print(f"Found {len(devices)} audio input devices:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']} ({device['channels']} channels, {device['sample_rate']}Hz)")
        
        # Start recording (will try default device)
        print("\n🎤 Starting microphone test (5 seconds)...")
        if mic.start_recording():
            time.sleep(5)  # Test for 5 seconds
            mic.stop_recording()
            
            # Show performance stats
            stats = mic.get_performance_stats()
            if stats:
                print(f"\n📊 Performance Stats:")
                print(f"  Average Latency: {stats['average_latency_ms']:.2f}ms")
                print(f"  Max Latency: {stats['max_latency_ms']:.2f}ms")
                print(f"  Total Frames: {stats['total_frames']}")
                print(f"  Frames/Second: {stats['frames_per_second']:.1f}")
            
            print("✅ Microphone test completed successfully!")
        else:
            print("❌ Failed to start microphone")
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")

if __name__ == "__main__":
    test_microphone()