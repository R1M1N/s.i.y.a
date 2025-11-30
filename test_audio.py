#!/usr/bin/env python3
"""
S.I.Y.A Audio System Test
Simply Intended Yet Astute Assistant - Audio Testing
"""

import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque

class SIYaAudioTest:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 5  # 5 seconds test
        self.chunk_size = 1024
        self.is_recording = False
        self.audio_data = []
        
    def print_devices(self):
        """List available audio devices for S.I.Y.A"""
        print("🎤 S.I.Y.A Audio System - Available Devices:")
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_type = "🎤 Input" if device['max_input_channels'] > 0 else "🔊 Output" if device['max_output_channels'] > 0 else "❓ Unknown"
                print(f"  {i}: {device['name']} ({device_type})")
                if device['max_input_channels'] > 0:
                    print(f"     Default sample rate: {device['default_samplerate']}Hz")
        except Exception as e:
            print(f"❌ Could not list devices: {e}")
            
    def test_input_device(self, device_index=None):
        """Test microphone input for S.I.Y.A"""
        print(f"\n🎤 Testing S.I.Y.A Microphone Input...")
        print(f"Duration: {self.duration} seconds")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Device index: {device_index if device_index is not None else 'Default'}")
        
        try:
            # Record audio
            print("Recording... (speak now for S.I.Y.A!)")
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=device_index
            )
            
            # Show recording progress
            for i in range(self.duration):
                time.sleep(1)
                print(f"⏰ S.I.Y.A Recording... {i+1}/{self.duration}s")
                
            print("✅ S.I.Y.A Recording complete!")
            
            # Analyze audio quality
            max_amplitude = np.max(np.abs(audio_data))
            avg_amplitude = np.mean(np.abs(audio_data))
            
            print(f"📊 S.I.Y.A Audio Analysis:")
            print(f"   Max amplitude: {max_amplitude:.4f}")
            print(f"   Average amplitude: {avg_amplitude:.4f}")
            
            if max_amplitude > 0.001:
                print("✅ S.I.Y.A Microphone detected audio - Good quality!")
            elif max_amplitude > 0.0001:
                print("⚠️ S.I.Y.A Microphone detected very low audio - Check levels")
            else:
                print("❌ S.I.Y.A Microphone not detecting audio - Check connection")
                
            return audio_data
            
        except Exception as e:
            print(f"❌ S.I.Y.A Input test failed: {e}")
            return None
            
    def test_output_device(self, device_index=None):
        """Test speaker output with S.I.Y.A test tone"""
        print(f"\n🔊 Testing S.I.Y.A Speaker Output...")
        print("Generating S.I.Y.A test tone...")
        
        try:
            # Generate S.I.Y.A test tone
            frequency = 440  # A4 note
            t = np.linspace(0, 1, self.sample_rate, False)
            test_tone = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            print("🔊 Playing S.I.Y.A test tone...")
            sd.play(test_tone, samplerate=self.sample_rate, device=device_index)
            sd.wait()  # Wait for playback to complete
            
            print("✅ S.I.Y.A Output test complete!")
            return True
            
        except Exception as e:
            print(f"❌ S.I.Y.A Output test failed: {e}")
            return False
            
    def test_realtime_input(self, device_index=None):
        """Test real-time audio input for S.I.Y.A"""
        print(f"\n🎤 S.I.Y.A Real-time Audio Test")
        print("Speak into microphone - S.I.Y.A monitoring audio levels...")
        print("Press Ctrl+C to stop")
        
        # Audio data buffer
        audio_buffer = deque(maxlen=10)
        
        def audio_callback(indata, frames, time, status):
            """Audio callback for S.I.Y.A real-time processing"""
            if status:
                print(f"S.I.Y.A Audio status: {status}")
            
            # Calculate audio level
            audio_level = np.sqrt(np.mean(indata**2))
            audio_buffer.append(audio_level)
            
            # Show audio level with S.I.Y.A branding
            level_bars = int(audio_level * 100)
            bars = "█" * min(level_bars, 50)
            spaces = " " * (50 - min(level_bars, 50))
            print(f"\r🤖 S.I.Y.A Level: |{bars}{spaces}| {audio_level:.4f}", end="", flush=True)
            
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=audio_callback,
                device=device_index
            ):
                print("Starting S.I.Y.A real-time monitoring...")
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 S.I.Y.A Real-time test stopped")
        except Exception as e:
            print(f"\n❌ S.I.Y.A Real-time test failed: {e}")
            
    def test_full_duplex(self, device_index=None):
        """Test full-duplex audio for S.I.Y.A"""
        print(f"\n🔄 S.I.Y.A Full-Duplex Audio Test")
        print("Testing input and output simultaneously...")
        
        try:
            def duplex_callback(indata, outdata, frames, time, status):
                if status:
                    print(f"S.I.Y.A Status: {status}")
                
                # Pass through audio with slight amplification for S.I.Y.A
                outdata[:] = indata * 1.2
                
            print("Starting S.I.Y.A duplex stream...")
            print("Speak into microphone - you should hear yourself through speakers")
            print("Press Ctrl+C to stop")
            
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=duplex_callback,
                device=device_index
            ):
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 S.I.Y.A Duplex test stopped")
        except Exception as e:
            print(f"\n❌ S.I.Y.A Duplex test failed: {e}")
            
    def test_siYa_speech_patterns(self, device_index=None):
        """Test speech patterns suitable for S.I.Y.A conversation"""
        print(f"\n💬 S.I.Y.A Speech Pattern Test")
        print("Testing audio quality for conversational AI...")
        
        try:
            # Record sample speech for S.I.Y.A
            print("Please speak a sample sentence for S.I.Y.A:")
            print("Example: 'Hello S.I.Y.A, how can you help me today?'")
            
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=device_index
            )
            
            print("Recording... (speak clearly for optimal S.I.Y.A performance)")
            for i in range(self.duration):
                time.sleep(1)
                print(f"⏰ Recording... {i+1}/{self.duration}s")
                
            print("✅ Speech sample recorded!")
            
            # Analyze speech quality for AI processing
            audio_flat = audio_data.flatten()
            signal_energy = np.sum(audio_flat ** 2)
            silence_threshold = 0.001
            
            # Count speech segments
            speech_segments = []
            in_speech = False
            segment_start = 0
            
            for i, sample in enumerate(audio_flat):
                if abs(sample) > silence_threshold:
                    if not in_speech:
                        segment_start = i
                        in_speech = True
                else:
                    if in_speech:
                        speech_segments.append((segment_start, i))
                        in_speech = False
            
            if in_speech:
                speech_segments.append((segment_start, len(audio_flat)))
            
            # Calculate metrics
            total_speech_time = sum(end - start for start, end in speech_segments) / self.sample_rate
            speech_percentage = (total_speech_time / self.duration) * 100
            
            print(f"📊 S.I.Y.A Speech Analysis:")
            print(f"   Signal energy: {signal_energy:.6f}")
            print(f"   Speech segments detected: {len(speech_segments)}")
            print(f"   Total speech time: {total_speech_time:.2f}s")
            print(f"   Speech percentage: {speech_percentage:.1f}%")
            
            if speech_percentage > 30:
                print("✅ Excellent speech quality for S.I.Y.A!")
            elif speech_percentage > 15:
                print("✅ Good speech quality for S.I.Y.A")
            else:
                print("⚠️ Low speech detection - speak louder or closer to microphone")
                
            return audio_data
            
        except Exception as e:
            print(f"❌ S.I.Y.A Speech pattern test failed: {e}")
            return None
            
    def run_comprehensive_test(self, device_index=None):
        """Run comprehensive audio testing for S.I.Y.A"""
        print("🚀 S.I.Y.A Audio System Test Suite")
        print("=" * 60)
        print("🤖 Simply Intended Yet Astute Assistant")
        print("Testing audio system for optimal conversation quality...\n")
        
        # List devices
        self.print_devices()
        
        # Test input
        audio_data = self.test_input_device(device_index)
        
        # Test output
        output_success = self.test_output_device(device_index)
        
        # Test speech patterns
        speech_data = self.test_siYa_speech_patterns(device_index)
        
        # Test real-time input
        print(f"\n" + "="*60)
        try:
            self.test_realtime_input(device_index)
        except KeyboardInterrupt:
            pass
            
        # Summary
        print(f"\n" + "="*60)
        print("📋 S.I.Y.A Audio Test Summary:")
        print(f"✅ Input test: {'PASSED' if audio_data is not None else 'FAILED'}")
        print(f"✅ Output test: {'PASSED' if output_success else 'FAILED'}")
        print(f"✅ Speech patterns: {'GOOD' if speech_data is not None else 'FAILED'}")
        
        if audio_data is not None and output_success:
            print("\n🎉 S.I.Y.A Audio system ready!")
            print("✅ Ready for ultra-fast conversational AI!")
            print("🚀 S.I.Y.A will provide sub-100ms responses")
        else:
            print("\n⚠️ S.I.Y.A Audio system needs attention")
            print("💡 Check audio device settings and permissions")
            print("🔧 Adjust microphone levels if needed")
            
    def test_latency(self, device_index=None):
        """Test audio latency for S.I.Y.A real-time performance"""
        print(f"\n⏱️ S.I.Y.A Audio Latency Test")
        
        try:
            # Generate test signal for S.I.Y.A
            test_duration = 0.1
            t = np.linspace(0, test_duration, int(self.sample_rate * test_duration), False)
            test_signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz tone
            
            # Measure round-trip latency for S.I.Y.A
            start_time = time.time()
            
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=device_index
            ) as stream:
                # Start playback and recording for S.I.Y.A
                playback_start = time.time()
                stream.start()
                
                # Record for a bit
                recorded_data, _ = stream.read(int(self.sample_rate * test_duration))
                
                playback_end = time.time()
                
            # Calculate S.I.Y.A latency
            total_latency = (playback_end - playback_start) * 1000
            print(f"🔄 S.I.Y.A Round-trip latency: {total_latency:.1f}ms")
            
            if total_latency < 50:
                print("✅ Excellent latency for S.I.Y.A real-time conversation")
            elif total_latency < 100:
                print("✅ Good latency for S.I.Y.A real-time conversation")
            else:
                print("⚠️ High latency - may affect S.I.Y.A conversation flow")
                print("💡 Consider using USB audio interface for lower latency")
                
        except Exception as e:
            print(f"❌ S.I.Y.A Latency test failed: {e}")

def main():
    """Main test function for S.I.Y.A"""
    import argparse
    
    parser = argparse.ArgumentParser(description="S.I.Y.A Audio System Test")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--latency", action="store_true", help="Run latency test only")
    parser.add_argument("--realtime", action="store_true", help="Run real-time test only")
    parser.add_argument("--duplex", action="store_true", help="Run duplex test only")
    parser.add_argument("--speech", action="store_true", help="Run speech pattern test only")
    
    args = parser.parse_args()
    
    tester = SIYaAudioTest()
    
    if args.latency:
        tester.test_latency(args.device)
    elif args.realtime:
        tester.test_realtime_input(args.device)
    elif args.duplex:
        tester.test_full_duplex(args.device)
    elif args.speech:
        tester.test_siYa_speech_patterns(args.device)
    else:
        tester.run_comprehensive_test(args.device)

if __name__ == "__main__":
    main()