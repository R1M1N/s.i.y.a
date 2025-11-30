#!/usr/bin/env python3
"""
S.I.Y.A Performance Monitor
Real-time performance tracking for sub-100ms response targets
Monitors ASR, LLM, TTS pipeline performance on RTX 4080
"""

import time
import psutil
import threading
from collections import deque, defaultdict
import json
import statistics
from datetime import datetime

class SIYAPerformanceMonitor:
    """Comprehensive performance monitoring for S.I.Y.A system"""
    
    def __init__(self, max_history=1000):
        """
        Initialize performance monitor
        
        Args:
            max_history: Maximum number of performance records to keep
        """
        self.max_history = max_history
        
        # Component timing data
        self.component_times = {
            'asr': deque(maxlen=max_history),
            'llm': deque(maxlen=max_history), 
            'tts': deque(maxlen=max_history),
            'total': deque(maxlen=max_history)
        }
        
        # Detailed performance metrics
        self.detailed_metrics = deque(maxlen=max_history)
        
        # Real-time tracking
        self.current_response = {}
        self.response_in_progress = False
        self.total_responses = 0
        self.successful_responses = 0
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        self.gpu_monitor = None
        
        # Performance thresholds (S.I.Y.A targets)
        self.thresholds = {
            'asr_ms': 50,    # <50ms for ASR
            'llm_ms': 30,    # <30ms for LLM  
            'tts_ms': 20,    # <20ms for TTS
            'total_ms': 100  # <100ms total
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def _check_gpu_availability(self):
        """Check if GPU monitoring is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except:
            return False
    
    def start_monitoring(self):
        """Start background monitoring threads"""
        # GPU monitoring thread
        if self.gpu_available:
            self.gpu_monitor = threading.Thread(target=self._gpu_monitoring_loop, daemon=True)
            self.gpu_monitor.start()
    
    def _gpu_monitoring_loop(self):
        """Background GPU monitoring loop"""
        try:
            import pynvml
            while True:
                # Get GPU stats
                gpu_info = self._get_gpu_stats()
                # Log or process GPU info as needed
                time.sleep(1)  # Update every second
        except:
            pass
    
    def _get_gpu_stats(self):
        """Get current GPU statistics"""
        if not self.gpu_available:
            return None
        
        try:
            import pynvml
            # Memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            memory_usage = {
                'used_mb': memory_info.used // 1024 // 1024,
                'total_mb': memory_info.total // 1024 // 1024,
                'usage_percent': (memory_info.used / memory_info.total) * 100
            }
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            gpu_utilization = {
                'gpu_percent': utilization.gpu,
                'memory_percent': utilization.memory
            }
            
            # Temperature and power
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
            except:
                temperature = None
                power_draw = None
            
            return {
                'timestamp': time.time(),
                'memory': memory_usage,
                'utilization': gpu_utilization,
                'temperature': temperature,
                'power_watts': power_draw
            }
        except:
            return None
    
    def start_response_tracking(self, user_input=None):
        """Start tracking a new response"""
        self.current_response = {
            'start_time': time.time(),
            'user_input': user_input,
            'asr_start': None,
            'asr_end': None,
            'llm_start': None,
            'llm_end': None,
            'tts_start': None,
            'tts_end': None,
            'end_time': None
        }
        self.response_in_progress = True
    
    def track_asr_start(self):
        """Mark ASR processing start"""
        if self.response_in_progress:
            self.current_response['asr_start'] = time.time()
    
    def track_asr_end(self):
        """Mark ASR processing end"""
        if self.response_in_progress:
            self.current_response['asr_end'] = time.time()
    
    def track_llm_start(self):
        """Mark LLM processing start"""
        if self.response_in_progress:
            self.current_response['llm_start'] = time.time()
    
    def track_llm_end(self):
        """Mark LLM processing end"""
        if self.response_in_progress:
            self.current_response['llm_end'] = time.time()
    
    def track_tts_start(self):
        """Mark TTS processing start"""
        if self.response_in_progress:
            self.current_response['tts_start'] = time.time()
    
    def track_tts_end(self):
        """Mark TTS processing end"""
        if self.response_in_progress:
            self.current_response['tts_end'] = time.time()
    
    def finish_response(self, response_text=None, success=True):
        """Finish response tracking and store metrics"""
        if not self.response_in_progress:
            return
        
        self.current_response['end_time'] = time.time()
        self.current_response['response_text'] = response_text
        self.current_response['success'] = success
        
        # Calculate component times
        asr_time = 0
        llm_time = 0
        tts_time = 0
        total_time = 0
        
        if self.current_response['asr_start'] and self.current_response['asr_end']:
            asr_time = (self.current_response['asr_end'] - self.current_response['asr_start']) * 1000
        
        if self.current_response['llm_start'] and self.current_response['llm_end']:
            llm_time = (self.current_response['llm_end'] - self.current_response['llm_start']) * 1000
        
        if self.current_response['tts_start'] and self.current_response['tts_end']:
            tts_time = (self.current_response['tts_end'] - self.current_response['tts_start']) * 1000
        
        if self.current_response['start_time'] and self.current_response['end_time']:
            total_time = (self.current_response['end_time'] - self.current_response['start_time']) * 1000
        
        # Store component times
        if asr_time > 0:
            self.component_times['asr'].append(asr_time)
        if llm_time > 0:
            self.component_times['llm'].append(llm_time)
        if tts_time > 0:
            self.component_times['tts'].append(tts_time)
        if total_time > 0:
            self.component_times['total'].append(total_time)
        
        # Store detailed metrics
        detailed_metrics = self.current_response.copy()
        detailed_metrics['asr_time_ms'] = asr_time
        detailed_metrics['llm_time_ms'] = llm_time
        detailed_metrics['tts_time_ms'] = tts_time
        detailed_metrics['total_time_ms'] = total_time
        
        self.detailed_metrics.append(detailed_metrics)
        
        # Update counters
        self.total_responses += 1
        if success:
            self.successful_responses += 1
        
        self.response_in_progress = False
    
    def get_performance_summary(self):
        """Get current performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_responses': self.total_responses,
            'successful_responses': self.successful_responses,
            'success_rate': (self.successful_responses / max(1, self.total_responses)) * 100,
            'components': {}
        }
        
        # Component performance
        for component in ['asr', 'llm', 'tts', 'total']:
            times = list(self.component_times[component])
            if times:
                summary['components'][component] = {
                    'count': len(times),
                    'average_ms': statistics.mean(times),
                    'median_ms': statistics.median(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p95_ms': statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                    'p99_ms': statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times),
                    'meets_target': component != 'total' or statistics.mean(times) < self.thresholds[f'{component}_ms']
                }
            else:
                summary['components'][component] = {
                    'count': 0,
                    'average_ms': 0,
                    'median_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0,
                    'p95_ms': 0,
                    'p99_ms': 0,
                    'meets_target': False
                }
        
        return summary
    
    def get_recent_performance(self, num_responses=10):
        """Get performance data for recent responses"""
        recent = list(self.detailed_metrics)[-num_responses:]
        
        if not recent:
            return None
        
        return {
            'responses': recent,
            'average_times': {
                'asr_ms': statistics.mean([r['asr_time_ms'] for r in recent if r['asr_time_ms'] > 0]),
                'llm_ms': statistics.mean([r['llm_time_ms'] for r in recent if r['llm_time_ms'] > 0]),
                'tts_ms': statistics.mean([r['tts_time_ms'] for r in recent if r['tts_time_ms'] > 0]),
                'total_ms': statistics.mean([r['total_time_ms'] for r in recent if r['total_time_ms'] > 0])
            }
        }
    
    def check_performance_health(self):
        """Check if S.I.Y.A performance is healthy"""
        summary = self.get_performance_summary()
        
        health_status = {
            'overall_healthy': True,
            'warnings': [],
            'critical_issues': []
        }
        
        # Check component performance
        for component, data in summary['components'].items():
            if data['count'] > 0:
                threshold_key = f'{component}_ms'
                threshold = self.thresholds.get(threshold_key)
                
                if threshold and data['average_ms'] > threshold:
                    if data['average_ms'] > threshold * 1.5:
                        health_status['critical_issues'].append(
                            f"{component.upper()} performance critical: {data['average_ms']:.1f}ms > {threshold}ms"
                        )
                        health_status['overall_healthy'] = False
                    else:
                        health_status['warnings'].append(
                            f"{component.upper()} performance slow: {data['average_ms']:.1f}ms > {threshold}ms"
                        )
        
        # Check success rate
        if summary['success_rate'] < 95:
            health_status['warnings'].append(
                f"Success rate low: {summary['success_rate']:.1f}%"
            )
        
        return health_status
    
    def export_performance_data(self, filename=None):
        """Export performance data to JSON file"""
        if filename is None:
            filename = f"siya_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'summary': self.get_performance_summary(),
            'detailed_metrics': list(self.detailed_metrics),
            'thresholds': self.thresholds,
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'platform': 'S.I.Y.A Performance Monitor'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

def test_performance_monitor():
    """Test performance monitor functionality"""
    print("🧪 Testing S.I.Y.A Performance Monitor...")
    
    monitor = SIYAPerformanceMonitor()
    
    # Simulate some responses
    for i in range(5):
        print(f"Simulating response {i+1}...")
        monitor.start_response_tracking(f"Test input {i+1}")
        
        # Simulate ASR processing
        monitor.track_asr_start()
        time.sleep(0.03)  # 30ms
        monitor.track_asr_end()
        
        # Simulate LLM processing
        monitor.track_llm_start()
        time.sleep(0.02)  # 20ms
        monitor.track_llm_end()
        
        # Simulate TTS processing
        monitor.track_tts_start()
        time.sleep(0.015)  # 15ms
        monitor.track_tts_end()
        
        monitor.finish_response(f"Test response {i+1}", success=True)
        time.sleep(0.1)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print("\n📊 Performance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Check health
    health = monitor.check_performance_health()
    print(f"\n🏥 Health Status: {'✅ Healthy' if health['overall_healthy'] else '⚠️ Issues Found'}")
    if health['warnings']:
        print("Warnings:", health['warnings'])
    if health['critical_issues']:
        print("Critical Issues:", health['critical_issues'])
    
    # Export data
    export_file = monitor.export_performance_data()
    print(f"\n💾 Performance data exported to: {export_file}")
    
    print("✅ Performance monitor test completed!")

if __name__ == "__main__":
    test_performance_monitor()