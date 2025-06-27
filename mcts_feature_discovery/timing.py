"""
Comprehensive Timing Infrastructure for MCTS Feature Discovery

Provides decorators, context managers, and utilities for detailed performance
monitoring and visualization across all system components.
"""

import time
import functools
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class TimingData:
    """Individual timing measurement."""
    operation: str
    duration: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    thread_id: int = None
    memory_usage_mb: Optional[float] = None
    
    def __post_init__(self):
        if self.thread_id is None:
            self.thread_id = threading.get_ident()

@dataclass
class TimingStats:
    """Aggregated timing statistics."""
    operation: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_time: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, duration: float):
        """Update statistics with new timing."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.count
        self.last_time = duration
        self.recent_times.append(duration)
        
        # Update percentiles for recent times
        if len(self.recent_times) >= 5:
            sorted_times = sorted(self.recent_times)
            n = len(sorted_times)
            self.percentiles = {
                50: sorted_times[n//2],
                90: sorted_times[int(n*0.9)],
                95: sorted_times[int(n*0.95)]
            }

class TimingCollector:
    """Thread-safe timing data collector with advanced analytics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize timing collector."""
        self.config = config or {}
        self.enabled = self.config.get('logging', {}).get('log_timing', True)
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._raw_timings: List[TimingData] = []
        self._stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        
        # Configuration
        self.max_raw_timings = self.config.get('timing', {}).get('max_records', 10000)
        self.auto_cleanup_threshold = 0.8
        
        # Session tracking
        self.session_start = time.time()
        self.session_stats = {
            'total_operations': 0,
            'total_time': 0.0,
            'operations_per_minute': 0.0
        }
        
        logger.debug("Initialized TimingCollector")
    
    def record(self, operation: str, duration: float, context: Dict[str, Any] = None, memory_mb: float = None):
        """Record timing data."""
        if not self.enabled:
            return
        
        timestamp = time.time()
        timing = TimingData(
            operation=operation,
            duration=duration,
            timestamp=timestamp,
            context=context or {},
            memory_usage_mb=memory_mb
        )
        
        with self._lock:
            # Add to raw timings
            self._raw_timings.append(timing)
            
            # Update statistics
            if operation not in self._stats:
                self._stats[operation] = TimingStats(operation=operation)
            self._stats[operation].update(duration)
            
            # Update session stats
            self.session_stats['total_operations'] += 1
            self.session_stats['total_time'] += duration
            elapsed_minutes = (timestamp - self.session_start) / 60
            if elapsed_minutes > 0:
                self.session_stats['operations_per_minute'] = self.session_stats['total_operations'] / elapsed_minutes
            
            # Auto cleanup if needed
            if len(self._raw_timings) > self.max_raw_timings * self.auto_cleanup_threshold:
                self._cleanup_old_timings()
    
    def _cleanup_old_timings(self):
        """Remove old timing records to manage memory."""
        keep_count = int(self.max_raw_timings * 0.7)  # Keep 70%
        removed_count = len(self._raw_timings) - keep_count
        
        # Keep most recent timings
        self._raw_timings = self._raw_timings[-keep_count:]
        
        logger.debug(f"Cleaned up {removed_count} old timing records")
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get timing statistics."""
        with self._lock:
            if operation:
                if operation in self._stats:
                    return {
                        'operation': operation,
                        'stats': self._stats[operation],
                        'recent_timings': list(self._stats[operation].recent_times)
                    }
                else:
                    return None
            else:
                # Return all statistics
                return {
                    'operations': {op: stats for op, stats in self._stats.items()},
                    'session': self.session_stats,
                    'total_records': len(self._raw_timings),
                    'elapsed_time': time.time() - self.session_start
                }
    
    def get_top_operations(self, metric: str = 'total_time', limit: int = 10) -> List[Dict[str, Any]]:
        """Get top operations by specified metric."""
        with self._lock:
            operations = []
            for op, stats in self._stats.items():
                operations.append({
                    'operation': op,
                    'total_time': stats.total_time,
                    'avg_time': stats.avg_time,
                    'count': stats.count,
                    'max_time': stats.max_time
                })
            
            # Sort by metric
            if metric in ['total_time', 'avg_time', 'count', 'max_time']:
                operations.sort(key=lambda x: x[metric], reverse=True)
            
            return operations[:limit]
    
    def get_timing_history(self, operation: str = None, last_n: int = 100) -> List[Dict[str, Any]]:
        """Get recent timing history."""
        with self._lock:
            timings = []
            for timing in self._raw_timings[-last_n:]:
                if operation is None or timing.operation == operation:
                    timings.append({
                        'operation': timing.operation,
                        'duration': timing.duration,
                        'timestamp': timing.timestamp,
                        'context': timing.context,
                        'memory_mb': timing.memory_usage_mb
                    })
            return timings
    
    def export_timings(self, format: str = 'json') -> str:
        """Export timing data for visualization."""
        stats = self.get_stats()
        
        if format == 'json':
            # Convert TimingStats objects to serializable format
            serializable_stats = {}
            for op, stats_obj in stats['operations'].items():
                serializable_stats[op] = {
                    'operation': stats_obj.operation,
                    'count': stats_obj.count,
                    'total_time': stats_obj.total_time,
                    'min_time': stats_obj.min_time if stats_obj.min_time != float('inf') else 0,
                    'max_time': stats_obj.max_time,
                    'avg_time': stats_obj.avg_time,
                    'last_time': stats_obj.last_time,
                    'percentiles': stats_obj.percentiles,
                    'recent_times': list(stats_obj.recent_times)
                }
            
            export_data = {
                'operations': serializable_stats,
                'session': stats['session'],
                'metadata': {
                    'total_records': stats['total_records'],
                    'elapsed_time': stats['elapsed_time'],
                    'export_timestamp': time.time()
                }
            }
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset(self):
        """Reset all timing data."""
        with self._lock:
            self._raw_timings.clear()
            self._stats.clear()
            self.session_start = time.time()
            self.session_stats = {
                'total_operations': 0,
                'total_time': 0.0,
                'operations_per_minute': 0.0
            }
        logger.info("Timing data reset")

# Global timing collector instance
_global_collector: Optional[TimingCollector] = None

def initialize_timing(config: Dict[str, Any] = None) -> TimingCollector:
    """Initialize global timing collector."""
    global _global_collector
    _global_collector = TimingCollector(config)
    return _global_collector

def get_timing_collector() -> Optional[TimingCollector]:
    """Get the global timing collector."""
    return _global_collector

def record_timing(operation: str, duration: float, context: Dict[str, Any] = None, memory_mb: float = None):
    """Record timing using global collector."""
    if _global_collector:
        _global_collector.record(operation, duration, context, memory_mb)

# Decorators and Context Managers

def timed(operation: str = None, include_memory: bool = False):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = None
            
            if include_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                memory_delta = None
                
                if include_memory and memory_before is not None:
                    try:
                        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_delta = memory_after - memory_before
                    except:
                        pass
                
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                record_timing(op_name, duration, context, memory_delta)
        
        return wrapper
    return decorator

@contextmanager
def timing_context(operation: str, context: Dict[str, Any] = None, include_memory: bool = False):
    """Context manager for timing code blocks."""
    start_time = time.time()
    memory_before = None
    
    if include_memory:
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        memory_delta = None
        
        if include_memory and memory_before is not None:
            try:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = memory_after - memory_before
            except:
                pass
        
        record_timing(operation, duration, context, memory_delta)

# Performance monitoring utilities

def performance_monitor(interval: float = 10.0) -> Dict[str, Any]:
    """Get current performance snapshot."""
    collector = get_timing_collector()
    if not collector:
        return {}
    
    stats = collector.get_stats()
    top_operations = collector.get_top_operations()
    
    return {
        'timestamp': time.time(),
        'session_stats': stats['session'],
        'top_operations_by_time': top_operations,
        'total_operations_tracked': len(stats['operations']),
        'memory_info': _get_memory_info()
    }

def _get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except ImportError:
        return {'error': 'psutil not available'}

# Timing analysis utilities

def analyze_timing_patterns(operation: str = None) -> Dict[str, Any]:
    """Analyze timing patterns for performance insights."""
    collector = get_timing_collector()
    if not collector:
        return {}
    
    if operation:
        stats = collector.get_stats(operation)
        if not stats:
            return {}
        
        # Analyze single operation
        recent_times = list(stats['stats'].recent_times)
        if len(recent_times) < 5:
            return {'error': 'Insufficient data for analysis'}
        
        # Detect trends
        trend = 'stable'
        if len(recent_times) >= 10:
            first_half = recent_times[:len(recent_times)//2]
            second_half = recent_times[len(recent_times)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            change = (avg_second - avg_first) / avg_first
            if change > 0.1:
                trend = 'increasing'
            elif change < -0.1:
                trend = 'decreasing'
        
        return {
            'operation': operation,
            'trend': trend,
            'stability': max(recent_times) / min(recent_times) if min(recent_times) > 0 else float('inf'),
            'recent_avg': sum(recent_times) / len(recent_times),
            'total_stats': stats['stats'].__dict__
        }
    
    else:
        # Analyze all operations
        all_stats = collector.get_stats()
        operations_analysis = {}
        
        for op_name in all_stats['operations'].keys():
            operations_analysis[op_name] = analyze_timing_patterns(op_name)
        
        return {
            'all_operations': operations_analysis,
            'session_summary': all_stats['session']
        }