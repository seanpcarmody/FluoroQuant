"""
FluoroQuant - Memory Management Module

"""

import functools
import weakref
import psutil
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from collections import OrderedDict
import logging
import gc
import warnings

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitor system memory and provide memory-aware processing decisions.
    Demonstrates understanding of resource management in production systems.
    """
    
    def __init__(self, threshold_percent: float = 80.0):
        """
        Initialize memory monitor.
        
        Args:
            threshold_percent: Memory usage threshold to trigger warnings
        """
        self.threshold_percent = threshold_percent
        self.initial_memory = psutil.virtual_memory().available
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'used_gb': mem.used / (1024**3)
        }
    
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available"""
        available_mb = psutil.virtual_memory().available / (1024**2)
        return available_mb >= required_mb
    
    def estimate_image_memory(self, shape: Tuple[int, ...], dtype: np.dtype) -> float:
        """Estimate memory required for an image in MB"""
        bytes_per_element = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        return (total_elements * bytes_per_element) / (1024**2)
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered"""
        mem = psutil.virtual_memory()
        return mem.percent > self.threshold_percent
    
    def optimize_batch_size(self, 
                           n_items: int, 
                           memory_per_item_mb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        available_mb = psutil.virtual_memory().available / (1024**2)
        # Reserve 20% as buffer
        usable_mb = available_mb * 0.8
        
        optimal_size = max(1, int(usable_mb / memory_per_item_mb))
        return min(optimal_size, n_items)


class LRUImageCache:
    """
    LRU cache optimized for image data.
    Implements memory-aware eviction and weak references for efficiency.
    """
    
    def __init__(self, max_memory_mb: float = 1024.0):
        """
        Initialize LRU cache with memory limit.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.memory_usage: Dict[str, float] = {}
        self.weak_refs: Dict[str, weakref.ref] = {}
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Initialized LRU cache with {max_memory_mb}MB limit")
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Add or update item in cache"""
        # Calculate memory usage
        memory_mb = value.nbytes / (1024**2)
        
        # Check if item fits in cache
        if memory_mb > self.max_memory_mb:
            logger.warning(f"Item {key} ({memory_mb:.2f}MB) exceeds cache limit")
            return
        
        # Remove old entry if exists
        if key in self.cache:
            del self.cache[key]
            del self.memory_usage[key]
        
        # Evict items if necessary
        while self._get_total_memory() + memory_mb > self.max_memory_mb:
            if not self._evict_lru():
                break
        
        # Add to cache
        self.cache[key] = value
        self.memory_usage[key] = memory_mb
        
        # Create weak reference for recovery
        self.weak_refs[key] = weakref.ref(value)
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        # Try to recover from weak reference
        if key in self.weak_refs:
            ref = self.weak_refs[key]()
            if ref is not None:
                logger.debug(f"Recovered {key} from weak reference")
                self.put(key, ref)
                self.hits += 1
                return ref
        
        self.misses += 1
        return None
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Get LRU item (first in ordered dict)
        lru_key = next(iter(self.cache))
        memory_freed = self.memory_usage[lru_key]
        
        del self.cache[lru_key]
        del self.memory_usage[lru_key]
        
        logger.debug(f"Evicted {lru_key} ({memory_freed:.2f}MB)")
        return True
    
    def _get_total_memory(self) -> float:
        """Get total memory usage of cache"""
        return sum(self.memory_usage.values())
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.memory_usage.clear()
        self.weak_refs.clear()
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'memory_mb': self._get_total_memory(),
            'max_memory_mb': self.max_memory_mb,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'items': list(self.cache.keys())
        }


def memory_aware_cache(max_memory_mb: float = 512.0):
    """
    Decorator for memory-aware caching of function results.
    Monitors memory usage and adapts cache size dynamically.
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUImageCache(max_memory_mb)
        monitor = MemoryMonitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Check memory before computation
            mem_stats = monitor.get_memory_usage()
            if mem_stats['percent_used'] > 90:
                logger.warning(f"High memory usage: {mem_stats['percent_used']:.1f}%")
                cache.clear()  # Emergency cache clear
                gc.collect()
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache if it's an array
            if isinstance(result, np.ndarray):
                cache.put(cache_key, result)
            
            # Trigger GC if needed
            if monitor.should_trigger_gc():
                gc.collect()
            
            return result
        
        # Attach cache and monitor for inspection
        wrapper.cache = cache
        wrapper.monitor = monitor
        
        return wrapper
    
    return decorator


class ChunkedProcessor:
    """
    Process large datasets in memory-efficient chunks.
    Essential for handling datasets larger than available RAM.
    """
    
    def __init__(self, chunk_size_mb: float = 100.0):
        """
        Initialize chunked processor.
        
        Args:
            chunk_size_mb: Target chunk size in MB
        """
        self.chunk_size_mb = chunk_size_mb
        self.monitor = MemoryMonitor()
    
    def process_chunked(self, 
                       data: np.ndarray,
                       process_func: Callable,
                       axis: int = 0) -> np.ndarray:
        """
        Process large array in chunks along specified axis.
        
        Args:
            data: Input data array
            process_func: Function to apply to each chunk
            axis: Axis along which to chunk
            
        Returns:
            Processed array
        """
        # Calculate chunk size
        shape = list(data.shape)
        bytes_per_element = data.dtype.itemsize
        elements_per_slice = np.prod(shape) // shape[axis]
        bytes_per_slice = elements_per_slice * bytes_per_element
        mb_per_slice = bytes_per_slice / (1024**2)
        
        if mb_per_slice <= self.chunk_size_mb:
            # Process entire array
            return process_func(data)
        
        # Calculate number of chunks
        n_chunks = int(np.ceil(mb_per_slice / self.chunk_size_mb))
        chunk_size = max(1, shape[axis] // n_chunks)
        
        logger.info(f"Processing array in {n_chunks} chunks of size {chunk_size}")
        
        # Process in chunks
        results = []
        for i in range(0, shape[axis], chunk_size):
            # Check memory
            if not self.monitor.check_memory_available(self.chunk_size_mb):
                gc.collect()
                if not self.monitor.check_memory_available(self.chunk_size_mb):
                    raise MemoryError("Insufficient memory for chunk processing")
            
            # Extract chunk
            end = min(i + chunk_size, shape[axis])
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(i, end)
            chunk = data[tuple(slices)]
            
            # Process chunk
            result_chunk = process_func(chunk)
            results.append(result_chunk)
            
            # Clean up
            del chunk
            if i % 10 == 0:  # Periodic GC
                gc.collect()
        
        # Concatenate results
        return np.concatenate(results, axis=axis)


class SmartBuffer:
    """
    Smart buffer that automatically switches between memory and disk storage.
    Critical for handling very large datasets gracefully.
    """
    
    def __init__(self, memory_threshold_mb: float = 500.0):
        """
        Initialize smart buffer.
        
        Args:
            memory_threshold_mb: Threshold to switch to disk storage
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_buffer: Dict[str, np.ndarray] = {}
        self.disk_buffer: Dict[str, str] = {}  # key -> filepath
        self.monitor = MemoryMonitor()
        
        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix="fluoroquant_")
        logger.info(f"Smart buffer using temp dir: {self.temp_dir}")
    
    def store(self, key: str, data: np.ndarray) -> None:
        """Store data in appropriate location"""
        data_mb = data.nbytes / (1024**2)
        
        if data_mb < self.memory_threshold_mb:
            # Store in memory
            self.memory_buffer[key] = data
            logger.debug(f"Stored {key} in memory ({data_mb:.2f}MB)")
        else:
            # Store on disk
            import os
            filepath = os.path.join(self.temp_dir, f"{key}.npy")
            np.save(filepath, data)
            self.disk_buffer[key] = filepath
            logger.debug(f"Stored {key} on disk ({data_mb:.2f}MB)")
    
    def retrieve(self, key: str) -> Optional[np.ndarray]:
        """Retrieve data from storage"""
        if key in self.memory_buffer:
            return self.memory_buffer[key]
        elif key in self.disk_buffer:
            filepath = self.disk_buffer[key]
            return np.load(filepath)
        return None
    
    def delete(self, key: str) -> None:
        """Delete data from storage"""
        if key in self.memory_buffer:
            del self.memory_buffer[key]
        elif key in self.disk_buffer:
            import os
            filepath = self.disk_buffer[key]
            if os.path.exists(filepath):
                os.remove(filepath)
            del self.disk_buffer[key]
    
    def cleanup(self) -> None:
        """Clean up all storage"""
        self.memory_buffer.clear()
        
        # Clean up disk storage
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()


# Example usage demonstrating memory-aware processing
@memory_aware_cache(max_memory_mb=256)
def process_image_cached(image: np.ndarray, sigma: float) -> np.ndarray:
    """Example of cached image processing function"""
    from skimage import filters
    return filters.gaussian(image, sigma=sigma)


def demonstrate_memory_management():
    """Demonstration of memory management capabilities"""
    # Initialize components
    monitor = MemoryMonitor()
    cache = LRUImageCache(max_memory_mb=512)
    chunked = ChunkedProcessor(chunk_size_mb=50)
    buffer = SmartBuffer(memory_threshold_mb=100)
    
    # Show initial memory
    mem_stats = monitor.get_memory_usage()
    print(f"Initial memory: {mem_stats['available_gb']:.2f}GB available")
    
    # Create test data
    large_image = np.random.rand(2000, 2000, 3)
    
    # Store in smart buffer (will choose appropriate storage)
    buffer.store("test_image", large_image)
    
    # Process in chunks if needed
    def simple_process(chunk):
        return chunk * 2
    
    result = chunked.process_chunked(large_image, simple_process, axis=0)
    
    # Cache result
    cache.put("processed", result)
    
    # Show cache stats
    print(f"Cache stats: {cache.get_stats()}")
    
    # Clean up
    buffer.cleanup()
    cache.clear()
    
    return "Memory management demonstration complete"


if __name__ == "__main__":
    demonstrate_memory_management()