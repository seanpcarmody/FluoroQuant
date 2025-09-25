"""
FluoroQuant - Batch Processing Module
"""

import os
import logging
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from functools import partial
import numpy as np
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_workers: int = None  # None = use cpu_count()
    chunk_size: int = 1
    memory_limit_gb: float = 8.0
    use_shared_memory: bool = True
    prefetch_count: int = 2
    fallback_to_threading: bool = True  # Fallback if multiprocessing fails

def _test_multiprocessing_worker():
    """Simple test function for multiprocessing capability"""
    return "test_success"

class BatchProcessor:
    """
    Unified batch processor with multiprocessing as primary approach
    """
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.stop_processing = False
        self.processing_thread = None
        
        # Determine optimal worker count
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count() - 1, 8)
        
        logger.info(f"Initialized batch processor with {self.config.max_workers} workers")
        
        # Pattern detection cache
        self.pattern_cache = {}
        self.common_patterns = {
            'ch1': ['_ch1', '_c1', '_C1', 'ch01', 'C01', '_1.', '_red', '_r'],
            'ch2': ['_ch2', '_c2', '_C2', 'ch02', 'C02', '_2.', '_green', '_g'],
            'ch3': ['_ch3', '_c3', '_C3', 'ch03', 'C03', '_3.', '_blue', '_b']
        }
    
    def process_batch(
        self,
        folder_path: str,
        base_pattern: Dict[str, Any],
        active_channels: Dict[str, bool],
        processing_params: Dict[str, Any],
        analysis_mode: str,
        multichannel_params: Dict[str, Any],
        processor,
        analyzer,
        progress_callback: Optional[Callable] = None,
        complete_callback: Optional[Callable] = None,
        use_multiprocessing: bool = True
    ) -> None:
        """
        Unified batch processing method with multiprocessing as primary approach.
        
        Args:
            use_multiprocessing: If True, use multiprocessing. If False or fails, fall back to threading.
        """
        self.stop_processing = False
        
        # Start processing in background thread 
        self.processing_thread = threading.Thread(
            target=self._unified_batch_worker,
            args=(folder_path, base_pattern, active_channels, processing_params,
                  analysis_mode, multichannel_params, processor, analyzer,
                  progress_callback, complete_callback, use_multiprocessing),
            daemon=True
        )
        self.processing_thread.start()
    
    def _unified_batch_worker(
        self,
        folder_path: str,
        base_pattern: Dict[str, Any],
        active_channels: Dict[str, bool],
        processing_params: Dict[str, Any],
        analysis_mode: str,
        multichannel_params: Dict[str, Any],
        processor,
        analyzer,
        progress_callback: Optional[Callable],
        complete_callback: Optional[Callable],
        use_multiprocessing: bool
    ) -> None:
        """Unified worker that chooses between multiprocessing and threading"""
        try:
            # Detect image groups
            if base_pattern:
                image_groups = self.apply_pattern_to_folder(folder_path, base_pattern, active_channels)
            else:
                image_groups = self.detect_pattern_fallback(folder_path)
            
            if not image_groups:
                logger.warning("No image groups detected")
                if complete_callback:
                    complete_callback([])
                return
            
            logger.info(f"Processing {len(image_groups)} groups")
            
            # Choose processing method
            results = []
            if use_multiprocessing and self._can_use_multiprocessing():
                try:
                    results = self._process_with_multiprocessing(
                        image_groups, active_channels, processing_params,
                        analysis_mode, multichannel_params, processor.__class__,
                        analyzer.__class__, progress_callback
                    )
                    logger.info(f"Multiprocessing completed: {len(results)} results")
                except Exception as e:
                    logger.warning(f"Multiprocessing failed: {e}")
                    if self.config.fallback_to_threading:
                        logger.info("Falling back to threading approach")
                        results = self._process_with_threading(
                            image_groups, active_channels, processing_params,
                            analysis_mode, multichannel_params, processor,
                            analyzer, progress_callback
                        )
                    else:
                        raise
            else:
                results = self._process_with_threading(
                    image_groups, active_channels, processing_params,
                    analysis_mode, multichannel_params, processor,
                    analyzer, progress_callback
                )
            
            if complete_callback:
                complete_callback(results)
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            traceback.print_exc()
            if complete_callback:
                complete_callback([])
    
    def _can_use_multiprocessing(self) -> bool:
        """Check if multiprocessing can be used safely"""
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_test_multiprocessing_worker)
                result = future.result(timeout=5)
                return result == "test_success"
        except Exception as e:
            logger.warning(f"Multiprocessing test failed: {e}")
            return False
    
    def _process_with_multiprocessing(
        self,
        image_groups: Dict[str, Dict[str, str]],
        active_channels: Dict[str, bool],
        processing_params: Dict[str, Any],
        analysis_mode: str,
        multichannel_params: Dict[str, Any],
        processor_class: type,
        analyzer_class: type,
        progress_callback: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """Process using multiprocessing for true parallelism"""
        
        # Prepare work items
        work_items = [
            (base_name, files, active_channels, processing_params,
             analysis_mode, multichannel_params, processor_class, analyzer_class)
            for base_name, files in image_groups.items()
        ]
        
        total_groups = len(work_items)
        results = []
        processed = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_item = {
                executor.submit(self._process_single_group_worker, item): item
                for item in work_items
            }
            
            # Process results as they complete
            for future in as_completed(future_to_item):
                if self.stop_processing:
                    logger.info("Batch processing stopped by user")
                    executor.shutdown(wait=False)
                    break
                
                try:
                    result = future.result(timeout=60)  # 60 second timeout per group
                    if result:
                        results.append(result)
                    
                    processed += 1
                    if progress_callback:
                        item = future_to_item[future]
                        progress_callback(processed, total_groups, 
                                        f"Processed {item[0]}")
                        
                except Exception as e:
                    item = future_to_item[future]
                    logger.error(f"Error processing {item[0]}: {e}")
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_groups, 
                                        f"Failed {item[0]}")
        
        return results
    
    def _process_with_threading(
        self,
        image_groups: Dict[str, Dict[str, str]],
        active_channels: Dict[str, bool],
        processing_params: Dict[str, Any],
        analysis_mode: str,
        multichannel_params: Dict[str, Any],
        processor,
        analyzer,
        progress_callback: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """Process using threading (sequential with I/O parallelism)"""
        
        results = []
        total_groups = len(image_groups)
        
        for i, (base_name, files) in enumerate(image_groups.items()):
            if self.stop_processing:
                logger.info("Batch processing stopped by user")
                break
            
            if progress_callback:
                progress_callback(i, total_groups, f"Processing {base_name}")
            
            try:
                result = self._process_single_group_static(
                    base_name, files, active_channels, processing_params,
                    analysis_mode, multichannel_params, processor, analyzer
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {base_name}: {e}")
        
        return results
    
    @staticmethod
    def _process_single_group_worker(args: Tuple) -> Optional[Dict[str, Any]]:
        """
        Worker function for multiprocessing.
        Creates processor and analyzer instances in the worker process.
        """
        (base_name, files, active_channels, processing_params,
         analysis_mode, multichannel_params, processor_class, analyzer_class) = args
        
        try:
            # Create instances in worker process
            processor = processor_class()
            analyzer = analyzer_class()
            
            return BatchProcessor._process_single_group_static(
                base_name, files, active_channels, processing_params,
                analysis_mode, multichannel_params, processor, analyzer
            )
        except Exception as e:
            logger.error(f"Worker error processing {base_name}: {e}")
            return None
    
    @staticmethod
    def _process_single_group_static(
        base_name: str,
        files: Dict[str, str],
        active_channels: Dict[str, bool],
        processing_params: Dict[str, Any],
        analysis_mode: str,
        multichannel_params: Dict[str, Any],
        processor: Any,
        analyzer: Any
    ) -> Optional[Dict[str, Any]]:
        """Process single group - static method for both multiprocessing and threading"""
        try:
            start_time = datetime.now()
            
            # Load images with memory management
            channels = {}
            for ch, filepath in files.items():
                if filepath and active_channels.get(ch, False):
                    try:
                        image = processor.load_image(filepath)
                        if image is not None:
                            channels[ch] = image
                    except MemoryError:
                        logger.error(f"Memory error loading {filepath}")
                        return None
            
            if not channels:
                logger.warning(f"No channels loaded for {base_name}")
                return None
            
            # Process images
            processed_channels = {}
            binary_masks = {}
            labeled_objects = {}
            
            for ch, image in channels.items():
                # Preprocess
                processed = processor.preprocess_image(image, processing_params['preprocessing'])
                processed_channels[ch] = processed
                
                # Threshold
                binary = processor.threshold_image(processed, processing_params['thresholding'])
                binary_masks[ch] = binary
                
                # Label objects
                labeled = processor.label_objects(
                    binary,
                    processing_params['thresholding']['min_size'],
                    processing_params['thresholding']['max_size']
                )
                labeled_objects[ch] = labeled
            
            # Analyze
            active_for_analysis = {ch: ch in channels for ch in ['ch1', 'ch2', 'ch3']}
            
            single_results = analyzer.analyze_single_channels(
                active_for_analysis,
                labeled_objects,
                processed_channels,
                binary_masks,
                analysis_mode
            )
            
            multi_results = {}
            if len(channels) > 1:
                multi_results = analyzer.analyze_multichannel(
                    active_for_analysis,
                    processed_channels,
                    binary_masks,
                    labeled_objects,
                    multichannel_params
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'base_name': base_name,
                'files_processed': files,
                'channels_analyzed': list(channels.keys()),
                'single_channel_results': single_results,
                'multichannel_results': multi_results,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing {base_name}: {e}")
            traceback.print_exc()
            return None
    
    def stop_batch_processing(self):
        """Stop batch processing gracefully"""
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Stopping batch processing...")
            self.processing_thread.join(timeout=2.0)
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not stop cleanly")
        logger.info("Batch processing stopped")
    
    # Keep all existing pattern detection methods unchanged
    def learn_pattern(self, channel: str, filepath: str, learned_patterns: Dict[str, str]):
        """Learn naming pattern from loaded file with caching"""
        filename = os.path.basename(filepath)
        learned_patterns[channel] = filename
        
        # Debug
        print(f"Learning pattern for {channel}: {filename}")
        
        # Cache the pattern for future use
        self.pattern_cache[channel] = self._extract_pattern_features(filename)
        logger.debug(f"Learned pattern for {channel}: {filename}")
    
    def _extract_pattern_features(self, filename: str) -> Dict[str, Any]:
        """Extract pattern features for intelligent matching"""
        name_no_ext = os.path.splitext(filename)[0]
        
        features = {
            'length': len(name_no_ext),
            'has_numbers': any(c.isdigit() for c in name_no_ext),
            'has_underscore': '_' in name_no_ext,
            'has_dash': '-' in name_no_ext,
            'position_markers': []
        }
        
        # Find position of channel markers
        for markers in self.common_patterns.values():
            for marker in markers:
                if marker in name_no_ext.lower():
                    pos = name_no_ext.lower().find(marker)
                    features['position_markers'].append((marker, pos))
        
        return features
    
    def extract_base_pattern(self, channel_filepaths, learned_patterns):
        """Simplified pattern extraction with proper prefix handling"""
        loaded_channels = {ch: path for ch, path in channel_filepaths.items() if path is not None}
        
        if len(loaded_channels) < 2:
            return None
        
        pattern = {
            'prefix': '',
            'suffix': '',
            'channel_ids': {}
        }
        
        for ch, path in loaded_channels.items():
            filename = os.path.basename(path)
            name_no_ext = os.path.splitext(filename)[0]
            
            # Look for common patterns
            found_marker = None
            for marker in ['channel1', 'channel2', 'channel3', 'ch1', 'ch2', 'ch3', 'c1', 'c2', 'c3', '_1', '_2', '_3']:
                if marker in name_no_ext.lower():
                    found_marker = marker
                    # Extract prefix
                    idx = name_no_ext.lower().find(marker.lower())
                    current_prefix = name_no_ext[:idx]
                    if not pattern['prefix']:  
                        pattern['prefix'] = current_prefix
                    break
            
            if found_marker:
                pattern['channel_ids'][ch] = found_marker
                print(f"Found marker for {ch}: {found_marker}")
            else:
                # Fallback - use channel name
                pattern['channel_ids'][ch] = ch
                print(f"Using fallback marker for {ch}: {ch}")
        
        print(f"Final pattern: {pattern}")
        return pattern if pattern['channel_ids'] else None
    
    def apply_pattern_to_folder(
        self,
        folder_path: str,
        base_pattern: Dict[str, Any],
        active_channels: Dict[str, bool]
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Apply pattern with improved matching algorithm"""
        if not base_pattern:
            return {}
        
        try:
            files = os.listdir(folder_path)
        except OSError as e:
            logger.error(f"Error reading folder {folder_path}: {e}")
            return {}
        
        image_files = [f for f in files if self.is_image_file(f)]
        
        groups = self._group_files_by_pattern(image_files, base_pattern, folder_path)
        
        # Filter for active channels
        filtered_groups = {}
        for base_name, channels in groups.items():
            has_active = any(channels[ch] and active_channels.get(ch, False) 
                           for ch in ['ch1', 'ch2', 'ch3'])
            if has_active:
                filtered_groups[base_name] = channels
        
        logger.info(f"Found {len(filtered_groups)} matching groups in {folder_path}")
        return filtered_groups
    
    def _group_files_by_pattern(
        self,
        files: List[str],
        pattern: Dict[str, Any],
        folder_path: str
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Group files using pattern matching"""
        groups = {}
        channel_ids = pattern.get('channel_ids', {})
        
        for filename in files:
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Find which channel this file belongs to
            matched_channel = None
            base_name = None
            
            for ch, ch_id in channel_ids.items():
                if ch_id.lower() in filename_no_ext.lower():
                    matched_channel = ch
                    idx = filename_no_ext.lower().find(ch_id.lower())
                    base_name = filename_no_ext[:idx].rstrip('_-')
                    break
            
            if matched_channel and base_name:
                if base_name not in groups:
                    groups[base_name] = {'ch1': None, 'ch2': None, 'ch3': None}
                groups[base_name][matched_channel] = os.path.join(folder_path, filename)
        
        return groups
    
    def detect_pattern_fallback(self, folder_path: str) -> Dict[str, Dict[str, Optional[str]]]:
        """Fallback pattern detection with improved heuristics"""
        try:
            files = os.listdir(folder_path)
            print(f"Fallback detection on files: {files}")
        except OSError as e:
            logger.error(f"Error reading folder {folder_path}: {e}")
            return {}
        
        image_files = [f for f in files if self.is_image_file(f)]
        print(f"Image files for fallback: {image_files}")
        
        groups = {}
        
        for filename in image_files:
            detected_channel = None
            base_name = filename
            
            print(f"Processing file: {filename}")
            
            # Try to detect channel using common patterns
            for channel, patterns in self.common_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in filename.lower():
                        detected_channel = channel
                        import re
                        base_name = re.sub(re.escape(pattern), '', filename, flags=re.IGNORECASE)
                        break
                if detected_channel:
                    break
            
            if not detected_channel:
                print(f"No pattern detected for {filename}, trying numeric patterns...")
                import re
                if re.search(r'[_\-]?1[_\-\.]', filename):
                    detected_channel = 'ch1'
                elif re.search(r'[_\-]?2[_\-\.]', filename):
                    detected_channel = 'ch2'
                elif re.search(r'[_\-]?3[_\-\.]', filename):
                    detected_channel = 'ch3'
            
            base_name = os.path.splitext(base_name)[0].strip('_- ')
            if not base_name:
                base_name = os.path.splitext(filename)[0]
            
            print(f"Base name: {base_name}, Channel: {detected_channel}")
            
            # Group files
            if base_name not in groups:
                groups[base_name] = {'ch1': None, 'ch2': None, 'ch3': None}
            
            if detected_channel:
                groups[base_name][detected_channel] = os.path.join(folder_path, filename)
            else:
                # Assign to first available channel
                for ch in ['ch1', 'ch2', 'ch3']:
                    if groups[base_name][ch] is None:
                        groups[base_name][ch] = os.path.join(folder_path, filename)
                        print(f"Assigned {filename} to {ch} as fallback")
                        break
        
        print(f"Final groups from fallback: {groups}")
        return groups
    
    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Check if file is a supported image format"""
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        return filename.lower().endswith(supported_extensions)
