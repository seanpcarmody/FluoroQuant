"""
FluoroQuant - Batch Processing Module
Handles pattern detection and batch processing operations
"""

import os
import threading
from datetime import datetime


class BatchProcessor:
    """Batch processing with intelligent pattern detection"""
    
    def __init__(self):
        self.processing_thread = None
        self.stop_processing = False
        self.common_patterns = {
            'ch1': ['_ch1', '_c1', '_C1', 'ch01', 'C01', '_1.', '_red', '_r'],
            'ch2': ['_ch2', '_c2', '_C2', 'ch02', 'C02', '_2.', '_green', '_g'],
            'ch3': ['_ch3', '_c3', '_C3', 'ch03', 'C03', '_3.', '_blue', '_b']
        }
    
    def learn_pattern(self, channel, filepath, learned_patterns):
        """Learn naming pattern from loaded file"""
        filename = os.path.basename(filepath)
        learned_patterns[channel] = filename
        
    def extract_base_pattern(self, channel_filepaths, learned_patterns):
        """Learn naming pattern from loaded files"""
        loaded_channels = {ch: path for ch, path in channel_filepaths.items() if path is not None}
        
        if len(loaded_channels) < 2:
            return None
        
        filenames = {}
        for ch, path in loaded_channels.items():
            filename = os.path.basename(path)
            name_no_ext = os.path.splitext(filename)[0]
            filenames[ch] = name_no_ext
        
        channels = list(filenames.keys())
        if len(channels) >= 2:
            ch1, ch2 = channels[0], channels[1]
            name1, name2 = filenames[ch1], filenames[ch2]
            
            # Look for standard channel naming conventions
            channel_markers = ['channel1', 'channel2', 'channel3', 
                            'ch1', 'ch2', 'ch3', 
                            'c1', 'c2', 'c3',
                            '_1', '_2', '_3']
            
            pattern = {'prefix': '', 'suffix': '', 'channel_ids': {}}
            
            # Find channel identifiers
            for ch, name in [(ch1, name1), (ch2, name2)]:
                for marker in channel_markers:
                    if marker in name.lower():
                        idx = name.lower().find(marker)
                        pattern['channel_ids'][ch] = name[idx:idx+len(marker)]
                        if not pattern['prefix']:
                            pattern['prefix'] = name[:idx]
                        break
            
            if pattern['channel_ids']:
                print(f"Pattern found - Prefix: '{pattern['prefix']}', IDs: {pattern['channel_ids']}")
                return pattern
            
            # Fallback: find first difference
            for i in range(min(len(name1), len(name2))):
                if name1[i] != name2[i]:
                    pattern['prefix'] = name1[:i]
                    pattern['channel_ids'] = {
                        ch1: name1[i:],
                        ch2: name2[i:]
                    }
                    break
            
            return pattern
        
        return None
    
    def apply_pattern_to_folder(self, folder_path, base_pattern, active_channels):
        """Apply learned pattern to detect matching files in folder"""
        if not base_pattern:
            return {}
        
        files = os.listdir(folder_path)
        image_files = [f for f in files if self.is_image_file(f)]
        
        channel_ids = base_pattern['channel_ids']
        groups = {}
        
        for filename in image_files:
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Find which channel this file belongs to
            matched_channel = None
            base_name = None
            
            for ch, ch_id in channel_ids.items():
                if ch_id.lower() in filename_no_ext.lower():
                    matched_channel = ch
                    # The base name is everything before the channel identifier
                    idx = filename_no_ext.lower().find(ch_id.lower())
                    base_name = filename_no_ext[:idx].rstrip('_')
                    break
            
            if matched_channel and base_name:
                if base_name not in groups:
                    groups[base_name] = {'ch1': None, 'ch2': None, 'ch3': None}
                groups[base_name][matched_channel] = os.path.join(folder_path, filename)
        
        # Filter for active channels
        filtered_groups = {}
        for base_name, channels in groups.items():
            has_active = any(channels[ch] and active_channels.get(ch, False) 
                            for ch in ['ch1', 'ch2', 'ch3'])
            if has_active:
                filtered_groups[base_name] = channels
        
        return filtered_groups

    def detect_pattern_fallback(self, folder_path):
        """Fallback pattern detection using common patterns"""
        files = os.listdir(folder_path)
        image_files = [f for f in files if self.is_image_file(f)]
        
        groups = {}
        
        for filename in image_files:
            detected_channel = None
            base_name = filename
            
            # Try to detect channel using common patterns
            for channel, patterns in self.common_patterns.items():
                for pattern in patterns:
                    if pattern in filename.lower():
                        detected_channel = channel
                        # Remove pattern from base name
                        base_name = filename.replace(pattern, '')
                        base_name = base_name.replace(pattern.upper(), '')
                        base_name = base_name.replace(pattern.lower(), '')
                        break
                if detected_channel:
                    break
            
            # Remove extension
            base_name = os.path.splitext(base_name)[0]
            
            # Group files
            if base_name not in groups:
                groups[base_name] = {'ch1': None, 'ch2': None, 'ch3': None}
            
            if detected_channel:
                groups[base_name][detected_channel] = os.path.join(folder_path, filename)
            else:
                # If no pattern detected, assume it's channel 1
                if groups[base_name]['ch1'] is None:
                    groups[base_name]['ch1'] = os.path.join(folder_path, filename)
        
        return groups
    
    def process_batch(self, folder_path, base_pattern, active_channels,
                     processing_params, analysis_mode, multichannel_params,
                     processor, analyzer, progress_callback, complete_callback):
        """Process batch of images in separate thread"""
        self.stop_processing = False
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._batch_process_thread,
            args=(folder_path, base_pattern, active_channels,
                  processing_params, analysis_mode, multichannel_params,
                  processor, analyzer, progress_callback, complete_callback)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _batch_process_thread(self, folder_path, base_pattern, active_channels,
                             processing_params, analysis_mode, multichannel_params,
                             processor, analyzer, progress_callback, complete_callback):
        """Thread function for batch processing"""
        try:
            # Detect image groups
            if base_pattern:
                image_groups = self.apply_pattern_to_folder(folder_path, base_pattern, active_channels)
            else:
                image_groups = self.detect_pattern_fallback(folder_path)
            
            total_groups = len(image_groups)
            results = []
            
            # Process each group
            for i, (base_name, files) in enumerate(image_groups.items()):
                if self.stop_processing:
                    break
                
                # Update progress
                progress_callback(i, total_groups, f"Processing {base_name}")
                
                # Process single group
                group_result = self.process_single_group(
                    base_name, files, active_channels,
                    processing_params, analysis_mode, multichannel_params,
                    processor, analyzer
                )
                
                if group_result:
                    results.append(group_result)
            
            # Call completion callback
            complete_callback(results)
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            complete_callback([])
    
    def process_single_group(self, base_name, files, active_channels,
                           processing_params, analysis_mode, multichannel_params,
                           processor, analyzer):
        """Process a single group of channel images"""
        try:
            # Load images
            channels = {}
            for ch, filepath in files.items():
                if filepath and active_channels.get(ch, False):
                    image = processor.load_image(filepath)
                    if image is not None:
                        channels[ch] = image
            
            if not channels:
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
            # Single channel analysis
            active_for_analysis = {ch: ch in channels for ch in ['ch1', 'ch2', 'ch3']}
            single_results = analyzer.analyze_single_channels(
                active_for_analysis,
                labeled_objects,
                processed_channels,
                binary_masks,
                analysis_mode
            )
            
            # Multi-channel analysis
            multi_results = {}
            if len(channels) > 1:
                multi_results = analyzer.analyze_multichannel(
                    active_for_analysis,
                    processed_channels,
                    binary_masks,
                    labeled_objects,
                    multichannel_params
                )
            
            # Compile results
            return {
                'base_name': base_name,
                'files_processed': files,
                'channels_analyzed': list(channels.keys()),
                'single_channel_results': single_results,
                'multichannel_results': multi_results,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            return None
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def is_image_file(self, filename):
        """Check if file is a supported image format"""
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        return filename.lower().endswith(supported_extensions)