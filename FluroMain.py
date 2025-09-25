#!/usr/bin/env python3
"""
FluoroQuant - Multi-Channel Fluorescence Analysis
Version 2.0
"""

import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os
import multiprocessing as mp


# Import our modules
try:
    from fluoro_processor import ImageProcessor
    from fluoro_analyzer import MultiChannelAnalyzer
    from fluoro_gui import FluoroGUI
    from fluoro_batch import BatchProcessor, BatchConfig
    from fluoro_export import ResultExporter
    from fluoro_memory import LRUImageCache, MemoryMonitor

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all module files are in the same directory:")
    print("- fluoro_processor.py")
    print("- fluoro_analyzer.py") 
    print("- fluoro_gui.py")
    print("- fluoro_batch.py")
    print("- fluoro_export.py")
    exit(1)


class FluoroQuant:
    '''Main method to utilize flourescnet quantification analysis and results'''
    def __init__(self):
                
        # Configure batch processor for optimal performance
        batch_config = BatchConfig(
            max_workers=mp.cpu_count() - 1,  # Leave one core free
            memory_limit_gb=8.0,
            chunk_size=1,
            use_shared_memory=True
        )
        
        self.processor = ImageProcessor()
        self.analyzer = MultiChannelAnalyzer()
        self.exporter = ResultExporter()
        self.batch_processor = BatchProcessor(batch_config)
        self.memory_monitor = MemoryMonitor(threshold_percent=80)
        self.image_cache = LRUImageCache(max_memory_mb=512)
        self.gui = FluoroGUI(self)
        
        # Data storage for loaded images
        self.channels = {'ch1': None, 'ch2': None, 'ch3': None}
        self.channel_names = {'ch1': 'Channel 1', 'ch2': 'Channel 2', 'ch3': 'Channel 3'}
        self.active_channels = {'ch1': False, 'ch2': False, 'ch3': False}
        self.channel_filepaths = {'ch1': None, 'ch2': None, 'ch3': None}
        
        # Processing results storage
        self.processed_channels = {'ch1': None, 'ch2': None, 'ch3': None}
        self.binary_masks = {'ch1': None, 'ch2': None, 'ch3': None}
        self.labeled_objects = {'ch1': None, 'ch2': None, 'ch3': None}
        
        # Analysis results
        self.quantitative_results = {}
        self.multichannel_results = {}
        
        # Batch processing data
        self.batch_folder = None
        self.detected_groups = {}
        self.base_pattern = None
        self.learned_patterns = {}
        
        # Processing state
        self.is_processing = False
    
    def run(self):
        """Start the application main loop"""
        print("Starting FluoroQuant...")
        try:
            self.gui.create_main_window()
            self.gui.main_window.mainloop()
        except Exception as e:
            print(f"Application error: {e}")
            messagebox.showerror("Application Error", f"An error occurred: {str(e)}")
    
    def load_channel(self, channel, filepath):
        """Load image for a specific channel"""
        if self.is_processing:
            self.gui.show_warning("Processing in progress. Please wait.")
            return False
            
        try:
            # Load and validate image
            image = self.processor.load_image(filepath)
            if image is not None:
                # Store image and metadata
                self.channels[channel] = image
                self.channel_filepaths[channel] = filepath
                self.active_channels[channel] = True
                
                # Learn naming pattern for batch processing
                self.batch_processor.learn_pattern(channel, filepath, self.learned_patterns)
                self.update_base_pattern()  # This should enable the button
                
                # Update GUI
                self.gui.update_channel_status(channel, filepath)
                self.gui.enable_processing_buttons()
                
                loaded_count = sum(1 for fp in self.channel_filepaths.values() if fp is not None)
                if loaded_count >= 2:  # Add this check
                    self.gui.enable_batch_folder_selection()
                
                # Auto-process if enabled
                if self.gui.auto_update_var and self.gui.auto_update_var.get():
                    self.process_current_image()
                
                print(f"Successfully loaded {channel}: {os.path.basename(filepath)}")
                return True
            else:
                self.gui.show_error(f"Could not load image: {filepath}")
                return False
                
        except Exception as e:
            self.gui.show_error(f"Error loading image: {str(e)}")
            print(f"Load error for {channel}: {e}")
            return False
    
    
    def process_current_image(self, full_process=True):
        """Process loaded images with current parameters"""
        if self.is_processing:
            return
            
        if not any(self.active_channels.values()):
            self.gui.show_warning("No active channels to process.")
            return
            
        self.is_processing = True
        
        try:
            params = self.gui.get_processing_parameters()
            
            if full_process:
                print("Processing images...")
            
            # Process each active channel
            for channel in ['ch1', 'ch2', 'ch3']:
                if self.active_channels[channel] and self.channels[channel] is not None:
                    if full_process:
                        print(f"Processing {channel}...")
                    
                    # Preprocessing pipeline
                    processed = self.processor.preprocess_image(
                        self.channels[channel], 
                        params['preprocessing']
                    )
                    self.processed_channels[channel] = processed
                    
                    # Generate binary mask
                    binary = self.processor.threshold_image(
                        processed,
                        params['thresholding']
                    )
                    self.binary_masks[channel] = binary
                    
                    # Identify and label objects
                    labeled = self.processor.label_objects(
                        binary,
                        params['thresholding']['min_size'],
                        params['thresholding']['max_size']
                    )
                    self.labeled_objects[channel] = labeled
            
            # Run analysis only on full process
            if full_process:
                self.analyze_results()
                self.gui.update_statistics(self.quantitative_results, self.multichannel_results)
            
            self.gui.update_preview_display(self)
            
            if full_process:
                print("Processing complete.")
            
        except Exception as e:
            self.gui.show_error(f"Processing error: {str(e)}")
            print(f"Processing error: {e}")
        finally:
            self.is_processing = False
    
    def analyze_results(self):
        """Perform quantitative and multi-channel analysis"""
        try:
            # Single channel analysis
            print("Performing single-channel analysis...")
            self.quantitative_results = self.analyzer.analyze_single_channels(
                self.active_channels,
                self.labeled_objects,
                self.processed_channels,
                self.binary_masks,
                self.gui.get_analysis_mode()
            )
            print("Single-channel analysis complete.")
            
            # Multi-channel analysis if multiple channels active
            active_count = sum(1 for active in self.active_channels.values() if active)
            if active_count > 1:
                print("Performing multi-channel analysis...")
                
                # Debug
                for ch in ['ch1', 'ch2', 'ch3']:
                    if self.active_channels[ch]:
                        if self.processed_channels[ch] is not None:
                            print(f"{ch} processed shape: {self.processed_channels[ch].shape}")
                        if self.binary_masks[ch] is not None:
                            print(f"{ch} binary mask shape: {self.binary_masks[ch].shape}")
                
                params = self.gui.get_multichannel_parameters()
                self.multichannel_results = self.analyzer.analyze_multichannel(
                    self.active_channels,
                    self.processed_channels,
                    self.binary_masks,
                    self.labeled_objects,
                    params
                )
            else:
                self.multichannel_results = {}
                
            print("Analysis complete.")
            
        except Exception as e:
            print(f"Analysis error details: {e}")
            import traceback
            traceback.print_exc()
            self.gui.show_error(f"Analysis error: {str(e)}")
            
    def update_base_pattern(self):
        """Update base pattern from loaded channels"""
        loaded_files = {ch: fp for ch, fp in self.channel_filepaths.items() if fp is not None}
        
        if len(loaded_files) >= 2:
            self.base_pattern = self.batch_processor.extract_base_pattern(
                self.channel_filepaths, self.learned_patterns
            )
            
            if self.base_pattern:
                self.gui.update_pattern_display(self.base_pattern)
                
            # Enable batch folder selection regardless of pattern success
            self.gui.enable_batch_folder_selection()
            print(f"Base pattern updated, batch folder selection enabled")
    
    def select_batch_folder(self, folder_path):
        """Select folder for batch processing"""
        try:
            self.batch_folder = folder_path
            
            # Debug: Print current state
            print(f"Base pattern: {self.base_pattern}")
            print(f"Active channels: {self.active_channels}")
            print(f"Learned patterns: {self.learned_patterns}")
            
            # Detect image groups using learned pattern
            if self.base_pattern:
                print("Using learned pattern...")
                self.detected_groups = self.batch_processor.apply_pattern_to_folder(
                    folder_path, self.base_pattern, self.active_channels
                )
            else:
                print("Using fallback pattern detection...")
                self.detected_groups = self.batch_processor.detect_pattern_fallback(folder_path)
            #Debug:
            #print(f"Detected groups: {self.detected_groups}")
        except Exception as e:
            self.gui.show_error(f"Error selecting batch folder: {str(e)}")
            print(f"Batch folder selection error: {e}")
        """Select folder for batch processing"""
        try:
            self.batch_folder = folder_path
            
            # Detect image groups using learned pattern
            if self.base_pattern:
                self.detected_groups = self.batch_processor.apply_pattern_to_folder(
                    folder_path, self.base_pattern, self.active_channels
                )
            else:
                # Fallback pattern detection
                self.detected_groups = self.batch_processor.detect_pattern_fallback(folder_path)
            
            if self.detected_groups:
                n_groups = len(self.detected_groups)
                self.gui.enable_batch_processing(n_groups)
                print(f"Found {n_groups} image groups for batch processing")
                
                # Show preview if GUI method exists
                if hasattr(self.gui, 'show_batch_preview'):
                    self.gui.show_batch_preview(self.detected_groups)
            else:
                self.gui.show_warning("No matching image groups found in folder.")
                
        except Exception as e:
            self.gui.show_error(f"Error selecting batch folder: {str(e)}")
            print(f"Batch folder selection error: {e}")
    
    def start_batch_processing(self):
        """Start batch processing of detected image groups"""
        if not self.detected_groups:
            self.gui.show_warning("No image groups detected. Select a folder first.")
            return
            
        if self.is_processing:
            self.gui.show_warning("Processing already in progress.")
            return
            
        try:
            # Get processing parameters
            processing_params = self.gui.get_processing_parameters()
            analysis_mode = self.gui.get_analysis_mode()
            multichannel_params = self.gui.get_multichannel_parameters()
            
            # Start batch processing in separate thread
            self.batch_processor.process_batch(
                self.batch_folder,
                self.base_pattern,
                self.active_channels,
                processing_params,
                analysis_mode,
                multichannel_params,
                self.processor,
                self.analyzer,
                self.gui.update_batch_progress,
                self.batch_processing_complete
            )
            
            print("Batch processing started...")
            
        except Exception as e:
            self.gui.show_error(f"Error starting batch processing: {str(e)}")
            print(f"Batch processing error: {e}")
    
    def batch_processing_complete(self, results):
        """Handle batch processing completion"""
        try:
            n_processed = len(results)
            print(f"Batch processing complete. Processed {n_processed} groups.")
            
            # Auto-export if enabled
            if self.gui.auto_export_batch and self.gui.auto_export_batch.get():
                self.export_batch_results(results)
            
            # Update GUI
            self.gui.batch_processing_complete(n_processed)
            
        except Exception as e:
            print(f"Error handling batch completion: {e}")
    
    def export_results(self, export_type='current'):
        """Export analysis results"""
        try:
            if export_type == 'current':
                # Export current single image results
                if not self.quantitative_results and not self.multichannel_results:
                    if any(self.active_channels.values()):
                        self.process_current_image()
                        if not self.quantitative_results and not self.multichannel_results:
                            self.gui.show_warning("No results to export. Make sure images are loaded and processed.")
                            return
                    else:
                        self.gui.show_warning("No results to export. Load and process images first.")
                        return
                
                filepath = self.gui.get_save_filepath()
                if not filepath:
                    return
                
                # Get export options
                options = self.gui.get_export_options()
                
                # Export results
                exported_files = self.exporter.export_results(
                    filepath,
                    self.quantitative_results,
                    self.multichannel_results,
                    self.gui.get_processing_parameters(),
                    options,
                    self.processed_channels,
                    self.binary_masks,
                    self.active_channels
                )
                
                if exported_files:
                    self.gui.show_info(f"Exported {len(exported_files)} files successfully!\n\nFiles created:\n" + 
                                    "\n".join([os.path.basename(f) for f in exported_files]))
                    print(f"Exported files: {exported_files}")
                else:
                    self.gui.show_warning("No files were exported. Check your export settings.")
                
            elif export_type == 'batch':
                self.gui.show_info("Use batch processing mode to export batch results automatically.")
                
        except Exception as e:
            self.gui.show_error(f"Export error: {str(e)}")
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()

    def export_batch_results(self, batch_results):
        """Export batch processing results"""
        try:
            if not batch_results:
                return
                
            # Use batch folder as output location
            output_folder = self.batch_folder
            options = self.gui.get_export_options()
            parameters = self.gui.get_processing_parameters()
            
            exported_files = self.exporter.export_batch_results(
                output_folder, batch_results, parameters, options
            )
            
            if exported_files:
                print(f"Batch export complete: {len(exported_files)} files")
                
        except Exception as e:
            print(f"Batch export error: {e}")
    
    def stop_batch_processing(self):
        """Stop ongoing batch processing"""
        try:
            self.batch_processor.stop_batch_processing()
            print("Batch processing stopped.")
        except Exception as e:
            print(f"Error stopping batch processing: {e}")
    
    def reset_all(self):
        """Reset all data and GUI to initial state"""
        try:
            print("Resetting application...")
            
            # Stop any ongoing processing
            if self.is_processing:
                self.stop_batch_processing()
            
            # Clear all data
            self.channels = {'ch1': None, 'ch2': None, 'ch3': None}
            self.channel_filepaths = {'ch1': None, 'ch2': None, 'ch3': None}
            self.active_channels = {'ch1': False, 'ch2': False, 'ch3': False}
            self.processed_channels = {'ch1': None, 'ch2': None, 'ch3': None}
            self.binary_masks = {'ch1': None, 'ch2': None, 'ch3': None}
            self.labeled_objects = {'ch1': None, 'ch2': None, 'ch3': None}
            
            # Clear results
            self.quantitative_results = {}
            self.multichannel_results = {}
            
            # Clear batch data
            self.batch_folder = None
            self.detected_groups = {}
            self.base_pattern = None
            self.learned_patterns = {}
            
            # Reset processing state
            self.is_processing = False
            
            # Update GUI
            self.gui.reset_display()
            
            print("Reset complete.")
            
        except Exception as e:
            print(f"Reset error: {e}")
            self.gui.show_error(f"Reset error: {str(e)}")


def main():
    """Main entry point for the application"""
    print("=" * 60)
    print("FluoroQuant v2.0")
    print("Multi-Channel Fluorescence Analysis Platform")
    print("=" * 60)
    
    try:
        # Create and run application
        app = FluoroQuant()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
    finally:
        print("Application shutting down.")


if __name__ == "__main__":
    main()
