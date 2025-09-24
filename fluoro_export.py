"""
FluoroQuant - Export Module
Handles data export in various formats
"""

import os
import json
import pandas as pd
import cv2
import numpy as np
from datetime import datetime


class ResultExporter:
    """Export analysis results in various formats"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_results(self, base_filepath, single_results, multi_results,
                      parameters, options, processed_images=None, 
                      binary_masks=None, active_channels=None):
        """Export current analysis results"""
        exported_files = []
        base_path = os.path.splitext(base_filepath)[0]
        
        # Export CSV files
        if options.get('export_csv', True):
            csv_files = self.export_csv_results(
                base_path, single_results, multi_results
            )
            exported_files.extend(csv_files)
        
        # Export JSON
        if options.get('export_json', True):
            json_file = self.export_json_results(
                base_path, single_results, multi_results, parameters
            )
            exported_files.append(json_file)
        
        # Export Excel
        if options.get('export_excel', False):
            excel_file = self.export_excel_results(
                base_path, single_results, multi_results
            )
            if excel_file:
                exported_files.append(excel_file)
        
        # Export images
        if options.get('export_images', True) and processed_images:
            image_files = self.export_processed_images(
                base_path, processed_images, binary_masks, active_channels
            )
            exported_files.extend(image_files)
        
        # Export overlays
        if options.get('export_overlays', True) and processed_images:
            overlay_files = self.export_overlay_images(
                base_path, processed_images, binary_masks, active_channels
            )
            exported_files.extend(overlay_files)
        
        return exported_files
    
    def export_csv_results(self, base_path, single_results, multi_results):
        """Export results as CSV files"""
        exported_files = []
        
        # Export single channel object data
        for channel, results in single_results.items():
            if results and results.get('objects'):
                df = pd.DataFrame(results['objects'])
                filename = f"{base_path}_{channel}_objects.csv"
                df.to_csv(filename, index=False)
                exported_files.append(filename)
        
        # Export channel summaries
        if single_results:
            summary_data = []
            for channel, results in single_results.items():
                if results:
                    summary = {
                        'channel': channel,
                        'total_objects': results['total_objects'],
                        'total_area': results['total_area'],
                        'coverage_percent': results['coverage_percent'],
                        'mean_object_area': results['mean_object_area'],
                        'total_fluorescence': results['total_fluorescence'],
                        'mean_intensity': results['mean_intensity'],
                        'background_intensity': results.get('background_intensity', 0),
                        'signal_to_background': results.get('signal_to_background', 0)
                    }
                    summary_data.append(summary)
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                summary_file = f"{base_path}_channel_summary.csv"
                df_summary.to_csv(summary_file, index=False)
                exported_files.append(summary_file)
        
        # Export multi-channel results
        if multi_results:
            multi_data = []
            for pair_key, results in multi_results.items():
                row = {'channel_pair': pair_key}
                
                # Add colocalization data
                if 'colocalization' in results:
                    row.update(results['colocalization'])
                
                # Add distance data
                if 'distance_analysis' in results:
                    row.update(results['distance_analysis'])
                
                multi_data.append(row)
            
            if multi_data:
                df_multi = pd.DataFrame(multi_data)
                multi_file = f"{base_path}_multichannel_analysis.csv"
                df_multi.to_csv(multi_file, index=False)
                exported_files.append(multi_file)
        
        return exported_files
    
    def export_json_results(self, base_path, single_results, multi_results, parameters):
        """Export complete results as JSON"""
        complete_results = {
            'analysis_timestamp': self.timestamp,
            'parameters': parameters,
            'single_channel_results': single_results,
            'multichannel_results': multi_results,
            'software_version': parameters.get('software_version', '2.0')
        }
        
        filename = f"{base_path}_complete_analysis.json"
        with open(filename, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        return filename
    
    def export_excel_results(self, base_path, single_results, multi_results):
        """Export results as Excel file with multiple sheets"""
        try:
            filename = f"{base_path}_analysis_results.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Channel summaries
                if single_results:
                    summary_data = []
                    for channel, results in single_results.items():
                        if results:
                            summary = {
                                'Channel': channel,
                                'Objects': results['total_objects'],
                                'Total Area': results['total_area'],
                                'Coverage %': results['coverage_percent'],
                                'Mean Area': results['mean_object_area'],
                                'Total Signal': results['total_fluorescence']
                            }
                            summary_data.append(summary)
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual channel objects
                for channel, results in single_results.items():
                    if results and results.get('objects'):
                        df = pd.DataFrame(results['objects'])
                        sheet_name = f'{channel}_objects'[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Multi-channel results
                if multi_results:
                    multi_data = []
                    for pair_key, results in multi_results.items():
                        row = {'Channel Pair': pair_key}
                        if 'colocalization' in results:
                            row.update(results['colocalization'])
                        if 'distance_analysis' in results:
                            row.update(results['distance_analysis'])
                        multi_data.append(row)
                    
                    if multi_data:
                        df_multi = pd.DataFrame(multi_data)
                        df_multi.to_excel(writer, sheet_name='Multi-Channel', index=False)
            
            return filename
            
        except Exception as e:
            print(f"Error exporting Excel file: {e}")
            return None
    
    def export_processed_images(self, base_path, processed_images, binary_masks, active_channels):
        """Export processed images"""
        exported_files = []
        
        for channel in ['ch1', 'ch2', 'ch3']:
            if active_channels.get(channel, False):
                # Export processed image
                if channel in processed_images and processed_images[channel] is not None:
                    img_uint8 = (processed_images[channel] * 255).astype(np.uint8)
                    filename = f"{base_path}_{channel}_processed.png"
                    cv2.imwrite(filename, img_uint8)
                    exported_files.append(filename)
                
                # Export binary mask
                if channel in binary_masks and binary_masks[channel] is not None:
                    mask_uint8 = (binary_masks[channel] * 255).astype(np.uint8)
                    filename = f"{base_path}_{channel}_binary.png"
                    cv2.imwrite(filename, mask_uint8)
                    exported_files.append(filename)
        
        return exported_files
    
    def export_overlay_images(self, base_path, processed_images, binary_masks, active_channels):
        """Export overlay and composite images with colocalization visualization"""
        exported_files = []
        
        # Single channel overlays
        for channel in ['ch1', 'ch2', 'ch3']:
            if active_channels.get(channel, False) and channel in processed_images:
                overlay = self.create_single_overlay(
                    processed_images[channel],
                    binary_masks.get(channel),
                    channel
                )
                if overlay is not None:
                    filename = f"{base_path}_{channel}_overlay.png"
                    cv2.imwrite(filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    exported_files.append(filename)
        
        # Colocalization overlays
        active = [ch for ch in ['ch1', 'ch2', 'ch3'] if active_channels.get(ch, False)]
        if len(active) >= 2:
            # Pairwise colocalization overlays
            pairs = [('ch1', 'ch2'), ('ch1', 'ch3'), ('ch2', 'ch3')]
            
            for ch1, ch2 in pairs:
                if ch1 in active and ch2 in active:
                    coloc_overlay = self.create_colocalization_overlay(
                        processed_images[ch1],
                        processed_images[ch2], 
                        binary_masks.get(ch1),
                        binary_masks.get(ch2),
                        ch1, ch2
                    )
                    if coloc_overlay is not None:
                        filename = f"{base_path}_colocalization_{ch1}_{ch2}.png"
                        cv2.imwrite(filename, cv2.cvtColor(coloc_overlay, cv2.COLOR_RGB2BGR))
                        exported_files.append(filename)
        
        # Multi-channel composite 
        if len(active) > 1:
            composite = self.create_multichannel_composite(processed_images, active)
            if composite is not None:
                filename = f"{base_path}_composite.png"
                cv2.imwrite(filename, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                exported_files.append(filename)
        
        return exported_files
    
    def create_colocalization_overlay(self, img1, img2, mask1, mask2, ch1, ch2):
        """
        Create colocalization overlay showing:
        - Channel 1 image as grayscale background
        - Channel 1 objects in cyan
        - Overlapping regions in red
        """
        if img1 is None or mask1 is None or mask2 is None:
            return None
        
        # Ensure compatible shapes
        if img1.shape != img2.shape or mask1.shape != mask2.shape:
            min_h = min(img1.shape[0], img2.shape[0], mask1.shape[0], mask2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1], mask1.shape[1], mask2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
            mask1 = mask1[:min_h, :min_w]
            mask2 = mask2[:min_h, :min_w]
        
        background = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)        
        overlay = background.copy()
        
        # Ch 1 objects in cyan 
        ch1_only = mask1 & ~mask2  
        overlay[ch1_only] = [0, 255, 255]  # Cyan
        
        # Overlap in red
        overlap = mask1 & mask2
        overlay[overlap] = [255, 0, 0]  # Red
        
        ch2_only = mask2 & ~mask1
        overlay[ch2_only] = [0, 100, 255]  # Light blue
        
        final_overlay = cv2.addWeighted(background, 0.3, overlay, 0.7, 0)
        
        return final_overlay

    def create_colocalization_legend(self, image_shape):
        """Create a small legend explaining the color coding"""
        legend_height = 80
        legend_width = 200
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        legend[10:25, 10:30] = [0, 255, 255]  # Cyan - Ch1 only
        legend[30:45, 10:30] = [255, 0, 0]    # Red - Overlap  
        legend[50:65, 10:30] = [0, 100, 255]  # Blue - Ch2 only
        
        return legend
    
    def create_single_overlay(self, image, mask, channel):
        """Create overlay of mask on image"""
        if image is None:
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Apply channel-specific color to mask
        if mask is not None:
            color_map = {
                'ch1': (255, 0, 0),    # Red
                'ch2': (0, 255, 0),    # Green
                'ch3': (0, 0, 255)     # Blue
            }
            color = color_map.get(channel, (255, 255, 255))
            
            # Create colored mask
            mask_colored = np.zeros_like(rgb)
            mask_colored[mask > 0] = color
            
            # Blend
            overlay = cv2.addWeighted(rgb, 0.7, mask_colored, 0.3, 0)
        else:
            overlay = rgb
        
        return overlay
    
    def create_multichannel_composite(self, images, active_channels):
        """Create RGB composite from multiple channels"""
        if not active_channels:
            return None
        
        # Get image shape from first active channel
        first_ch = active_channels[0]
        if first_ch not in images or images[first_ch] is None:
            return None
        
        shape = images[first_ch].shape
        composite = np.zeros((*shape, 3), dtype=np.uint8)
        
        # Map channels to RGB
        color_map = {'ch1': 0, 'ch2': 1, 'ch3': 2}
        
        for ch in active_channels:
            if ch in images and images[ch] is not None and ch in color_map:
                composite[:, :, color_map[ch]] = (images[ch] * 255).astype(np.uint8)
        
        return composite
        
    def export_batch_results(self, output_folder, batch_results, parameters, options):
        """Export batch processing results with all selected formats"""
        exported_files = []
        
        try:
            # Compile all results
            all_objects = []
            all_summaries = []
            all_multichannel = []
            
            for group_result in batch_results:
                base_name = group_result['base_name']
                
                # Single channel data
                for channel, channel_results in group_result['single_channel_results'].items():
                    if channel_results:
                        # Objects
                        if channel_results.get('objects'):
                            for obj in channel_results['objects']:
                                obj['image_group'] = base_name
                                obj['channel'] = channel
                                all_objects.append(obj)
                        
                        # Summary
                        summary = {
                            'image_group': base_name,
                            'channel': channel,
                            'total_objects': channel_results['total_objects'],
                            'total_area': channel_results['total_area'],
                            'coverage_percent': channel_results['coverage_percent'],
                            'mean_object_area': channel_results['mean_object_area'],
                            'total_fluorescence': channel_results['total_fluorescence']
                        }
                        all_summaries.append(summary)
                
                # Multi-channel data
                for pair_key, pair_results in group_result['multichannel_results'].items():
                    multi_row = {
                        'image_group': base_name,
                        'channel_pair': pair_key
                    }
                    if 'colocalization' in pair_results:
                        multi_row.update(pair_results['colocalization'])
                    if 'distance_analysis' in pair_results:
                        multi_row.update(pair_results['distance_analysis'])
                    all_multichannel.append(multi_row)
            
            # Export based on selected options
            prefix = f"FluoroQuant_batch_{self.timestamp}"
            
            # CSV exports
            if options.get('export_csv', True):
                print("Exporting batch CSV files...")
                # All objects
                if all_objects:
                    df_objects = pd.DataFrame(all_objects)
                    objects_file = os.path.join(output_folder, f"{prefix}_all_objects.csv")
                    df_objects.to_csv(objects_file, index=False)
                    exported_files.append(objects_file)
                    print(f"Exported objects CSV: {objects_file}")
                
                # Summaries
                if all_summaries:
                    df_summary = pd.DataFrame(all_summaries)
                    summary_file = os.path.join(output_folder, f"{prefix}_summary.csv")
                    df_summary.to_csv(summary_file, index=False)
                    exported_files.append(summary_file)
                    print(f"Exported summary CSV: {summary_file}")
                
                # Multi-channel
                if all_multichannel:
                    df_multi = pd.DataFrame(all_multichannel)
                    multi_file = os.path.join(output_folder, f"{prefix}_multichannel.csv")
                    df_multi.to_csv(multi_file, index=False)
                    exported_files.append(multi_file)
                    print(f"Exported multichannel CSV: {multi_file}")
            
            # JSON export
            if options.get('export_json', True):
                print("Exporting batch JSON file...")
                complete_batch = {
                    'batch_processing_timestamp': self.timestamp,
                    'total_groups_processed': len(batch_results),
                    'analysis_parameters': parameters,
                    'batch_results': batch_results,
                    'summary_statistics': {
                        'total_objects_all_groups': len(all_objects),
                        'groups_processed': len(batch_results),
                        'channels_analyzed': list(set(obj.get('channel', '') for obj in all_objects))
                    }
                }
                
                json_file = os.path.join(output_folder, f"{prefix}_complete.json")
                with open(json_file, 'w') as f:
                    json.dump(complete_batch, f, indent=2, default=str)
                exported_files.append(json_file)
                print(f"Exported JSON: {json_file}")
            
            # Excel export
            if options.get('export_excel', False):
                print("Exporting batch Excel file...")
                excel_file = self.export_batch_excel(
                    output_folder, prefix, all_objects, all_summaries, all_multichannel
                )
                if excel_file:
                    exported_files.append(excel_file)
                    print(f"Exported Excel: {excel_file}")
            
            # Note: Images and overlays are not typically exported in batch mode
            # as it would create too many files, but we could add this if needed
            if options.get('export_images', False):
                print("Note: Individual images not exported in batch mode to avoid excessive files")
            
            print(f"Batch export complete: {len(exported_files)} files created")
            return exported_files
            
        except Exception as e:
            print(f"Batch export error: {e}")
            import traceback
            traceback.print_exc()
            return []

    
    def export_batch_excel(self, output_folder, prefix, objects, summaries, multichannel):
        """Export batch results as Excel file"""
        try:
            filename = os.path.join(output_folder, f"{prefix}_results.xlsx")
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                if summaries:
                    df_summary = pd.DataFrame(summaries)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Objects sheet (first 1000 only due to Excel limitations)
                if objects:
                    df_objects = pd.DataFrame(objects[:1000])
                    df_objects.to_excel(writer, sheet_name='Objects_Sample', index=False)
                
                # Multi-channel sheet
                if multichannel:
                    df_multi = pd.DataFrame(multichannel)
                    df_multi.to_excel(writer, sheet_name='Multi-Channel', index=False)
            
            return filename
            
        except Exception as e:
            print(f"Error creating Excel file: {e}")
            return None