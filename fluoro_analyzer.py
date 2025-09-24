"""
FluoroQuant - Analysis Module
Handles quantitative and multi-channel analysis
"""

import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


class MultiChannelAnalyzer:
    """Analyzer for single and multi-channel fluorescence data"""
    
    def __init__(self):
        self.channel_names = {'ch1': 'Channel 1', 'ch2': 'Channel 2', 'ch3': 'Channel 3'}
    
    def analyze_single_channels(self, active_channels, labeled_objects, 
                               intensity_images, binary_masks, analysis_mode='general'):
        """Perform single channel quantitative analysis"""
        results = {}
        
        for channel in ['ch1', 'ch2', 'ch3']:
            if not active_channels[channel] or labeled_objects[channel] is None:
                continue
            
            # Get region properties
            props = regionprops(
                labeled_objects[channel], 
                intensity_image=intensity_images[channel]
            )
            
            # Analyze each object
            objects_data = []
            for prop in props:
                obj_data = self.analyze_object(prop, analysis_mode)
                objects_data.append(obj_data)
            
            # Calculate channel summary
            channel_summary = self.calculate_channel_summary(
                objects_data, 
                binary_masks[channel],
                intensity_images[channel]
            )
            channel_summary['channel'] = channel
            channel_summary['objects'] = objects_data
            
            results[channel] = channel_summary
        
        return results
    
    def analyze_object(self, prop, analysis_mode):
        """Analyze single object properties"""
        # Basic properties
        obj_data = {
            'object_id': prop.label,
            'area_pixels': prop.area,
            'perimeter': prop.perimeter,
            'centroid_x': prop.centroid[1],
            'centroid_y': prop.centroid[0],
            'mean_intensity': prop.mean_intensity,
            'max_intensity': prop.max_intensity,
            'min_intensity': prop.min_intensity,
            'total_intensity': prop.mean_intensity * prop.area,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'extent': prop.extent,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation
        }
        
        # Calculate additional metrics
        if prop.perimeter > 0:
            obj_data['circularity'] = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        else:
            obj_data['circularity'] = 0
        
        if prop.minor_axis_length > 0:
            obj_data['aspect_ratio'] = prop.major_axis_length / prop.minor_axis_length
        else:
            obj_data['aspect_ratio'] = 0
        
        # Mode-specific analysis
        if analysis_mode == 'neurons':
            obj_data.update(self.analyze_neuron(prop))
        elif analysis_mode == 'cells':
            obj_data.update(self.analyze_cell(prop))
        elif analysis_mode == 'colonies':
            obj_data.update(self.analyze_colony(prop))
        
        return obj_data
    
    def analyze_neuron(self, prop):
        """Neuron-specific analysis"""
        return {
            'neurite_length': prop.major_axis_length,
            'neurite_width': prop.minor_axis_length,
            'branching_index': prop.perimeter / prop.area if prop.area > 0 else 0,
            'complexity': prop.perimeter ** 2 / (4 * np.pi * prop.area) if prop.area > 0 else 0
        }
    
    def analyze_cell(self, prop):
        """Cell-specific analysis"""
        return {
            'cell_roundness': prop.circularity if hasattr(prop, 'circularity') else 0,
            'compactness': prop.area / prop.convex_area if prop.convex_area > 0 else 0,
            'nuclear_intensity': prop.max_intensity,
            'cytoplasmic_intensity': prop.mean_intensity,
            'n_c_ratio': prop.max_intensity / prop.mean_intensity if prop.mean_intensity > 0 else 0
        }
    
    def analyze_colony(self, prop):
        """Colony-specific analysis"""
        return {
            'colony_size': prop.area,
            'colony_density': prop.mean_intensity,
            'colony_heterogeneity': prop.max_intensity - prop.min_intensity,
            'growth_area': prop.convex_area
        }
    
    def calculate_channel_summary(self, objects_data, binary_mask, intensity_image):
        """Calculate summary statistics for channel"""
        if not objects_data:
            return {
                'total_objects': 0,
                'total_area': 0,
                'coverage_percent': 0,
                'mean_object_area': 0,
                'total_fluorescence': 0,
                'mean_intensity': 0,
                'background_intensity': 0
            }
        
        # Calculate background intensity
        background_mask = ~binary_mask
        if np.any(background_mask):
            background_intensity = np.mean(intensity_image[background_mask])
        else:
            background_intensity = 0
        
        return {
            'total_objects': len(objects_data),
            'total_area': sum(obj['area_pixels'] for obj in objects_data),
            'coverage_percent': np.mean(binary_mask) * 100,
            'mean_object_area': np.mean([obj['area_pixels'] for obj in objects_data]),
            'std_object_area': np.std([obj['area_pixels'] for obj in objects_data]),
            'total_fluorescence': sum(obj['total_intensity'] for obj in objects_data),
            'mean_intensity': np.mean([obj['mean_intensity'] for obj in objects_data]),
            'background_intensity': background_intensity,
            'signal_to_background': np.mean([obj['mean_intensity'] for obj in objects_data]) / background_intensity if background_intensity > 0 else np.inf
        }
    
    def analyze_multichannel(self, active_channels, intensity_images, 
                        binary_masks, labeled_objects, params):
        """Perform multi-channel analysis"""
        # Validate image shapes
        if not self.validate_image_shapes(active_channels, intensity_images):
            print("Proceeding with automatic shape correction...")
        
        results = {}
            
        # Get active channel pairs based on parameters
        pairs = []
        if params['pairs']['ch1_ch2'] and active_channels['ch1'] and active_channels['ch2']:
            pairs.append(('ch1', 'ch2'))
        if params['pairs']['ch1_ch3'] and active_channels['ch1'] and active_channels['ch3']:
            pairs.append(('ch1', 'ch3'))
        if params['pairs']['ch2_ch3'] and active_channels['ch2'] and active_channels['ch3']:
            pairs.append(('ch2', 'ch3'))
        
        # Analyze each pair
        for ch1, ch2 in pairs:
            pair_key = f"{ch1}_{ch2}"
            
            # Colocalization analysis
            coloc_results = self.analyze_colocalization(
                intensity_images[ch1], intensity_images[ch2],
                binary_masks[ch1], binary_masks[ch2],
                params['method']
            )
            
            # Distance analysis
            distance_results = self.analyze_distances(
                labeled_objects[ch1], labeled_objects[ch2],
                params['distance_threshold']
            )
            
            results[pair_key] = {
                'channels': (ch1, ch2),
                'colocalization': coloc_results,
                'distance_analysis': distance_results
            }
        
        return results
    def validate_image_shapes(self, active_channels, intensity_images):
        """Validate that all active channel images have the same dimensions"""
        active_imgs = {ch: img for ch, img in intensity_images.items() 
                    if active_channels[ch] and img is not None}
        
        if len(active_imgs) < 2:
            return True
        
        shapes = [img.shape for img in active_imgs.values()]
        if not all(shape == shapes[0] for shape in shapes):
            print("Warning: Channel images have different dimensions:")
            for ch, img in active_imgs.items():
                print(f"  {ch}: {img.shape}")
            return False
        
        return True

    def analyze_colocalization(self, img1, img2, mask1, mask2, method='manders'):
        """Analyze colocalization between two channels"""
        # Check if images have the same shape
        if img1.shape != img2.shape:
            print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
            # Resize images to match the smaller dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
            mask1 = mask1[:min_height, :min_width]
            mask2 = mask2[:min_height, :min_width]
            
            print(f"Resized to common dimensions: {img1.shape}")
        
        # Calculate overlap
        overlap_mask = mask1 & mask2
        overlap_pixels = np.sum(overlap_mask)
        union_pixels = np.sum(mask1 | mask2)
        
        if union_pixels == 0:
            return {
                'overlap_coefficient': 0,
                'manders_m1': 0,
                'manders_m2': 0,
                'pearson_correlation': 0,
                'overlap_pixels': 0,
                'ch1_pixels': 0,
                'ch2_pixels': 0,
                'overlap_percentage': 0
            }
        
        # Basic overlap metrics
        results = {
            'overlap_pixels': overlap_pixels,
            'ch1_pixels': np.sum(mask1),
            'ch2_pixels': np.sum(mask2),
            'union_pixels': union_pixels,
            'overlap_percentage': (overlap_pixels / union_pixels) * 100
        }
        
        # Method-specific calculations
        if method == 'manders' or method == 'all':
            results.update(self.calculate_manders_coefficients(
                img1, img2, mask1, mask2, overlap_mask
            ))
        
        # Method-specific calculations - always calculate all metrics
        results.update(self.calculate_manders_coefficients(
            img1, img2, mask1, mask2, overlap_mask
        ))

        results['pearson_correlation'] = self.calculate_pearson_correlation(
            img1, img2, overlap_mask
        )

        results['overlap_coefficient'] = self.calculate_overlap_coefficient(
            img1, img2, mask1, mask2
        )
        
        return results
    
    def calculate_manders_coefficients(self, img1, img2, mask1, mask2, overlap_mask):
        """Calculate Manders coefficients"""
        # M1: fraction of channel 1 overlapping with channel 2
        if np.sum(img1[mask1]) > 0:
            m1 = np.sum(img1[overlap_mask]) / np.sum(img1[mask1])
        else:
            m1 = 0
        
        # M2: fraction of channel 2 overlapping with channel 1
        if np.sum(img2[mask2]) > 0:
            m2 = np.sum(img2[overlap_mask]) / np.sum(img2[mask2])
        else:
            m2 = 0
        
        return {
            'manders_m1': m1,
            'manders_m2': m2,
            'manders_average': (m1 + m2) / 2
        }
    
    def calculate_pearson_correlation(self, img1, img2, mask):
        """Calculate Pearson correlation coefficient"""
        if np.sum(mask) < 2:
            return 0
        
        # Get intensities in overlap region
        try:
            intensities1 = img1[mask]
            intensities2 = img2[mask]
            
            # Check if we have enough data points
            if len(intensities1) < 2 or len(intensities2) < 2:
                return 0
                
            # Check for zero variance (constant values)
            if np.std(intensities1) == 0 or np.std(intensities2) == 0:
                return 0
            
            # Calculate correlation
            correlation, p_value = pearsonr(intensities1, intensities2)
            
            # Check for NaN or infinite values
            if np.isnan(correlation) or np.isinf(correlation):
                return 0
                
            return float(correlation)
            
        except Exception as e:
            print(f"Pearson correlation calculation error: {e}")
            return 0
    
    def calculate_overlap_coefficient(self, img1, img2, mask1, mask2):
        """Calculate overlap coefficient"""
        try:
            overlap_mask = mask1 & mask2
            overlap_sum = np.sum(img1[overlap_mask]) + np.sum(img2[overlap_mask])
            total_sum = np.sum(img1[mask1]) + np.sum(img2[mask2])
            
            if total_sum > 0:
                return float(overlap_sum / total_sum)
            else:
                return 0.0
        except Exception as e:
            print(f"Overlap coefficient calculation error: {e}")
            return 0.0
    
    def analyze_distances(self, labeled1, labeled2, max_distance):
        """Analyze distances between objects in two channels"""
        if labeled1 is None or labeled2 is None:
            return {}
        
        # Get object properties
        props1 = regionprops(labeled1)
        props2 = regionprops(labeled2)
        
        if not props1 or not props2:
            return {
                'mean_nearest_distance_1to2': 0,
                'mean_nearest_distance_2to1': 0,
                'median_nearest_distance_1to2': 0,
                'median_nearest_distance_2to1': 0,
                'std_nearest_distance_1to2': 0,
                'std_nearest_distance_2to1': 0,
                'min_distance': 0,
                'max_nearest_distance': 0,
                'objects_within_threshold_1to2': 0,
                'objects_within_threshold_2to1': 0,
                'percentage_close_1to2': 0,
                'percentage_close_2to1': 0,
                'total_objects_ch1': len(props1) if props1 else 0,
                'total_objects_ch2': len(props2) if props2 else 0
            }
        
        # Get centroids - ensure they're 2D arrays
        centroids1 = np.array([[prop.centroid[0], prop.centroid[1]] for prop in props1])
        centroids2 = np.array([[prop.centroid[0], prop.centroid[1]] for prop in props2])
        
        # Ensure centroids have the right shape
        if centroids1.ndim != 2 or centroids2.ndim != 2:
            print(f"Warning: Centroid shape issue - centroids1: {centroids1.shape}, centroids2: {centroids2.shape}")
            return {}
        
        if centroids1.shape[1] != 2 or centroids2.shape[1] != 2:
            print(f"Warning: Centroid dimension issue - centroids1: {centroids1.shape}, centroids2: {centroids2.shape}")
            return {}
        
        # Calculate distance matrix
        try:
            distances = cdist(centroids1, centroids2)
        except ValueError as e:
            print(f"Distance calculation error: {e}")
            print(f"Centroids1 shape: {centroids1.shape}, Centroids2 shape: {centroids2.shape}")
            return {}
        
        # Nearest neighbor distances
        min_distances_1to2 = np.min(distances, axis=1)
        min_distances_2to1 = np.min(distances, axis=0)
        
        # Objects within threshold
        close_objects_1to2 = np.sum(min_distances_1to2 <= max_distance)
        close_objects_2to1 = np.sum(min_distances_2to1 <= max_distance)
        
        return {
            'mean_nearest_distance_1to2': float(np.mean(min_distances_1to2)),
            'mean_nearest_distance_2to1': float(np.mean(min_distances_2to1)),
            'median_nearest_distance_1to2': float(np.median(min_distances_1to2)),
            'median_nearest_distance_2to1': float(np.median(min_distances_2to1)),
            'std_nearest_distance_1to2': float(np.std(min_distances_1to2)),
            'std_nearest_distance_2to1': float(np.std(min_distances_2to1)),
            'min_distance': float(np.min(distances)),
            'max_nearest_distance': float(np.max([np.max(min_distances_1to2), np.max(min_distances_2to1)])),
            'objects_within_threshold_1to2': int(close_objects_1to2),
            'objects_within_threshold_2to1': int(close_objects_2to1),
            'percentage_close_1to2': float((close_objects_1to2 / len(props1)) * 100),
            'percentage_close_2to1': float((close_objects_2to1 / len(props2)) * 100),
            'total_objects_ch1': len(props1),
            'total_objects_ch2': len(props2)
        }
    
    def calculate_intensity_correlation_quotient(self, img1, img2, mask):
        """Calculate intensity correlation quotient (ICQ)"""
        if np.sum(mask) == 0:
            return 0
        
        # Get intensities
        i1 = img1[mask]
        i2 = img2[mask]
        
        # Calculate means
        mean1 = np.mean(i1)
        mean2 = np.mean(i2)
        
        # Calculate PDM (Product of Differences from Mean)
        pdm = (i1 - mean1) * (i2 - mean2)
        
        # ICQ is the fraction of pixels with positive PDM minus 0.5
        icq = (np.sum(pdm > 0) / len(pdm)) - 0.5
        
        return icq
    
    def analyze_object_overlap(self, labeled1, labeled2):
        """Analyze overlap between individual objects"""
        props1 = regionprops(labeled1)
        props2 = regionprops(labeled2)
        
        overlap_pairs = []
        
        for p1 in props1:
            mask1 = labeled1 == p1.label
            
            for p2 in props2:
                mask2 = labeled2 == p2.label
                overlap = np.sum(mask1 & mask2)
                
                if overlap > 0:
                    overlap_pairs.append({
                        'obj1_id': p1.label,
                        'obj2_id': p2.label,
                        'overlap_pixels': overlap,
                        'overlap_fraction_1': overlap / p1.area,
                        'overlap_fraction_2': overlap / p2.area,
                        'distance': np.sqrt(
                            (p1.centroid[0] - p2.centroid[0])**2 + 
                            (p1.centroid[1] - p2.centroid[1])**2
                        )
                    })
        
        return overlap_pairs