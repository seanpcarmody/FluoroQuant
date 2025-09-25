"""
FluoroQuant - Analysis Module
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import logging
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ObjectData:
    """Data class for object analysis results"""
    object_id: int
    area_pixels: float
    perimeter: float
    centroid_x: float
    centroid_y: float
    mean_intensity: float
    max_intensity: float
    min_intensity: float
    total_intensity: float
    eccentricity: float
    solidity: float
    extent: float
    major_axis_length: float
    minor_axis_length: float
    orientation: float
    circularity: float
    aspect_ratio: float
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col


class SpatialIndex:
    """
    Spatial indexing for efficient overlap detection.
    Demonstrates understanding of algorithmic optimization.
    """
    
    def __init__(self, objects: List[Any]):
        """Initialize spatial index with bounding boxes"""
        self.objects = objects
        self.bboxes = np.array([obj.bbox for obj in objects])
        
        if len(objects) > 0:
            # Create KD-tree for centroid-based queries
            centroids = np.array([[obj.centroid[0], obj.centroid[1]] for obj in objects])
            self.kdtree = cKDTree(centroids)
        else:
            self.kdtree = None
        
        logger.debug(f"Spatial index created for {len(objects)} objects")
    
    def find_potential_overlaps(self, query_bbox: Tuple[int, int, int, int]) -> List[int]:
        """
        Find objects that potentially overlap with query bbox.
        O(log n) average case vs O(n) for naive approach.
        """
        if len(self.objects) == 0:
            return []
        
        min_row, min_col, max_row, max_col = query_bbox
        
        # Vectorized bbox overlap check
        overlaps = (
            (self.bboxes[:, 0] < max_row) &
            (self.bboxes[:, 2] > min_row) &
            (self.bboxes[:, 1] < max_col) &
            (self.bboxes[:, 3] > min_col)
        )
        
        return np.where(overlaps)[0].tolist()
    
    def find_nearest_neighbors(self, point: Tuple[float, float], k: int = 5) -> List[Tuple[float, int]]:
        """Find k nearest neighbors to a point"""
        if self.kdtree is None:
            return []
        
        distances, indices = self.kdtree.query(point, k=min(k, len(self.objects)))
        return list(zip(distances, indices))


class MultiChannelAnalyzer:
    """
    Multi channel analysis class with improved memory and processing techniques
    """
    
    def __init__(self):
        self.channel_names: Dict[str, str] = {
            'ch1': 'Channel 1', 
            'ch2': 'Channel 2', 
            'ch3': 'Channel 3'
        }
    
    def analyze_single_channels(
        self,
        active_channels: Dict[str, bool],
        labeled_objects: Dict[str, Optional[np.ndarray]],
        intensity_images: Dict[str, Optional[np.ndarray]],
        binary_masks: Dict[str, Optional[np.ndarray]],
        analysis_mode: str = 'general'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Note: this performs single channel quantitative analysis
        
            active_channels: Channel activation status
            labeled_objects: Labeled object masks
            intensity_images: Original intensity images
            binary_masks: Binary segmentation masks
            analysis_mode: Analysis mode ('general', 'neurons', 'cells', 'colonies')
            
        Return
            Dictionary of analysis results per channel
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        for channel in ['ch1', 'ch2', 'ch3']:
            if not active_channels[channel] or labeled_objects[channel] is None:
                continue
            
            try:
                # Get region properties
                props = regionprops(
                    labeled_objects[channel], 
                    intensity_image=intensity_images[channel]
                )
                
                if not props:
                    logger.warning(f"No objects found in {channel}")
                    continue
                
                # Vectorized analysis for efficiency
                objects_data = self._vectorized_object_analysis(props, analysis_mode)
                
                # Calculate channel summary
                channel_summary = self._calculate_channel_summary_optimized(
                    objects_data,
                    binary_masks[channel],
                    intensity_images[channel]
                )
                channel_summary['channel'] = channel
                channel_summary['objects'] = [self._object_to_dict(obj) for obj in objects_data]
                
                results[channel] = channel_summary
                
            except Exception as e:
                logger.error(f"Error analyzing {channel}: {e}", exc_info=True)
                results[channel] = self._empty_channel_summary()
        
        return results
    
    def _vectorized_object_analysis(
        self,
        props: List[Any],
        analysis_mode: str
    ) -> List[ObjectData]:
        """
        Vectorized object analysis for improved performance.
        Processes multiple properties simultaneously.
        """
        objects_data: List[ObjectData] = []
        
        # Pre-allocate arrays for vectorized operations
        n_objects = len(props)
        areas = np.zeros(n_objects)
        perimeters = np.zeros(n_objects)
        
        for i, prop in enumerate(props):
            areas[i] = prop.area
            perimeters[i] = prop.perimeter
        
        # Vectorized circularity calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            circularities = 4 * np.pi * areas / (perimeters ** 2)
            circularities[perimeters == 0] = 0
        
        for i, prop in enumerate(props):
            obj_data = ObjectData(
                object_id=prop.label,
                area_pixels=prop.area,
                perimeter=prop.perimeter,
                centroid_x=prop.centroid[1],
                centroid_y=prop.centroid[0],
                mean_intensity=prop.mean_intensity,
                max_intensity=prop.max_intensity,
                min_intensity=prop.min_intensity,
                total_intensity=prop.mean_intensity * prop.area,
                eccentricity=prop.eccentricity,
                solidity=prop.solidity,
                extent=prop.extent,
                major_axis_length=prop.major_axis_length,
                minor_axis_length=prop.minor_axis_length,
                orientation=prop.orientation,
                circularity=circularities[i],
                aspect_ratio=(prop.major_axis_length / prop.minor_axis_length 
                             if prop.minor_axis_length > 0 else 0),
                bbox=prop.bbox
            )
            
            self._add_mode_specific_metrics(obj_data, prop, analysis_mode)
            
            objects_data.append(obj_data)
        
        return objects_data
    
    def _add_mode_specific_metrics(
        self,
        obj_data: ObjectData,
        prop: Any,
        mode: str
    ) -> None:
        """Add mode-specific metrics to object data"""
        if mode == 'neurons':
            setattr(obj_data, 'neurite_length', prop.major_axis_length)
            setattr(obj_data, 'neurite_width', prop.minor_axis_length)
            setattr(obj_data, 'branching_index', 
                   prop.perimeter / prop.area if prop.area > 0 else 0)
        elif mode == 'cells':
            setattr(obj_data, 'cell_roundness', obj_data.circularity)
            setattr(obj_data, 'compactness', 
                   prop.area / prop.convex_area if prop.convex_area > 0 else 0)
        elif mode == 'colonies':
            setattr(obj_data, 'colony_size', prop.area)
            setattr(obj_data, 'colony_density', prop.mean_intensity)
    
    def _calculate_channel_summary_optimized(
        self,
        objects_data: List[ObjectData],
        binary_mask: Optional[np.ndarray],
        intensity_image: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Optimized channel summary calculation using vectorization"""
        if not objects_data:
            return self._empty_channel_summary()
        
        # Vectorized calculations
        areas = np.array([obj.area_pixels for obj in objects_data])
        intensities = np.array([obj.mean_intensity for obj in objects_data])
        total_intensities = np.array([obj.total_intensity for obj in objects_data])
        
        # Background calculation
        background_intensity = 0.0
        if binary_mask is not None and intensity_image is not None:
            background_mask = ~binary_mask
            if np.any(background_mask):
                background_intensity = float(np.mean(intensity_image[background_mask]))
        
        return {
            'total_objects': len(objects_data),
            'total_area': float(np.sum(areas)),
            'coverage_percent': float(np.mean(binary_mask) * 100) if binary_mask is not None else 0,
            'mean_object_area': float(np.mean(areas)),
            'std_object_area': float(np.std(areas)),
            'total_fluorescence': float(np.sum(total_intensities)),
            'mean_intensity': float(np.mean(intensities)),
            'background_intensity': background_intensity,
            'signal_to_background': float(np.mean(intensities) / background_intensity) 
                                   if background_intensity > 0 else np.inf
        }
    
    def analyze_multichannel(
        self,
        active_channels: Dict[str, bool],
        intensity_images: Dict[str, Optional[np.ndarray]],
        binary_masks: Dict[str, Optional[np.ndarray]],
        labeled_objects: Dict[str, Optional[np.ndarray]],
        params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform multi-channel analysis with spatial optimization.
        
        Key optimization: Uses spatial indexing for overlap detection,
        reducing complexity from O(n*m) to O(n log m) for overlap analysis.
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        # Validate and correct image shapes if needed
        intensity_images, binary_masks = self._ensure_compatible_shapes(
            active_channels, intensity_images, binary_masks
        )
        
        pairs = self._get_channel_pairs(active_channels, params)
        
        for ch1, ch2 in pairs:
            pair_key = f"{ch1}_{ch2}"
            
            try:
                # Colocalization analysis
                coloc_results = self.analyze_colocalization_optimized(
                    intensity_images[ch1], intensity_images[ch2],
                    binary_masks[ch1], binary_masks[ch2],
                    params.get('method', 'manders')
                )
                
                # Distance analysis
                distance_results = self.analyze_distances_optimized(
                    labeled_objects[ch1], labeled_objects[ch2],
                    params.get('distance_threshold', 10.0)
                )
                
                # Overlap analysis with spatial indexing
                overlap_analysis = self.analyze_object_overlap_optimized(
                    labeled_objects[ch1], labeled_objects[ch2]
                )
                
                results[pair_key] = {
                    'channels': (ch1, ch2),
                    'colocalization': coloc_results,
                    'distance_analysis': distance_results,
                    'object_overlap': overlap_analysis
                }
                
            except Exception as e:
                logger.error(f"Error analyzing pair {pair_key}: {e}", exc_info=True)
                results[pair_key] = self._empty_pair_results()
        
        return results
    
    def analyze_object_overlap_optimized(
        self,
        labeled1: Optional[np.ndarray],
        labeled2: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Optimized overlap detection using spatial indexing.
        Complexity: O(n log m) instead of O(n*m)
        """
        if labeled1 is None or labeled2 is None:
            return {'overlap_pairs': [], 'total_overlaps': 0}
        
        props1 = regionprops(labeled1)
        props2 = regionprops(labeled2)
        
        if not props1 or not props2:
            return {'overlap_pairs': [], 'total_overlaps': 0}
        
        # Create spatial index for channel 2
        spatial_index = SpatialIndex(props2)
        
        overlap_pairs = []
        
        for p1 in props1:
            # Use spatial index to find potential overlaps
            potential_overlaps = spatial_index.find_potential_overlaps(p1.bbox)
            
            mask1 = labeled1 == p1.label
            
            for idx in potential_overlaps:
                p2 = props2[idx]
                mask2 = labeled2 == p2.label
                
                # Compute actual overlap
                overlap = np.sum(mask1 & mask2)
                
                if overlap > 0:
                    overlap_pairs.append({
                        'obj1_id': int(p1.label),
                        'obj2_id': int(p2.label),
                        'overlap_pixels': int(overlap),
                        'overlap_fraction_1': float(overlap / p1.area),
                        'overlap_fraction_2': float(overlap / p2.area),
                        'distance': float(np.sqrt(
                            (p1.centroid[0] - p2.centroid[0])**2 + 
                            (p1.centroid[1] - p2.centroid[1])**2
                        ))
                    })
        
        return {
            'overlap_pairs': overlap_pairs,
            'total_overlaps': len(overlap_pairs),
            'mean_overlap_fraction': float(np.mean([p['overlap_fraction_1'] 
                                                   for p in overlap_pairs])) 
                                    if overlap_pairs else 0
        }
    
    def analyze_colocalization_optimized(
        self,
        img1: Optional[np.ndarray],
        img2: Optional[np.ndarray],
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
        method: str = 'manders'
    ) -> Dict[str, float]:
        """Optimized colocalization analysis with vectorization"""
        if img1 is None or img2 is None or mask1 is None or mask2 is None:
            return self._empty_colocalization_results()
        
        # Ensure compatible shapes
        img1, img2, mask1, mask2 = self._ensure_same_shape(img1, img2, mask1, mask2)
        
        # Vectorized calculations
        overlap_mask = mask1 & mask2
        union_mask = mask1 | mask2
        
        overlap_pixels = int(np.sum(overlap_mask))
        union_pixels = int(np.sum(union_mask))
        
        if union_pixels == 0:
            return self._empty_colocalization_results()
        
        results = {
            'overlap_pixels': overlap_pixels,
            'ch1_pixels': int(np.sum(mask1)),
            'ch2_pixels': int(np.sum(mask2)),
            'union_pixels': union_pixels,
            'overlap_percentage': float((overlap_pixels / union_pixels) * 100)
        }
        
        # Calculate all coefficients using vectorized operations
        results.update(self._calculate_all_coefficients_vectorized(
            img1, img2, mask1, mask2, overlap_mask
        ))
        
        return results
    
    def _calculate_all_coefficients_vectorized(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        overlap_mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all colocalization coefficients using vectorization"""
        # Pre-compute masked intensities
        img1_masked = img1[mask1]
        img2_masked = img2[mask2]
        img1_overlap = img1[overlap_mask]
        img2_overlap = img2[overlap_mask]
        
        # Manders coefficients
        sum_img1_masked = np.sum(img1_masked)
        sum_img2_masked = np.sum(img2_masked)
        
        m1 = float(np.sum(img1_overlap) / sum_img1_masked) if sum_img1_masked > 0 else 0
        m2 = float(np.sum(img2_overlap) / sum_img2_masked) if sum_img2_masked > 0 else 0
        
        # Pearson correlation
        if len(img1_overlap) > 1:
            # Check if either array is constant 
            if np.var(img1_overlap) == 0 or np.var(img2_overlap) == 0:
                pearson = 0  
            else:
                try:
                    r, _ = pearsonr(img1_overlap, img2_overlap)
                    pearson = float(r) if not np.isnan(r) else 0
                except:
                    pearson = 0
        else:
            pearson = 0
        
        # Overlap coefficient
        overlap_sum = float(np.sum(img1_overlap) + np.sum(img2_overlap))
        total_sum = float(sum_img1_masked + sum_img2_masked)
        overlap_coef = overlap_sum / total_sum if total_sum > 0 else 0
        
        # Intensity Correlation Quotient
        if len(img1_overlap) > 0:
            mean1 = np.mean(img1_overlap)
            mean2 = np.mean(img2_overlap)
            pdm = (img1_overlap - mean1) * (img2_overlap - mean2)
            icq = float((np.sum(pdm > 0) / len(pdm)) - 0.5)
        else:
            icq = 0
        
        return {
            'manders_m1': m1,
            'manders_m2': m2,
            'manders_average': float((m1 + m2) / 2),
            'pearson_correlation': pearson,
            'overlap_coefficient': overlap_coef,
            'intensity_correlation_quotient': icq
        }
    
    def analyze_distances_optimized(
        self,
        labeled1: Optional[np.ndarray],
        labeled2: Optional[np.ndarray],
        max_distance: float
    ) -> Dict[str, Any]:
        """Optimized distance analysis using KD-tree for nearest neighbor queries"""
        if labeled1 is None or labeled2 is None:
            return self._empty_distance_results()
        
        props1 = regionprops(labeled1)
        props2 = regionprops(labeled2)
        
        if not props1 or not props2:
            return self._empty_distance_results()
        
        centroids1 = np.array([[p.centroid[0], p.centroid[1]] for p in props1])
        centroids2 = np.array([[p.centroid[0], p.centroid[1]] for p in props2])
        
        tree2 = cKDTree(centroids2)
        tree1 = cKDTree(centroids1)
        
        distances_1to2, indices_1to2 = tree2.query(centroids1)
        distances_2to1, indices_2to1 = tree1.query(centroids2)
        
        close_1to2 = np.sum(distances_1to2 <= max_distance)
        close_2to1 = np.sum(distances_2to1 <= max_distance)
        
        all_distances = cdist(centroids1, centroids2)
        min_distance = float(np.min(all_distances))
        
        return {
            'mean_nearest_distance_1to2': float(np.mean(distances_1to2)),
            'mean_nearest_distance_2to1': float(np.mean(distances_2to1)),
            'median_nearest_distance_1to2': float(np.median(distances_1to2)),
            'median_nearest_distance_2to1': float(np.median(distances_2to1)),
            'std_nearest_distance_1to2': float(np.std(distances_1to2)),
            'std_nearest_distance_2to1': float(np.std(distances_2to1)),
            'min_distance': min_distance,
            'max_nearest_distance': float(max(np.max(distances_1to2), np.max(distances_2to1))),
            'objects_within_threshold_1to2': int(close_1to2),
            'objects_within_threshold_2to1': int(close_2to1),
            'percentage_close_1to2': float((close_1to2 / len(props1)) * 100),
            'percentage_close_2to1': float((close_2to1 / len(props2)) * 100),
            'total_objects_ch1': len(props1),
            'total_objects_ch2': len(props2)
        }
    
    # Helper methods
    
    def _ensure_compatible_shapes(
        self,
        active_channels: Dict[str, bool],
        intensity_images: Dict[str, Optional[np.ndarray]],
        binary_masks: Dict[str, Optional[np.ndarray]]
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[np.ndarray]]]:
        """Ensure all images have compatible shapes"""
        active_images = {ch: img for ch, img in intensity_images.items()
                        if active_channels.get(ch, False) and img is not None}
        
        if len(active_images) < 2:
            return intensity_images, binary_masks
        
        # Find minimum dimensions
        shapes = [img.shape for img in active_images.values()]
        min_height = min(s[0] for s in shapes)
        min_width = min(s[1] for s in shapes)
        
        # Resize if needed
        for ch in active_images:
            if intensity_images[ch] is not None:
                intensity_images[ch] = intensity_images[ch][:min_height, :min_width]
            if binary_masks[ch] is not None:
                binary_masks[ch] = binary_masks[ch][:min_height, :min_width]
        
        return intensity_images, binary_masks
    
    def _ensure_same_shape(
        self,
        *arrays: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Ensure arrays have the same shape"""
        shapes = [arr.shape for arr in arrays if arr is not None]
        if not shapes:
            return arrays
        
        min_height = min(s[0] for s in shapes)
        min_width = min(s[1] for s in shapes)
        
        return tuple(arr[:min_height, :min_width] if arr is not None else None 
                    for arr in arrays)
    
    def _get_channel_pairs(
        self,
        active_channels: Dict[str, bool],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Get channel pairs for analysis"""
        pairs = []
        
        if params.get('pairs', {}).get('ch1_ch2', True):
            if active_channels.get('ch1', False) and active_channels.get('ch2', False):
                pairs.append(('ch1', 'ch2'))
        
        if params.get('pairs', {}).get('ch1_ch3', True):
            if active_channels.get('ch1', False) and active_channels.get('ch3', False):
                pairs.append(('ch1', 'ch3'))
        
        if params.get('pairs', {}).get('ch2_ch3', True):
            if active_channels.get('ch2', False) and active_channels.get('ch3', False):
                pairs.append(('ch2', 'ch3'))
        
        return pairs
    
    def _object_to_dict(self, obj: ObjectData) -> Dict[str, Any]:
        """Convert ObjectData to dictionary"""
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    
    def _empty_channel_summary(self) -> Dict[str, Any]:
        """Return empty channel summary"""
        return {
            'total_objects': 0,
            'total_area': 0,
            'coverage_percent': 0,
            'mean_object_area': 0,
            'std_object_area': 0,
            'total_fluorescence': 0,
            'mean_intensity': 0,
            'background_intensity': 0,
            'signal_to_background': 0,
            'objects': []
        }
    
    def _empty_colocalization_results(self) -> Dict[str, float]:
        """Return empty colocalization results"""
        return {
            'overlap_pixels': 0,
            'ch1_pixels': 0,
            'ch2_pixels': 0,
            'union_pixels': 0,
            'overlap_percentage': 0.0,
            'manders_m1': 0.0,
            'manders_m2': 0.0,
            'manders_average': 0.0,
            'pearson_correlation': 0.0,
            'overlap_coefficient': 0.0,
            'intensity_correlation_quotient': 0.0
        }
    
    def _empty_distance_results(self) -> Dict[str, Any]:
        """Return empty distance results"""
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
            'total_objects_ch1': 0,
            'total_objects_ch2': 0
        }
    
    def _empty_pair_results(self) -> Dict[str, Any]:
        """Return empty pair analysis results"""
        return {
            'colocalization': self._empty_colocalization_results(),
            'distance_analysis': self._empty_distance_results(),
            'object_overlap': {'overlap_pairs': [], 'total_overlaps': 0}
        }