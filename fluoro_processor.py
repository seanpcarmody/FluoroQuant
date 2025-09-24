"""
FluoroQuant - Image Processing Module
Handles all image preprocessing and thresholding operations
"""

import cv2
import numpy as np
from skimage import filters, morphology, exposure, measure
from skimage.filters import (threshold_otsu, threshold_triangle, 
                            threshold_li, threshold_yen, threshold_local)
from skimage.morphology import disk, remove_small_objects, label


class ImageProcessor:
    """Image processing operations for fluorescence analysis"""
    
    def __init__(self):
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
    
    def load_image(self, filepath):
        """Load and normalize image"""
        try:
            # Try loading as grayscale
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Try loading as color and convert
                color_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if color_image is not None:
                    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Could not load image from {filepath}")
            
            # Normalize to 0-1 range
            image = image.astype(np.float64) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def preprocess_image(self, image, params):
        """Apply preprocessing steps to image"""
        processed = image.copy()
        
        # CLAHE enhancement
        if params.get('clahe', False):
            # Convert to uint8 for CLAHE
            img_uint8 = (processed * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=params.get('clahe_clip', 2.0),
                tileGridSize=(8, 8)
            )
            img_uint8 = clahe.apply(img_uint8)
            processed = img_uint8.astype(np.float64) / 255.0
        
        # Gaussian blur
        if params.get('gaussian', False):
            sigma = params.get('gaussian_sigma', 1.0)
            processed = filters.gaussian(processed, sigma=sigma)
        
        # Median filter
        if params.get('median', False):
            size = int(params.get('median_size', 3))
            processed = filters.median(processed, disk(size))
        
        # Gamma correction
        gamma = params.get('gamma', 1.0)
        if gamma != 1.0:
            processed = exposure.adjust_gamma(processed, gamma=gamma)
        
        # Contrast and brightness adjustment
        contrast = params.get('contrast', 1.0)
        brightness = params.get('brightness', 0)
        if contrast != 1.0 or brightness != 0:
            processed = np.clip(
                processed * contrast + brightness / 255.0,
                0, 1
            )
        
        return processed
    
    def threshold_image(self, image, params):
        """Apply thresholding to create binary image"""
        method = params.get('method', 'otsu')
        
        if method == 'otsu':
            threshold = threshold_otsu(image)
            binary = image > threshold
            
        elif method == 'triangle':
            threshold = threshold_triangle(image)
            binary = image > threshold
            
        elif method == 'li':
            threshold = threshold_li(image)
            binary = image > threshold
            
        elif method == 'yen':
            threshold = threshold_yen(image)
            binary = image > threshold
            
        elif method == 'local':
            block_size = params.get('local_block_size', 35)
            # Ensure odd block size
            if block_size % 2 == 0:
                block_size += 1
            threshold = threshold_local(image, block_size=block_size, offset=0.01)
            binary = image > threshold
            
        elif method == 'manual':
            threshold = params.get('manual_threshold', 128) / 255.0
            binary = image > threshold
            
        else:
            # Default to Otsu
            threshold = threshold_otsu(image)
            binary = image > threshold
        
        # Apply morphological operations if requested
        if params.get('morphology', False):
            morph_size = int(params.get('morph_size', 3))
            selem = disk(morph_size)
            binary = morphology.opening(binary, selem)
            binary = morphology.closing(binary, selem)
        
        return binary
    
    def label_objects(self, binary, min_size=50, max_size=10000):
        """Label connected components with size filtering"""
        # Remove small objects
        cleaned = remove_small_objects(binary, min_size=min_size)
        
        # Label connected components
        labeled = label(cleaned)
        
        # Remove large objects if max_size is set
        if max_size < np.inf:
            props = measure.regionprops(labeled)
            for prop in props:
                if prop.area > max_size:
                    labeled[labeled == prop.label] = 0
            
            # Relabel to ensure consecutive labels
            labeled = label(labeled > 0)
        
        return labeled
    
    def create_composite_image(self, channels, active_channels):
        """Create RGB composite from multiple channels"""
        # Get first active channel to determine shape
        active = [ch for ch in ['ch1', 'ch2', 'ch3'] 
                 if active_channels[ch] and channels[ch] is not None]
        
        if not active:
            return None
        
        shape = channels[active[0]].shape
        composite = np.zeros((*shape, 3))
        
        # Map channels to RGB
        color_map = {'ch1': 0, 'ch2': 1, 'ch3': 2}
        
        for ch in active:
            if ch in color_map:
                composite[:, :, color_map[ch]] = channels[ch]
        
        return composite
    
    def apply_colormap(self, image, colormap='viridis'):
        """Apply colormap to grayscale image"""
        # Normalize to 0-255
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == 'red':
            colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_AUTUMN)
            colored[:, :, [0, 1]] = 0  # Keep only red channel
        elif colormap == 'green':
            colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_SUMMER)
            colored[:, :, [0, 2]] = 0  # Keep only green channel
        elif colormap == 'blue':
            colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_WINTER)
            colored[:, :, [1, 2]] = 0  # Keep only blue channel
        else:
            colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_VIRIDIS)
        
        return colored
    
    def create_overlay(self, base_image, mask, color=(255, 0, 0), alpha=0.5):
        """Create overlay of mask on base image"""
        # Convert base image to RGB if grayscale
        if len(base_image.shape) == 2:
            base_rgb = cv2.cvtColor(
                (base_image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )
        else:
            base_rgb = (base_image * 255).astype(np.uint8)
        
        # Create colored mask
        mask_colored = np.zeros_like(base_rgb)
        mask_colored[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(base_rgb, 1-alpha, mask_colored, alpha, 0)
        
        return overlay
    
    def enhance_contrast_adaptive(self, image, clip_limit=2.0, tile_size=8):
        """Apply adaptive histogram equalization"""
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        enhanced = clahe.apply(img_uint8)
        
        # Convert back to float
        return enhanced.astype(np.float64) / 255.0
    
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter for edge-preserving smoothing"""
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        
        # Convert back to float
        return filtered.astype(np.float64) / 255.0
    
    def detect_edges(self, image, method='canny', low_threshold=0.1, high_threshold=0.3):
        """Detect edges in image"""
        if method == 'canny':
            edges = filters.canny(
                image,
                low_threshold=low_threshold,
                high_threshold=high_threshold
            )
        elif method == 'sobel':
            edges = filters.sobel(image)
        elif method == 'laplacian':
            edges = filters.laplace(image)
        else:
            edges = filters.canny(image)
        
        return edges
    
    def fill_holes(self, binary):
        """Fill holes in binary objects"""
        return morphology.binary_fill_holes(binary)
    
    def watershed_segmentation(self, image, binary):
        """Apply watershed segmentation to separate touching objects"""
        # Compute distance transform
        distance = morphology.distance_transform_edt(binary)
        
        # Find local maxima
        local_maxima = morphology.local_maxima(distance)
        markers = measure.label(local_maxima)
        
        # Apply watershed
        labels = morphology.watershed(-distance, markers, mask=binary)
        
        return labels
    
    def get_image_statistics(self, image):
        """Calculate basic image statistics"""
        return {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'q25': np.percentile(image, 25),
            'q75': np.percentile(image, 75)
        }
    
    def normalize_image(self, image, method='minmax'):
        """Normalize image intensities"""
        if method == 'minmax':
            # Min-max normalization
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image
                
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                normalized = (image - mean) / std
                # Clip to reasonable range
                normalized = np.clip(normalized, -3, 3)
                # Scale to 0-1
                normalized = (normalized + 3) / 6
            else:
                normalized = image
                
        elif method == 'percentile':
            # Percentile normalization
            p1 = np.percentile(image, 1)
            p99 = np.percentile(image, 99)
            if p99 > p1:
                normalized = np.clip((image - p1) / (p99 - p1), 0, 1)
            else:
                normalized = image
        else:
            normalized = image
        
        return normalized