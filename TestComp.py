"""
FluoroQuant - Comprehensive Test Suite
Demonstrates testing rigor expected at DESRES
Run with: pytest test_fluoro_suite.py -v --cov=fluoro --cov-report=html
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple
import tempfile
import os

# Import modules to test
from fluoro_analyzer import (
    MultiChannelAnalyzer, 
    SpatialIndex, 
    ObjectData
)
from fluoro_batch import (
    BatchProcessor,
    BatchConfig
)


class TestFixtures:
    """Test fixtures and helper methods"""
    
    @staticmethod
    def create_test_image(shape: Tuple[int, int] = (100, 100), 
                          n_objects: int = 5) -> np.ndarray:
        """Create synthetic test image with known objects"""
        image = np.zeros(shape, dtype=np.float64)
        
        # Add circular objects
        for i in range(n_objects):
            center_y = np.random.randint(20, shape[0] - 20)
            center_x = np.random.randint(20, shape[1] - 20)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask] = np.random.uniform(0.5, 1.0)
        
        return image
    
    @staticmethod
    def create_test_mask(shape: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Create binary mask with known properties"""
        mask = np.zeros(shape, dtype=bool)
        # Add a few rectangular regions
        mask[20:40, 30:50] = True
        mask[60:80, 60:90] = True
        return mask
    
    @staticmethod
    def create_labeled_objects(shape: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Create labeled object array"""
        labeled = np.zeros(shape, dtype=np.int32)
        labeled[20:40, 30:50] = 1
        labeled[60:80, 60:90] = 2
        labeled[30:45, 70:85] = 3
        return labeled


@pytest.fixture
def test_images():
    """Fixture providing test images"""
    fixtures = TestFixtures()
    return {
        'ch1': fixtures.create_test_image(),
        'ch2': fixtures.create_test_image(),
        'ch3': fixtures.create_test_image()
    }


@pytest.fixture
def test_masks():
    """Fixture providing test masks"""
    fixtures = TestFixtures()
    return {
        'ch1': fixtures.create_test_mask(),
        'ch2': fixtures.create_test_mask(),
        'ch3': fixtures.create_test_mask()
    }


@pytest.fixture
def test_labeled():
    """Fixture providing labeled objects"""
    fixtures = TestFixtures()
    return {
        'ch1': fixtures.create_labeled_objects(),
        'ch2': fixtures.create_labeled_objects(),
        'ch3': fixtures.create_labeled_objects()
    }


@pytest.fixture
def analyzer():
    """Fixture providing analyzer instance"""
    return MultiChannelAnalyzer()


@pytest.fixture
def batch_processor():
    """Fixture providing batch processor instance"""
    config = BatchConfig(max_workers=2)
    return BatchProcessor(config)


class TestSpatialIndex:
    """Test spatial indexing functionality"""
    
    def test_spatial_index_creation(self):
        """Test spatial index initialization"""
        from skimage.measure import regionprops
        
        labeled = TestFixtures.create_labeled_objects()
        props = regionprops(labeled)
        
        index = SpatialIndex(props)
        
        assert index.objects == props
        assert len(index.bboxes) == len(props)
        assert index.kdtree is not None
    
    def test_find_potential_overlaps(self):
        """Test overlap detection using spatial index"""
        from skimage.measure import regionprops
        
        labeled = TestFixtures.create_labeled_objects()
        props = regionprops(labeled)
        index = SpatialIndex(props)
        
        # Query with overlapping bbox
        query_bbox = (25, 35, 45, 55)  # Overlaps with object 1
        overlaps = index.find_potential_overlaps(query_bbox)
        
        assert len(overlaps) > 0
        assert 0 in overlaps  # First object should overlap
    
    def test_nearest_neighbors(self):
        """Test KD-tree nearest neighbor queries"""
        from skimage.measure import regionprops
        
        labeled = TestFixtures.create_labeled_objects()
        props = regionprops(labeled)
        index = SpatialIndex(props)
        
        # Find nearest neighbors to a point
        point = (30, 40)
        neighbors = index.find_nearest_neighbors(point, k=2)
        
        assert len(neighbors) <= 2
        assert all(isinstance(d, float) for d, _ in neighbors)
        assert all(isinstance(i, (int, np.integer)) for _, i in neighbors)


class TestMandersCoefficients:
    """Test Manders coefficient calculations"""
    
    def test_manders_perfect_colocalization(self, analyzer):
        """Test Manders coefficients with perfect overlap"""
        # Create identical images and masks
        img1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        img2 = img1.copy()
        mask1 = img1 > 0
        mask2 = mask1.copy()
        
        results = analyzer.analyze_colocalization_optimized(
            img1, img2, mask1, mask2, method='manders'
        )
        
        assert results['manders_m1'] == pytest.approx(1.0)
        assert results['manders_m2'] == pytest.approx(1.0)
        assert results['overlap_percentage'] == pytest.approx(100.0)
    
    def test_manders_no_overlap(self, analyzer):
        """Test Manders coefficients with no overlap"""
        img1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        img2 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
        mask1 = img1 > 0
        mask2 = img2 > 0
        
        results = analyzer.analyze_colocalization_optimized(
            img1, img2, mask1, mask2, method='manders'
        )
        
        assert results['manders_m1'] == pytest.approx(0.0)
        assert results['manders_m2'] == pytest.approx(0.0)
        assert results['overlap_pixels'] == 0
    
    def test_manders_partial_overlap(self, analyzer):
        """Test Manders coefficients with partial overlap"""
        img1 = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        img2 = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.8, 0.8]
        ])
        mask1 = img1 > 0
        mask2 = img2 > 0
        
        results = analyzer.analyze_colocalization_optimized(
            img1, img2, mask1, mask2, method='manders'
        )
        
        # Calculate expected values
        overlap = mask1 & mask2
        m1_expected = np.sum(img1[overlap]) / np.sum(img1[mask1])
        m2_expected = np.sum(img2[overlap]) / np.sum(img2[mask2])
        
        assert results['manders_m1'] == pytest.approx(m1_expected, rel=1e-5)
        assert results['manders_m2'] == pytest.approx(m2_expected, rel=1e-5)
        assert 0 < results['manders_m1'] < 1
        assert 0 < results['manders_m2'] < 1


class TestICQ:
    """Test Intensity Correlation Quotient calculations"""
    
    def test_icq_positive_correlation(self, analyzer):
        """Test ICQ with positively correlated intensities"""
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        img2 = img1 * 2  # Perfect positive correlation
        mask = np.ones_like(img1, dtype=bool)
        
        results = analyzer._calculate_all_coefficients_vectorized(
            img1, img2, mask, mask, mask
        )
        
        # ICQ should be positive for positive correlation
        assert results['intensity_correlation_quotient'] > 0
    
    def test_icq_negative_correlation(self, analyzer):
        """Test ICQ with negatively correlated intensities"""
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        img2 = 10 - img1  # Negative correlation
        mask = np.ones_like(img1, dtype=bool)
        
        results = analyzer._calculate_all_coefficients_vectorized(
            img1, img2, mask, mask, mask
        )
        
        # ICQ should be negative for negative correlation
        assert results['intensity_correlation_quotient'] < 0
    
    def test_icq_no_correlation(self, analyzer):
        """Test ICQ with uncorrelated intensities"""
        np.random.seed(42)
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(10, 10)
        mask = np.ones_like(img1, dtype=bool)
        
        results = analyzer._calculate_all_coefficients_vectorized(
            img1, img2, mask, mask, mask
        )
        
        # ICQ should be close to 0 for uncorrelated data
        assert abs(results['intensity_correlation_quotient']) < 0.2


class TestDistanceAnalysis:
    """Test distance analysis with spatial optimization"""
    
    def test_distance_analysis_same_objects(self, analyzer):
        """Test distance analysis when objects are at same position"""
        labeled = TestFixtures.create_labeled_objects()
        
        results = analyzer.analyze_distances_optimized(
            labeled, labeled, max_distance=10.0
        )
        
        # Same objects should have zero distance
        assert results['min_distance'] == pytest.approx(0.0)
        assert results['mean_nearest_distance_1to2'] == pytest.approx(0.0)
    
    def test_distance_analysis_different_objects(self, analyzer):
        """Test distance analysis with different object positions"""
        labeled1 = np.zeros((100, 100), dtype=int)
        labeled2 = np.zeros((100, 100), dtype=int)
        
        # Object in top-left
        labeled1[10:20, 10:20] = 1
        # Object in bottom-right
        labeled2[80:90, 80:90] = 1
        
        results = analyzer.analyze_distances_optimized(
            labeled1, labeled2, max_distance=10.0)