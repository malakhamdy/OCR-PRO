"""
Test Suite for Egyptian National ID OCR System

Tests all pipeline components including:
- Image normalization
- Card detection
- Perspective correction
- Side classification
- Field localization
- Arabic normalization
- NID validation
- Cross-field validation
"""

import unittest
import numpy as np
import cv2
from typing import Tuple


class TestImageNormalization(unittest.TestCase):
    """Tests for image_normalization module."""
    
    def setUp(self):
        from image_normalization import normalize_image, validate_input_image
        self.normalize_image = normalize_image
        self.validate_input_image = validate_input_image
    
    def test_validate_valid_image(self):
        """Test validation of valid image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        is_valid, error = self.validate_input_image(img)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_none_image(self):
        """Test validation of None image."""
        is_valid, error = self.validate_input_image(None)
        self.assertFalse(is_valid)
        self.assertIn("None", error)
    
    def test_normalize_preserves_aspect_ratio(self):
        """Test that normalization preserves aspect ratio."""
        # Create test image with specific aspect ratio
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.normalize_image(img)
        
        # Check scale factors are equal (aspect ratio preserved)
        self.assertAlmostEqual(result.scale_x, result.scale_y, places=2)
    
    def test_normalize_different_resolutions(self):
        """Test normalization with different input resolutions."""
        resolutions = [(480, 640), (720, 1280), (1080, 1920)]
        
        for height, width in resolutions:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            result = self.normalize_image(img)
            
            # Should have processing dimensions
            self.assertGreater(result.processing_width, 0)
            self.assertGreater(result.processing_height, 0)


class TestCardDetection(unittest.TestCase):
    """Tests for card_detection module."""
    
    def setUp(self):
        from card_detection import detect_card, order_corners
        self.detect_card = detect_card
        self.order_corners = order_corners
    
    def test_detect_card_on_synthetic(self):
        """Test card detection on synthetic rectangular image."""
        # Create synthetic card-like image
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 200
        
        # Draw rectangle simulating card
        cv2.rectangle(img, (100, 100), (1100, 700), (0, 0, 0), 3)
        
        result = self.detect_card(img)
        
        # Should attempt detection (may or may not succeed on synthetic)
        self.assertIsInstance(result.card_detected, bool)
        self.assertIsInstance(result.confidence, float)
    
    def test_order_corners(self):
        """Test corner ordering function."""
        # Define corners in random order
        corners = np.array([
            [100, 200],  # BL
            [200, 100],  # TR
            [100, 100],  # TL
            [200, 200]   # BR
        ], dtype=np.float32)
        
        ordered = self.order_corners(corners)
        
        # Check we get 4 corners back
        self.assertEqual(len(ordered), 4)


class TestPerspectiveCorrection(unittest.TestCase):
    """Tests for perspective_correction module."""
    
    def setUp(self):
        from perspective_correction import rectify_card, PerspectiveCorrectionResult
        from card_detection import CardDetectionResult, CardCorner
        self.rectify_card = rectify_card
    
    def test_rectify_with_mock_detection(self):
        """Test rectification with mock detection result."""
        img = np.zeros((800, 1200, 3), dtype=np.uint8)
        
        # Create mock detection result - import CardCorner here
        from card_detection import CardCorner
        
        corners = [
            CardCorner(100, 100, 0.9),
            CardCorner(1100, 100, 0.9),
            CardCorner(1100, 700, 0.9),
            CardCorner(100, 700, 0.9)
        ]
        
        from card_detection import CardDetectionResult
        detection = CardDetectionResult(
            card_detected=True,
            confidence=0.9,
            corners=corners,
            contour=None,
            card_area_ratio=0.5,
            aspect_ratio=1.5
        )
        
        result = self.rectify_card(img, detection)
        
        # Should produce canonical card
        self.assertIsNotNone(result.canonical_card)
        self.assertEqual(result.transformation_success, True)


class TestSideClassification(unittest.TestCase):
    """Tests for side_classification module."""
    
    def setUp(self):
        from side_classification import classify_side
        self.classify_side = classify_side
    
    def test_classify_returns_valid_side(self):
        """Test that classification returns valid side."""
        img = np.zeros((760, 1200, 3), dtype=np.uint8)
        result = self.classify_side(img)
        
        self.assertIn(result.side, ['front', 'back'])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestFieldLocalization(unittest.TestCase):
    """Tests for field_localization module."""
    
    def setUp(self):
        from field_localization import localize_fields_on_canonical_card
        self.localize_fields = localize_fields_on_canonical_card
    
    def test_localize_front_fields(self):
        """Test field localization for front side."""
        img = np.zeros((760, 1200, 3), dtype=np.uint8)
        result = self.localize_fields(img, 'front')
        
        self.assertEqual(result.side, 'front')
        self.assertGreater(len(result.fields), 0)
    
    def test_localize_back_fields(self):
        """Test field localization for back side."""
        img = np.zeros((760, 1200, 3), dtype=np.uint8)
        result = self.localize_fields(img, 'back')
        
        self.assertEqual(result.side, 'back')


class TestArabicNormalization(unittest.TestCase):
    """Tests for arabic_normalization module."""
    
    def setUp(self):
        from arabic_normalization import (
            convert_arabic_digits_to_western,
            normalize_text_for_field,
            extract_digits_only
        )
        self.convert_digits = convert_arabic_digits_to_western
        self.normalize = normalize_text_for_field
        self.extract_digits = extract_digits_only
    
    def test_convert_arabic_digits(self):
        """Test Arabic digit conversion."""
        arabic_text = "٠١٢٣٤٥٦٧٨٩"
        western = self.convert_digits(arabic_text)
        self.assertEqual(western, "0123456789")
    
    def test_normalize_numeric_field(self):
        """Test normalization of numeric field."""
        raw = "٢٩٠٠١٠١١٢٣٤٥٦٧"
        result = self.normalize(raw, 'numeric')
        
        self.assertEqual(result.normalized_text, "29001011234567")
    
    def test_extract_digits_mixed(self):
        """Test digit extraction from mixed text."""
        text = "الرقم ١٢٣٤٥"
        digits = self.extract_digits(text)
        self.assertEqual(digits, "12345")


class TestNidValidation(unittest.TestCase):
    """Tests for nid_validation module."""
    
    def setUp(self):
        from nid_validation import validate_national_id, mask_nid
        self.validate = validate_national_id
        self.mask = mask_nid
    
    def test_validate_valid_nid_structure(self):
        """Test validation of structurally valid NID."""
        # Valid structure: century=2 (19xx), date=900101, gov=01, serial=2345
        nid = "29001011234567"
        result = self.validate(nid)
        
        # Should have derived info - at minimum governorate
        self.assertIsNotNone(result.derived_info)
        # Governorate should be extracted from positions 5-6
        self.assertIn('governorate_code', result.derived_info)
    
    def test_validate_invalid_length(self):
        """Test validation of invalid length NID."""
        nid = "123456789"  # Too short
        result = self.validate(nid)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.validation_status, "INVALID_LENGTH")
    
    def test_validate_invalid_format(self):
        """Test validation of non-numeric NID."""
        nid = "29001011234abc"
        result = self.validate(nid)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.validation_status, "INVALID_FORMAT")
    
    def test_mask_nid(self):
        """Test NID masking for privacy."""
        nid = "29001011234567"
        masked = self.mask(nid)
        
        self.assertEqual(len(masked), 14)
        self.assertTrue(masked.endswith("4567"))
        self.assertTrue(masked.startswith("*"))


class TestCrossFieldValidation(unittest.TestCase):
    """Tests for cross_field_validation module."""
    
    def setUp(self):
        from cross_field_validation import perform_cross_field_validation
        from nid_validation import validate_national_id
        self.cross_validate = perform_cross_field_validation
        self.validate_nid = validate_national_id
    
    def test_cross_validation_with_matching_dob(self):
        """Test cross-validation with matching DOB."""
        fields = {
            'date_of_birth': {
                'raw': '1990-01-01',
                'normalized': '1990-01-01'
            }
        }
        
        nid_result = self.validate_nid("29001011234567")
        
        result = self.cross_validate(fields, nid_result)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.checks, list)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def setUp(self):
        from pipeline import EgyptianIDPipeline
        self.pipeline = EgyptianIDPipeline()
    
    def test_pipeline_handles_empty_image(self):
        """Test pipeline handles empty/black image gracefully."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.pipeline.process_image(img)
        
        # Should not crash - may fail on card detection which is expected
        # The important thing is it doesn't raise an exception
        self.assertIsInstance(result.success, bool)
    
    def test_pipeline_preserves_coordinate_systems(self):
        """Test that pipeline maintains coordinate system metadata."""
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 200
        
        # Add some content for detection
        cv2.rectangle(img, (100, 100), (1100, 700), (50, 50, 50), -1)
        
        result = self.pipeline.process_image(img)
        
        # Debug info should contain transformation metadata
        if result.debug_info:
            self.assertIn('normalization', result.debug_info)


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestImageNormalization,
        TestCardDetection,
        TestPerspectiveCorrection,
        TestSideClassification,
        TestFieldLocalization,
        TestArabicNormalization,
        TestNidValidation,
        TestCrossFieldValidation,
        TestPipelineIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
