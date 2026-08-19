"""
Egyptian National ID OCR System - Configuration Module

Centralized configuration for the entire pipeline.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ImageConfig:
    """Image processing configuration."""
    # Canonical card dimensions (for rectified output)
    CANONICAL_CARD_WIDTH: int = 1200
    CANONICAL_CARD_HEIGHT: int = 760
    
    # Processing canvas constraints
    MAX_PROCESSING_WIDTH: int = 1920
    MAX_PROCESSING_HEIGHT: int = 1280
    
    # Minimum acceptable image quality
    MIN_CARD_AREA_RATIO: float = 0.15  # Card must occupy at least 15% of image
    MIN_RESOLUTION: int = 300  # Minimum pixels on shortest side


@dataclass
class DetectionConfig:
    """Card detection configuration."""
    # Contour approximation epsilon
    CONTOUR_EPSILON_FACTOR: float = 0.02
    
    # Minimum contour area ratio
    MIN_CONTOUR_AREA_RATIO: float = 0.1
    
    # Quadrilateral detection thresholds
    MIN_CORNER_DISTANCE: float = 50.0  # Minimum distance between corners
    MAX_ASPECT_RATIO_DEVIATION: float = 0.3  # Allowed deviation from expected aspect ratio
    
    # Expected card aspect ratio (W/H)
    EXPECTED_ASPECT_RATIO: float = 1.587  # Standard ID-1 card ratio (85.6/54)
    
    # Confidence thresholds
    MIN_DETECTION_CONFIDENCE: float = 0.7


@dataclass
class FieldLocalizationConfig:
    """Field localization configuration for canonical card space."""
    
    # Front side field regions (normalized coordinates 0-1)
    FRONT_FIELDS = {
        'national_id': {
            'x_norm': (0.35, 0.95),
            'y_norm': (0.18, 0.26),
            'type': 'numeric',
            'required': True
        },
        'name': {
            'x_norm': (0.35, 0.95),
            'y_norm': (0.28, 0.38),
            'type': 'arabic_text',
            'required': True
        },
        'date_of_birth': {
            'x_norm': (0.35, 0.65),
            'y_norm': (0.40, 0.48),
            'type': 'date',
            'required': True
        },
        'gender': {
            'x_norm': (0.35, 0.55),
            'y_norm': (0.50, 0.58),
            'type': 'arabic_short',
            'required': True
        },
        'birth_governorate': {
            'x_norm': (0.35, 0.75),
            'y_norm': (0.60, 0.68),
            'type': 'arabic_text',
            'required': False
        },
        'profession': {
            'x_norm': (0.35, 0.85),
            'y_norm': (0.70, 0.78),
            'type': 'arabic_text',
            'required': False
        },
        'address': {
            'x_norm': (0.35, 0.95),
            'y_norm': (0.80, 0.92),
            'type': 'arabic_multiline',
            'required': False
        }
    }
    
    # Back side field regions (normalized coordinates 0-1)
    BACK_FIELDS = {
        'barcode': {
            'x_norm': (0.10, 0.90),
            'y_norm': (0.15, 0.45),
            'type': 'barcode',
            'required': False
        },
        'additional_info': {
            'x_norm': (0.10, 0.90),
            'y_norm': (0.50, 0.85),
            'type': 'arabic_multiline',
            'required': False
        }
    }
    
    # Crop padding factors (relative to field height)
    CROP_PADDING_X_FACTOR: float = 0.05
    CROP_PADDING_Y_FACTOR: float = 0.15


@dataclass
class PreprocessingConfig:
    """Field-specific preprocessing configurations."""
    
    # NID preprocessing variants
    NID_VARIANTS = [
        {
            'name': 'standard',
            'grayscale': True,
            'denoise_strength': 10,
            'contrast_method': 'normalize',
            'threshold': None,
            'sharpen': False
        },
        {
            'name': 'high_contrast',
            'grayscale': True,
            'denoise_strength': 5,
            'contrast_method': 'clahe',
            'clahe_clip_limit': 2.0,
            'threshold': 'adaptive',
            'sharpen': True
        },
        {
            'name': 'minimal',
            'grayscale': True,
            'denoise_strength': 0,
            'contrast_method': 'none',
            'threshold': None,
            'sharpen': False
        }
    ]
    
    # Name field preprocessing variants
    NAME_VARIANTS = [
        {
            'name': 'arabic_standard',
            'grayscale': True,
            'denoise_strength': 8,
            'contrast_method': 'normalize',
            'preserve_arabic_dots': True
        },
        {
            'name': 'arabic_enhanced',
            'grayscale': True,
            'denoise_strength': 5,
            'contrast_method': 'clahe',
            'clahe_clip_limit': 1.5,
            'preserve_arabic_dots': True
        }
    ]
    
    # Address field preprocessing variants
    ADDRESS_VARIANTS = [
        {
            'name': 'multiline_standard',
            'grayscale': True,
            'denoise_strength': 10,
            'contrast_method': 'normalize'
        },
        {
            'name': 'multiline_enhanced',
            'grayscale': True,
            'denoise_strength': 8,
            'contrast_method': 'clahe',
            'clahe_clip_limit': 2.0
        }
    ]
    
    # DOB preprocessing variants
    DOB_VARIANTS = [
        {
            'name': 'date_standard',
            'grayscale': True,
            'denoise_strength': 10,
            'contrast_method': 'normalize',
            'threshold': None
        },
        {
            'name': 'date_high_contrast',
            'grayscale': True,
            'denoise_strength': 5,
            'contrast_method': 'clahe',
            'threshold': 'adaptive'
        }
    ]
    
    # Gender field preprocessing
    GENDER_VARIANTS = [
        {
            'name': 'short_text',
            'grayscale': True,
            'denoise_strength': 8,
            'contrast_method': 'normalize'
        }
    ]


@dataclass
class OCRConfig:
    """OCR engine configuration."""
    
    # PaddleOCR settings
    PADDLE_LANG: str = 'ar'  # Arabic
    PADDLE_DET_DB_THRESH: float = 0.3
    PADDLE_DET_BOX_THRESH: float = 0.5
    PADDLE_REC_BATCH_NUM: int = 1
    PADDLE_USE_ANGLE_CLS: bool = False
    PADDLE_SHOW_LOG_VISUALIZATION: bool = False
    
    # Confidence thresholds
    MIN_OCR_CONFIDENCE: float = 0.5
    HIGH_OCR_CONFIDENCE: float = 0.85
    
    # Multi-pass OCR
    MAX_OCR_PASSES: int = 3


@dataclass
class ValidationConfig:
    """Validation rules configuration."""
    
    # National ID validation
    NID_LENGTH: int = 14
    NID_CENTURY_CODE_START: int = 2  # 2=19xx, 3=20xx
    NID_GOVERNORATE_START: int = 5
    NID_GOVERNORATE_LENGTH: int = 2
    NID_CHECKSUM_POSITION: int = 13  # Last digit (0-indexed)
    
    # Valid governorate codes
    VALID_GOVERNORATE_CODES = {
        '01': 'القاهرة',
        '02': 'الإسكندرية',
        '03': 'بورسعيد',
        '04': 'السويس',
        '11': 'دمياط',
        '12': 'الغربية',
        '13': 'المنوفية',
        '14': 'القليوبية',
        '15': 'الكفر الشيخ',
        '16': 'الشرقية',
        '17': 'الدقهلية',
        '18': 'البحيرة',
        '19': 'مطروح',
        '21': 'الجيزة',
        '22': 'بني سويف',
        '23': 'الفيوم',
        '24': 'المنيا',
        '25': 'أسيوط',
        '26': 'سوهاج',
        '27': 'قنا',
        '28': 'الأقصر',
        '29': 'أسوان',
        '31': 'البحر الأحمر',
        '32': 'الوادي الجديد',
        '33': 'جنوب سيناء',
        '34': 'شمال سيناء',
        '88': 'خارج الجمهورية'
    }
    
    # Date validation
    MIN_VALID_YEAR: int = 1900
    MAX_VALID_YEAR: int = 2100
    
    # Confidence thresholds for verification
    CROSS_VALIDATION_MATCH_THRESHOLD: float = 0.9


@dataclass
class QualityConfig:
    """Image quality assessment thresholds."""
    
    # Blur detection (Laplacian variance)
    BLUR_THRESHOLD_LOW: float = 50.0
    BLUR_THRESHOLD_HIGH: float = 100.0
    
    # Brightness
    MIN_BRIGHTNESS: float = 30.0
    MAX_BRIGHTNESS: float = 220.0
    OPTIMAL_BRIGHTNESS_RANGE: Tuple[float, float] = (80.0, 180.0)
    
    # Contrast
    MIN_CONTRAST: float = 20.0
    
    # Noise estimation
    MAX_NOISE_RATIO: float = 0.3
    
    # Glare detection
    GLARE_PIXEL_RATIO_THRESHOLD: float = 0.15


# Global config instances
IMAGE_CONFIG = ImageConfig()
DETECTION_CONFIG = DetectionConfig()
FIELD_CONFIG = FieldLocalizationConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
OCR_CONFIG = OCRConfig()
VALIDATION_CONFIG = ValidationConfig()
QUALITY_CONFIG = QualityConfig()
