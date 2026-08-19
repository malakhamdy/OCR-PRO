"""
Field-Specific Preprocessing Module

Applies optimized preprocessing for each field type.
Supports multiple variants per field for multi-pass OCR.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from config import PREPROCESSING_CONFIG


@dataclass
class PreprocessedCrop:
    """Result of field preprocessing."""
    crop: np.ndarray
    variant_name: str
    preprocessing_steps: List[str]
    original_dimensions: Tuple[int, int]
    processed_dimensions: Tuple[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_name': self.variant_name,
            'preprocessing_steps': self.preprocessing_steps,
            'original_dimensions': self.original_dimensions,
            'processed_dimensions': self.processed_dimensions
        }


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_denoise(image: np.ndarray, strength: int) -> np.ndarray:
    """Apply non-local means denoising."""
    if strength <= 0:
        return image
    
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def apply_contrast_normalization(image: np.ndarray) -> np.ndarray:
    """Normalize contrast using histogram equalization."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)


def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for binarization."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Gaussian adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """Apply mild sharpening filter."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    # Blend with original to avoid over-sharpening
    return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)


def upscale_for_ocr(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    """Upscale image for better OCR performance."""
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def preprocess_field_crop(
    crop: np.ndarray,
    field_type: str,
    variant_index: int = 0
) -> PreprocessedCrop:
    """
    Apply field-specific preprocessing to a crop.
    
    Args:
        crop: Field crop image
        field_type: Type of field ('numeric', 'arabic_text', 'date', etc.)
        variant_index: Which preprocessing variant to use
        
    Returns:
        PreprocessedCrop with processed image and metadata
    """
    original_height, original_width = crop.shape[:2]
    preprocessing_steps = []
    processed = crop.copy()
    
    # Select preprocessing configuration based on field type
    if field_type == 'numeric':
        variants = PREPROCESSING_CONFIG.NID_VARIANTS
    elif field_type in ['arabic_text', 'arabic_short']:
        variants = PREPROCESSING_CONFIG.NAME_VARIANTS
    elif field_type == 'arabic_multiline':
        variants = PREPROCESSING_CONFIG.ADDRESS_VARIANTS
    elif field_type == 'date':
        variants = PREPROCESSING_CONFIG.DOB_VARIANTS
    else:
        # Default to minimal processing
        variants = [{'name': 'default', 'grayscale': True, 'denoise_strength': 5}]
    
    # Get selected variant
    variant_idx = min(variant_index, len(variants) - 1)
    variant = variants[variant_idx]
    
    variant_name = variant.get('name', 'unknown')
    
    # Apply preprocessing steps based on variant configuration
    
    # Step 1: Grayscale conversion
    if variant.get('grayscale', True):
        processed = apply_grayscale(processed)
        preprocessing_steps.append('grayscale')
    
    # Step 2: Denoising
    denoise_strength = variant.get('denoise_strength', 0)
    if denoise_strength > 0:
        processed = apply_denoise(processed, denoise_strength)
        preprocessing_steps.append(f'denoise_{denoise_strength}')
    
    # Step 3: Contrast enhancement
    contrast_method = variant.get('contrast_method', 'normalize')
    if contrast_method == 'clahe':
        clip_limit = variant.get('clahe_clip_limit', 2.0)
        processed = apply_clahe(processed, clip_limit)
        preprocessing_steps.append(f'clahe_{clip_limit}')
    elif contrast_method == 'normalize':
        processed = apply_contrast_normalization(processed)
        preprocessing_steps.append('contrast_normalize')
    # else: 'none' - skip contrast adjustment
    
    # Step 4: Thresholding (optional, use carefully for Arabic text)
    threshold_method = variant.get('threshold', None)
    if threshold_method == 'adaptive' and not variant.get('preserve_arabic_dots', False):
        processed = apply_adaptive_threshold(processed)
        preprocessing_steps.append('adaptive_threshold')
    
    # Step 5: Sharpening (use carefully for Arabic text)
    if variant.get('sharpen', False) and not variant.get('preserve_arabic_dots', False):
        processed = apply_sharpening(processed)
        preprocessing_steps.append('sharpen')
    
    # Step 6: Upscaling for small crops
    height, width = processed.shape[:2]
    if height < 30 or width < 50:
        scale_factor = max(30 / height if height < 30 else 1.0, 
                          50 / width if width < 50 else 1.0)
        scale_factor = min(scale_factor, 3.0)  # Cap at 3x
        processed = upscale_for_ocr(processed, scale_factor)
        preprocessing_steps.append(f'upscale_{scale_factor:.1f}x')
    
    processed_height, processed_width = processed.shape[:2]
    
    return PreprocessedCrop(
        crop=processed,
        variant_name=variant_name,
        preprocessing_steps=preprocessing_steps,
        original_dimensions=(original_width, original_height),
        processed_dimensions=(processed_width, processed_height)
    )


def generate_preprocessing_variants(
    crop: np.ndarray,
    field_type: str
) -> List[PreprocessedCrop]:
    """
    Generate multiple preprocessing variants for multi-pass OCR.
    
    Args:
        crop: Field crop image
        field_type: Type of field
        
    Returns:
        List of PreprocessedCrop with different preprocessing approaches
    """
    variants = []
    
    # Determine which variant set to use
    if field_type == 'numeric':
        variant_configs = PREPROCESSING_CONFIG.NID_VARIANTS
    elif field_type in ['arabic_text', 'arabic_short']:
        variant_configs = PREPROCESSING_CONFIG.NAME_VARIANTS
    elif field_type == 'arabic_multiline':
        variant_configs = PREPROCESSING_CONFIG.ADDRESS_VARIANTS
    elif field_type == 'date':
        variant_configs = PREPROCESSING_CONFIG.DOB_VARIANTS
    else:
        variant_configs = [{'name': 'default', 'grayscale': True, 'denoise_strength': 5}]
    
    # Generate all variants
    for i, _ in enumerate(variant_configs):
        try:
            preprocessed = preprocess_field_crop(crop, field_type, variant_index=i)
            variants.append(preprocessed)
        except Exception as e:
            # Skip failed variants
            continue
    
    # If no variants succeeded, create a minimal fallback
    if not variants:
        fallback = PreprocessedCrop(
            crop=crop.copy(),
            variant_name='fallback',
            preprocessing_steps=['none'],
            original_dimensions=(crop.shape[1], crop.shape[0]),
            processed_dimensions=(crop.shape[1], crop.shape[0])
        )
        variants.append(fallback)
    
    return variants


def assess_crop_quality(crop: np.ndarray) -> Dict[str, Any]:
    """
    Assess quality of field crop for OCR.
    
    Args:
        crop: Field crop image
        
    Returns:
        Dictionary with quality metrics
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    height, width = gray.shape
    
    # Resolution check
    resolution_score = min((height * width) / 10000.0, 1.0)
    
    # Blur detection (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    blur_normalized = min(blur_score / 200.0, 1.0)  # Normalize to 0-1
    
    # Contrast check
    contrast = np.std(gray)
    contrast_normalized = min(contrast / 50.0, 1.0)
    
    # Brightness check
    brightness = np.mean(gray)
    brightness_optimal = 1.0 - abs(brightness - 128) / 128  # Peak at 128
    
    # Overall quality score
    overall_quality = (
        resolution_score * 0.2 +
        blur_normalized * 0.4 +
        contrast_normalized * 0.25 +
        brightness_optimal * 0.15
    )
    
    return {
        'resolution_score': resolution_score,
        'blur_score': blur_normalized,
        'contrast_score': contrast_normalized,
        'brightness_score': brightness_optimal,
        'overall_quality': overall_quality,
        'dimensions': (width, height),
        'is_acceptable': overall_quality > 0.4
    }
