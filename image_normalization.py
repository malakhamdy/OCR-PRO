"""
Image Normalization Module

Handles input image validation, normalization, and coordinate tracking.
Preserves original image while creating standardized processing canvas.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from config import IMAGE_CONFIG


@dataclass
class ImageNormalizationResult:
    """Result of image normalization."""
    original_image: np.ndarray
    processing_image: np.ndarray
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    scale_x: float
    scale_y: float
    padding_x: int
    padding_y: int
    transformation_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_dimensions': (self.original_width, self.original_height),
            'processing_dimensions': (self.processing_width, self.processing_height),
            'scale': (self.scale_x, self.scale_y),
            'padding': (self.padding_x, self.padding_y),
            'transformation_metadata': self.transformation_metadata
        }


def validate_input_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate input image meets minimum requirements.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) < 2:
        return False, "Invalid image dimensions"
    
    height, width = image.shape[:2]
    
    min_resolution = IMAGE_CONFIG.MIN_RESOLUTION
    if height < min_resolution or width < min_resolution:
        return False, f"Image too small: {width}x{height}, minimum {min_resolution}px required"
    
    return True, ""


def normalize_image(image: np.ndarray) -> ImageNormalizationResult:
    """
    Normalize input image to standardized processing space.
    
    - Preserves aspect ratio
    - Uses letterboxing/padding when necessary
    - Records all transformation metadata
    - Does NOT distort the image
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        ImageNormalizationResult with original and processing images
    """
    original_height, original_width = image.shape[:2]
    
    # Validate input
    is_valid, error_msg = validate_input_image(image)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Calculate target dimensions while preserving aspect ratio
    max_width = IMAGE_CONFIG.MAX_PROCESSING_WIDTH
    max_height = IMAGE_CONFIG.MAX_PROCESSING_HEIGHT
    
    # Calculate scale factor to fit within max dimensions
    scale = min(max_width / original_width, max_height / original_height)
    
    # Don't upscale if image is already within limits
    if scale >= 1.0:
        scale = 1.0
        new_width = original_width
        new_height = original_height
    else:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    
    # Resize image if needed
    if scale < 1.0:
        processing_resized = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
    else:
        processing_resized = image.copy()
    
    # Create processing canvas with padding (letterboxing)
    processing_canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    padding_x = (max_width - new_width) // 2
    padding_y = (max_height - new_height) // 2
    
    processing_canvas[padding_y:padding_y+new_height, padding_x:padding_x+new_width] = processing_resized
    
    # Calculate transformation parameters
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    
    # Store transformation metadata
    transformation_metadata = {
        'original_size': (original_width, original_height),
        'resized_size': (new_width, new_height),
        'canvas_size': (max_width, max_height),
        'scale_factor': scale,
        'padding_applied': True,
        'padding_x': padding_x,
        'padding_y': padding_y,
        'aspect_ratio_preserved': True
    }
    
    result = ImageNormalizationResult(
        original_image=image.copy(),
        processing_image=processing_canvas,
        original_width=original_width,
        original_height=original_height,
        processing_width=max_width,
        processing_height=max_height,
        scale_x=scale_x,
        scale_y=scale_y,
        padding_x=padding_x,
        padding_y=padding_y,
        transformation_metadata=transformation_metadata
    )
    
    return result


def transform_coordinates_from_original_to_processing(
    x: float, 
    y: float, 
    normalization_result: ImageNormalizationResult
) -> Tuple[float, float]:
    """
    Transform coordinates from original image space to processing canvas space.
    
    Args:
        x: X coordinate in original image
        y: Y coordinate in original image
        normalization_result: Result from normalize_image
        
    Returns:
        Transformed (x, y) in processing canvas coordinates
    """
    new_x = x * normalization_result.scale_x + normalization_result.padding_x
    new_y = y * normalization_result.scale_y + normalization_result.padding_y
    return new_x, new_y


def transform_coordinates_from_processing_to_original(
    x: float, 
    y: float, 
    normalization_result: ImageNormalizationResult
) -> Tuple[float, float]:
    """
    Transform coordinates from processing canvas space to original image space.
    
    Args:
        x: X coordinate in processing canvas
        y: Y coordinate in processing canvas
        normalization_result: Result from normalize_image
        
    Returns:
        Transformed (x, y) in original image coordinates
    """
    new_x = (x - normalization_result.padding_x) / normalization_result.scale_x
    new_y = (y - normalization_result.padding_y) / normalization_result.scale_y
    return new_x, new_y


def transform_bbox_from_original_to_processing(
    bbox: Tuple[float, float, float, float],
    normalization_result: ImageNormalizationResult
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box from original image to processing canvas.
    
    Args:
        bbox: (x, y, w, h) in original coordinates
        normalization_result: Result from normalize_image
        
    Returns:
        Transformed bbox in processing canvas coordinates
    """
    x, y, w, h = bbox
    
    new_x = x * normalization_result.scale_x + normalization_result.padding_x
    new_y = y * normalization_result.scale_y + normalization_result.padding_y
    new_w = w * normalization_result.scale_x
    new_h = h * normalization_result.scale_y
    
    return (new_x, new_y, new_w, new_h)


def get_effective_roi_in_processing(
    normalization_result: ImageNormalizationResult
) -> Tuple[int, int, int, int]:
    """
    Get the effective region of interest in the processing canvas.
    
    Returns the area where the actual image content is located (excluding padding).
    
    Args:
        normalization_result: Result from normalize_image
        
    Returns:
        (x, y, w, h) of effective image area in processing canvas
    """
    orig_w = normalization_result.original_width
    orig_h = normalization_result.original_height
    
    effective_w = int(orig_w * normalization_result.scale_x)
    effective_h = int(orig_h * normalization_result.scale_y)
    
    x = normalization_result.padding_x
    y = normalization_result.padding_y
    
    return (x, y, effective_w, effective_h)
