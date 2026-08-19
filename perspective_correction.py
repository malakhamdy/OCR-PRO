"""
Perspective Correction Module

Rectifies detected card to canonical representation.
Applies homography transformation to correct perspective distortion.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from config import IMAGE_CONFIG
from card_detection import CardDetectionResult


@dataclass
class PerspectiveCorrectionResult:
    """Result of perspective correction."""
    canonical_card: np.ndarray
    canonical_width: int
    canonical_height: int
    homography_matrix: np.ndarray
    inverse_homography: np.ndarray
    source_corners: np.ndarray
    destination_corners: np.ndarray
    transformation_success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'canonical_dimensions': (self.canonical_width, self.canonical_height),
            'transformation_success': self.transformation_success,
            'source_corners': self.source_corners.tolist() if self.source_corners is not None else [],
            'destination_corners': self.destination_corners.tolist(),
            'error_message': self.error_message
        }


def create_destination_corners(
    canonical_width: int, 
    canonical_height: int
) -> np.ndarray:
    """
    Create destination corners for canonical card space.
    
    Args:
        canonical_width: Width of canonical card
        canonical_height: Height of canonical card
        
    Returns:
        Array of 4 corners in clockwise order (TL, TR, BR, BL)
    """
    return np.array([
        [0, 0],
        [canonical_width - 1, 0],
        [canonical_width - 1, canonical_height - 1],
        [0, canonical_height - 1]
    ], dtype=np.float32)


def rectify_card(
    image: np.ndarray,
    detection_result: CardDetectionResult,
    canonical_width: Optional[int] = None,
    canonical_height: Optional[int] = None
) -> PerspectiveCorrectionResult:
    """
    Rectify detected card to canonical representation.
    
    Applies perspective transformation using homography.
    
    Args:
        image: Input image (processing canvas)
        detection_result: Result from card detection
        canonical_width: Target width for canonical card
        canonical_height: Target height for canonical card
        
    Returns:
        PerspectiveCorrectionResult with rectified card
    """
    if canonical_width is None:
        canonical_width = IMAGE_CONFIG.CANONICAL_CARD_WIDTH
    if canonical_height is None:
        canonical_height = IMAGE_CONFIG.CANONICAL_CARD_HEIGHT
    
    # Check if card was detected
    if not detection_result.card_detected:
        return PerspectiveCorrectionResult(
            canonical_card=np.zeros((canonical_height, canonical_width, 3), dtype=np.uint8),
            canonical_width=canonical_width,
            canonical_height=canonical_height,
            homography_matrix=None,
            inverse_homography=None,
            source_corners=None,
            destination_corners=None,
            transformation_success=False,
            error_message="Card not detected"
        )
    
    # Get ordered source corners
    source_corners = detection_result.get_ordered_corners()
    
    if source_corners is None:
        return PerspectiveCorrectionResult(
            canonical_card=np.zeros((canonical_height, canonical_width, 3), dtype=np.uint8),
            canonical_width=canonical_width,
            canonical_height=canonical_height,
            homography_matrix=None,
            inverse_homography=None,
            source_corners=None,
            destination_corners=None,
            transformation_success=False,
            error_message="Invalid corner detection"
        )
    
    # Create destination corners
    destination_corners = create_destination_corners(canonical_width, canonical_height)
    
    try:
        # Compute homography matrix
        homography_matrix, _ = cv2.findHomography(source_corners, destination_corners, cv2.RANSAC, 5.0)
        
        if homography_matrix is None:
            return PerspectiveCorrectionResult(
                canonical_card=np.zeros((canonical_height, canonical_width, 3), dtype=np.uint8),
                canonical_width=canonical_width,
                canonical_height=canonical_height,
                homography_matrix=None,
                inverse_homography=None,
                source_corners=source_corners,
                destination_corners=destination_corners,
                transformation_success=False,
                error_message="Failed to compute homography"
            )
        
        # Apply perspective transformation
        canonical_card = cv2.warpPerspective(
            image,
            homography_matrix,
            (canonical_width, canonical_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Compute inverse homography for coordinate transformation back
        inverse_homography = np.linalg.inv(homography_matrix)
        
        return PerspectiveCorrectionResult(
            canonical_card=canonical_card,
            canonical_width=canonical_width,
            canonical_height=canonical_height,
            homography_matrix=homography_matrix,
            inverse_homography=inverse_homography,
            source_corners=source_corners,
            destination_corners=destination_corners,
            transformation_success=True
        )
        
    except Exception as e:
        return PerspectiveCorrectionResult(
            canonical_card=np.zeros((canonical_height, canonical_width, 3), dtype=np.uint8),
            canonical_width=canonical_width,
            canonical_height=canonical_height,
            homography_matrix=None,
            inverse_homography=None,
            source_corners=source_corners,
            destination_corners=destination_corners,
            transformation_success=False,
            error_message=f"Homography computation failed: {str(e)}"
        )


def transform_point_from_canonical_to_processing(
    x: float,
    y: float,
    correction_result: PerspectiveCorrectionResult
) -> Optional[Tuple[float, float]]:
    """
    Transform point from canonical card space to processing image space.
    
    Uses inverse homography to map points back.
    
    Args:
        x: X coordinate in canonical space
        y: Y coordinate in canonical space
        correction_result: Result from perspective correction
        
    Returns:
        Transformed (x, y) in processing image coordinates, or None if transformation fails
    """
    if correction_result.inverse_homography is None:
        return None
    
    # Create homogeneous coordinate
    point = np.array([[x, y]], dtype=np.float32)
    point_homogeneous = np.ones((1, 3), dtype=np.float32)
    point_homogeneous[0, :2] = point
    
    # Apply inverse homography
    transformed = cv2.perspectiveTransform(
        point_homogeneous.reshape(1, -1, 2).astype(np.float32),
        correction_result.inverse_homography
    )
    
    result = transformed[0, 0]
    return (float(result[0]), float(result[1]))


def transform_bbox_from_canonical_to_processing(
    bbox: Tuple[float, float, float, float],
    correction_result: PerspectiveCorrectionResult
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
    Transform bounding box from canonical card space to processing image space.
    
    Transforms all 4 corners of the bbox.
    
    Args:
        bbox: (x, y, w, h) in canonical coordinates
        correction_result: Result from perspective correction
        
    Returns:
        Four corners in processing image space, or None if transformation fails
    """
    if correction_result.inverse_homography is None:
        return None
    
    x, y, w, h = bbox
    
    # Define 4 corners of bbox in canonical space
    corners_canonical = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)
    
    # Transform corners
    corners_transformed = cv2.perspectiveTransform(
        corners_canonical.reshape(1, -1, 2),
        correction_result.inverse_homography
    )
    
    return tuple(tuple(c) for c in corners_transformed[0])


def validate_rectification(result: PerspectiveCorrectionResult) -> Tuple[bool, str]:
    """
    Validate perspective correction result.
    
    Args:
        result: PerspectiveCorrectionResult to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not result.transformation_success:
        return False, result.error_message or "Transformation failed"
    
    if result.canonical_card is None:
        return False, "Canonical card is None"
    
    expected_height, expected_width = result.canonical_height, result.canonical_width
    actual_height, actual_width = result.canonical_card.shape[:2]
    
    if actual_height != expected_height or actual_width != expected_width:
        return False, f"Dimension mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    
    # Check if canonical card has meaningful content (not all black)
    mean_brightness = np.mean(result.canonical_card)
    if mean_brightness < 5:
        return False, "Canonical card appears to be empty (all black)"
    
    return True, ""
