"""
Card Detection Module

Detects Egyptian National ID card in input image.
Finds card corners and estimates perspective transformation.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

from config import DETECTION_CONFIG, IMAGE_CONFIG


@dataclass
class CardCorner:
    """Represents a detected card corner."""
    x: float
    y: float
    confidence: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class CardDetectionResult:
    """Result of card detection."""
    card_detected: bool
    confidence: float
    corners: List[CardCorner]  # TL, TR, BR, BL order
    contour: Optional[np.ndarray]
    card_area_ratio: float
    aspect_ratio: float
    error_message: Optional[str] = None
    
    def get_ordered_corners(self) -> Optional[np.ndarray]:
        """Get corners as numpy array in ordered format for homography."""
        if not self.corners or len(self.corners) != 4:
            return None
        
        # Order: top-left, top-right, bottom-right, bottom-left
        ordered = np.array([
            self.corners[0].to_tuple(),
            self.corners[1].to_tuple(),
            self.corners[2].to_tuple(),
            self.corners[3].to_tuple()
        ], dtype=np.float32)
        
        return ordered
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'card_detected': self.card_detected,
            'confidence': self.confidence,
            'corners': [(c.x, c.y) for c in self.corners],
            'card_area_ratio': self.card_area_ratio,
            'aspect_ratio': self.aspect_ratio,
            'error_message': self.error_message
        }


def detect_card(image: np.ndarray) -> CardDetectionResult:
    """
    Detect Egyptian National ID card in image.
    
    Uses edge detection, contour finding, and geometric validation.
    
    Args:
        image: Input image (BGR format, already normalized)
        
    Returns:
        CardDetectionResult with detected card information
    """
    height, width = image.shape[:2]
    image_area = height * width
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny with adaptive thresholds
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(
        closed_edges.copy(), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # Strategy 2: Try threshold-based detection
        return _detect_by_threshold(gray, image_area, height, width)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    min_area_ratio = DETECTION_CONFIG.MIN_CONTOUR_AREA_RATIO
    expected_aspect_ratio = DETECTION_CONFIG.EXPECTED_ASPECT_RATIO
    max_aspect_deviation = DETECTION_CONFIG.MAX_ASPECT_RATIO_DEVIATION
    
    best_contour = None
    best_score = 0.0
    best_corners = None
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = DETECTION_CONFIG.CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if contour has 4 points (quadrilateral)
        if len(approx) != 4:
            continue
        
        # Check contour area
        area = cv2.contourArea(approx)
        area_ratio = area / image_area
        
        if area_ratio < min_area_ratio:
            continue
        
        # Get bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        contour_aspect_ratio = w / float(h) if h > 0 else 0
        
        # Check aspect ratio deviation
        aspect_deviation = abs(contour_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio
        
        if aspect_deviation > max_aspect_deviation:
            continue
        
        # Calculate score based on area and aspect ratio match
        score = area_ratio * (1.0 - min(aspect_deviation, 1.0))
        
        if score > best_score:
            best_score = score
            best_contour = approx
            best_corners = approx.reshape(4, 2)
    
    if best_contour is None or best_corners is None:
        # Fallback: try to find any large rectangular contour
        for contour in contours[:5]:  # Check top 5 largest contours
            area = cv2.contourArea(contour)
            area_ratio = area / image_area
            
            if area_ratio < min_area_ratio * 0.5:
                continue
            
            epsilon = DETECTION_CONFIG.CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                best_contour = approx
                # Get 4 corners from minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                best_corners = order_corners(box)
                best_score = area_ratio
                break
    
    if best_contour is None:
        return CardDetectionResult(
            card_detected=False,
            confidence=0.0,
            corners=[],
            contour=None,
            card_area_ratio=0.0,
            aspect_ratio=0.0,
            error_message="No valid card contour found"
        )
    
    # Order corners: TL, TR, BR, BL
    if best_corners is None:
        best_corners = order_corners(best_contour.reshape(-1, 2))
    elif len(best_corners) == 4:
        best_corners = order_corners(best_corners)
    
    # Calculate metrics
    area = cv2.contourArea(best_contour)
    card_area_ratio = area / image_area
    
    # Calculate aspect ratio from ordered corners
    tl, tr, br, bl = best_corners
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    aspect_ratio = avg_width / avg_height if avg_height > 0 else 0
    
    # Create corner objects with confidence
    corners = [
        CardCorner(x=c[0], y=c[1], confidence=best_score)
        for c in best_corners
    ]
    
    # Determine overall confidence
    confidence = min(best_score * 2.0, 1.0)  # Scale score to confidence
    
    # Boost confidence for valid detections with good area ratio and aspect ratio
    if card_area_ratio > 0.15 and abs(aspect_ratio - DETECTION_CONFIG.EXPECTED_ASPECT_RATIO) < 0.3:
        confidence = max(confidence, 0.6)
    
    return CardDetectionResult(
        card_detected=confidence >= DETECTION_CONFIG.MIN_DETECTION_CONFIDENCE,
        confidence=confidence,
        corners=corners,
        contour=best_contour,
        card_area_ratio=card_area_ratio,
        aspect_ratio=aspect_ratio
    )


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners in clockwise direction starting from top-left.
    
    Args:
        corners: Array of 4 corners (x, y)
        
    Returns:
        Ordered corners: TL, TR, BR, BL
    """
    # Sort by y-coordinate to find top and bottom points
    y_sorted = corners[corners[:, 1].argsort()]
    
    # Top two points
    top_two = y_sorted[:2]
    # Bottom two points
    bottom_two = y_sorted[2:]
    
    # Sort top two by x-coordinate (left to right)
    tl = top_two[top_two[:, 0].argsort()][0]
    tr = top_two[top_two[:, 0].argsort()][1]
    
    # Sort bottom two by x-coordinate (right to left for clockwise)
    br = bottom_two[bottom_two[:, 0].argsort()][1]
    bl = bottom_two[bottom_two[:, 0].argsort()][0]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_by_threshold(gray: np.ndarray, image_area: float, height: int, width: int) -> CardDetectionResult:
    """
    Fallback detection strategy using thresholding.
    
    Useful for low-contrast images where edge detection fails.
    """
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return _detect_full_image_as_card(height, width)
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    min_area_ratio = DETECTION_CONFIG.MIN_CONTOUR_AREA_RATIO * 0.5  # More lenient
    expected_aspect_ratio = DETECTION_CONFIG.EXPECTED_ASPECT_RATIO
    max_aspect_deviation = DETECTION_CONFIG.MAX_ASPECT_RATIO_DEVIATION
    
    for contour in contours[:10]:  # Check top 10
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        if area_ratio < min_area_ratio:
            continue
        
        # Approximate to polygon
        epsilon = DETECTION_CONFIG.CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 4:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            aspect_deviation = abs(aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio
            
            if aspect_deviation <= max_aspect_deviation or area_ratio > 0.5:
                # Found a good candidate
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                corners = order_corners(box)
                
                card_area_ratio = area / image_area
                confidence = min(area_ratio * 1.5, 1.0)
                
                corner_objects = [
                    CardCorner(x=c[0], y=c[1], confidence=confidence)
                    for c in corners
                ]
                
                return CardDetectionResult(
                    card_detected=confidence >= DETECTION_CONFIG.MIN_DETECTION_CONFIDENCE * 0.8,
                    confidence=confidence,
                    corners=corner_objects,
                    contour=approx,
                    card_area_ratio=card_area_ratio,
                    aspect_ratio=aspect_ratio
                )
    
    # No good contour found, use full image
    return _detect_full_image_as_card(height, width)


def _detect_full_image_as_card(height: int, width: int) -> CardDetectionResult:
    """
    Treat the entire image as the card.
    
    Used as final fallback when no distinct card boundary is detected.
    This is common when the uploaded image IS the card (no background).
    """
    expected_ar = DETECTION_CONFIG.EXPECTED_ASPECT_RATIO
    actual_ar = width / float(height) if height > 0 else 0
    ar_deviation = abs(actual_ar - expected_ar) / expected_ar
    
    # Confidence based on how close aspect ratio is to expected
    confidence = max(0.5, 1.0 - ar_deviation)
    
    corners = [
        CardCorner(x=0, y=0, confidence=confidence),  # TL
        CardCorner(x=width, y=0, confidence=confidence),  # TR
        CardCorner(x=width, y=height, confidence=confidence),  # BR
        CardCorner(x=0, y=height, confidence=confidence)  # BL
    ]
    
    return CardDetectionResult(
        card_detected=True,  # Always detected as we're using full image
        confidence=max(confidence, DETECTION_CONFIG.MIN_DETECTION_CONFIDENCE),  # Ensure meets threshold
        corners=corners,
        contour=None,
        card_area_ratio=1.0,
        aspect_ratio=actual_ar
    )


def validate_card_detection(result: CardDetectionResult, image_shape: Tuple) -> Tuple[bool, str]:
    """
    Validate card detection result.
    
    Args:
        result: CardDetectionResult to validate
        image_shape: Shape of the image (height, width)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not result.card_detected:
        return False, "Card not detected"
    
    if len(result.corners) != 4:
        return False, f"Expected 4 corners, got {len(result.corners)}"
    
    height, width = image_shape[:2]
    
    # Check all corners are within image bounds
    for corner in result.corners:
        if corner.x < 0 or corner.x >= width:
            return False, f"Corner x={corner.x} outside image bounds"
        if corner.y < 0 or corner.y >= height:
            return False, f"Corner y={corner.y} outside image bounds"
    
    # Check minimum distance between corners
    min_distance = DETECTION_CONFIG.MIN_CORNER_DISTANCE
    for i in range(len(result.corners)):
        for j in range(i + 1, len(result.corners)):
            dist = np.linalg.norm(
                np.array(result.corners[i].to_tuple()) - 
                np.array(result.corners[j].to_tuple())
            )
            if dist < min_distance:
                return False, f"Corners too close: distance={dist}"
    
    # Check aspect ratio
    expected_ar = DETECTION_CONFIG.EXPECTED_ASPECT_RATIO
    max_deviation = DETECTION_CONFIG.MAX_ASPECT_RATIO_DEVIATION
    ar_deviation = abs(result.aspect_ratio - expected_ar) / expected_ar
    
    if ar_deviation > max_deviation:
        return False, f"Aspect ratio deviation too high: {ar_deviation:.2f}"
    
    return True, ""
