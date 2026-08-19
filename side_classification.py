"""
Side Classification Module

Determines whether uploaded image shows front or back of Egyptian National ID.
Uses visual/layout cues for classification.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

from config import FIELD_CONFIG


@dataclass
class SideClassificationResult:
    """Result of side classification."""
    side: str  # 'front' or 'back'
    confidence: float
    cues_detected: List[str]
    front_score: float
    back_score: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'side': self.side,
            'confidence': self.confidence,
            'cues_detected': self.cues_detected,
            'front_score': self.front_score,
            'back_score': self.back_score,
            'error_message': self.error_message
        }


def classify_side(canonical_card: np.ndarray) -> SideClassificationResult:
    """
    Classify whether canonical card image is front or back side.
    
    Uses multiple cues:
    - Portrait/photo region detection (front)
    - Barcode region detection (back)
    - Text layout patterns
    - Field distribution
    
    Args:
        canonical_card: Rectified card image
        
    Returns:
        SideClassificationResult with classification decision
    """
    height, width = canonical_card.shape[:2]
    
    cues_detected = []
    front_score = 0.0
    back_score = 0.0
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(canonical_card, cv2.COLOR_BGR2GRAY)
    
    # ========== FRONT SIDE CUES ==========
    
    # Cue 1: Portrait/photo region detection (typically on left side of front)
    # Look for rectangular region with specific texture characteristics
    portrait_region = gray[0:int(height*0.45), 0:int(width*0.30)]
    if portrait_region.size > 0:
        portrait_variance = np.var(portrait_region)
        # Photo regions typically have higher variance than background
        if portrait_variance > 800:
            front_score += 0.3
            cues_detected.append("portrait_region_detected")
    
    # Cue 2: Check for structured text fields in expected front locations
    # Front has NID, name, DOB, gender in specific pattern
    upper_middle_region = gray[int(height*0.15):int(height*0.45), int(width*0.30):int(width*0.95)]
    if upper_middle_region.size > 0:
        # Count edges/text density
        edges = cv2.Canny(upper_middle_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.05:
            front_score += 0.25
            cues_detected.append("structured_text_fields_detected")
    
    # Cue 3: Check for Arabic text patterns in name field region
    name_region_y_start = int(height * FIELD_CONFIG.FRONT_FIELDS['name']['y_norm'][0])
    name_region_y_end = int(height * FIELD_CONFIG.FRONT_FIELDS['name']['y_norm'][1])
    name_region_x_start = int(width * FIELD_CONFIG.FRONT_FIELDS['name']['x_norm'][0])
    name_region_x_end = int(width * FIELD_CONFIG.FRONT_FIELDS['name']['x_norm'][1])
    
    name_region = gray[name_region_y_start:name_region_y_end, name_region_x_start:name_region_x_end]
    if name_region.size > 0:
        # Arabic text has characteristic stroke patterns
        binary = cv2.adaptiveThreshold(name_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        text_pixel_ratio = np.sum(binary > 0) / binary.size
        if 0.1 < text_pixel_ratio < 0.5:
            front_score += 0.2
            cues_detected.append("name_field_text_detected")
    
    # ========== BACK SIDE CUES ==========
    
    # Cue 1: Barcode/PDF417 detection (typically in upper portion of back)
    barcode_region = gray[int(height*0.10):int(height*0.45), int(width*0.05):int(width*0.95)]
    if barcode_region.size > 0:
        # Barcodes have distinctive horizontal line patterns
        # Use horizontal Sobel to detect horizontal lines
        sobel_x = cv2.Sobel(barcode_region.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_abs = np.abs(sobel_x)
        horizontal_line_strength = np.mean(sobel_x_abs)
        
        if horizontal_line_strength > 15:
            back_score += 0.4
            cues_detected.append("barcode_region_detected")
    
    # Cue 2: Back side typically has more uniform text distribution
    # Check lower half for additional text blocks
    lower_half = gray[int(height*0.50):, :]
    if lower_half.size > 0:
        edges_lower = cv2.Canny(lower_half, 50, 150)
        edge_density_lower = np.sum(edges_lower > 0) / edges_lower.size
        if edge_density_lower > 0.03:
            back_score += 0.2
            cues_detected.append("back_text_block_detected")
    
    # ========== NORMALIZE SCORES ==========
    
    total_score = front_score + back_score
    if total_score > 0:
        front_score = front_score / total_score
        back_score = back_score / total_score
    else:
        # Default to front if no strong cues
        front_score = 0.5
        back_score = 0.5
    
    # Determine final classification
    if front_score >= back_score:
        side = 'front'
        confidence = front_score
    else:
        side = 'back'
        confidence = back_score
    
    # Boost confidence if one side is clearly dominant
    score_diff = abs(front_score - back_score)
    if score_diff > 0.3:
        confidence = min(confidence + 0.1, 1.0)
        cues_detected.append("clear_side_dominance")
    
    # Ensure minimum confidence
    if confidence < 0.5:
        confidence = 0.5
        cues_detected.append("low_confidence_classification")
    
    return SideClassificationResult(
        side=side,
        confidence=confidence,
        cues_detected=cues_detected,
        front_score=front_score,
        back_score=back_score
    )


def validate_side_classification(result: SideClassificationResult) -> Tuple[bool, str]:
    """
    Validate side classification result.
    
    Args:
        result: SideClassificationResult to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if result.side not in ['front', 'back']:
        return False, f"Invalid side: {result.side}"
    
    if result.confidence < 0.0 or result.confidence > 1.0:
        return False, f"Invalid confidence: {result.confidence}"
    
    if not result.cues_detected:
        return False, "No classification cues detected"
    
    return True, ""
