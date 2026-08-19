"""
Field Localization Module

Dynamically localizes fields on canonical card using normalized coordinates.
Supports scale-invariant field detection based on card geometry.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

from config import FIELD_CONFIG


@dataclass
class FieldLocation:
    """Represents a localized field location."""
    field_name: str
    bbox: Tuple[float, float, float, float]  # x, y, w, h in canonical coordinates
    coordinate_space: str  # 'canonical_card'
    field_type: str
    required: bool
    confidence: float
    anchors_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_name': self.field_name,
            'bbox': self.bbox,
            'coordinate_space': self.coordinate_space,
            'field_type': self.field_type,
            'required': self.required,
            'confidence': self.confidence,
            'anchors_used': self.anchors_used
        }


@dataclass
class LocalizationResult:
    """Result of field localization."""
    side: str  # 'front' or 'back'
    fields: List[FieldLocation]
    template_id: Optional[str]
    localization_confidence: float
    warnings: List[str]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'side': self.side,
            'fields': [f.to_dict() for f in self.fields],
            'template_id': self.template_id,
            'localization_confidence': self.localization_confidence,
            'warnings': self.warnings,
            'error_message': self.error_message
        }


def normalize_bbox_from_regions(
    x_norm: Tuple[float, float],
    y_norm: Tuple[float, float],
    card_width: int,
    card_height: int,
    padding_factor_x: float = 0.05,
    padding_factor_y: float = 0.15
) -> Tuple[float, float, float, float]:
    """
    Convert normalized region coordinates to absolute bounding box.
    
    Args:
        x_norm: (x_min_norm, x_max_norm) normalized x range
        y_norm: (y_min_norm, y_max_norm) normalized y range
        card_width: Width of canonical card
        card_height: Height of canonical card
        padding_factor_x: Horizontal padding factor
        padding_factor_y: Vertical padding factor
        
    Returns:
        Bounding box (x, y, w, h) in canonical coordinates with padding
    """
    # Calculate base coordinates
    x_start = x_norm[0] * card_width
    x_end = x_norm[1] * card_width
    y_start = y_norm[0] * card_height
    y_end = y_norm[1] * card_height
    
    # Add padding
    field_width = x_end - x_start
    field_height = y_end - y_start
    
    padding_x = field_width * padding_factor_x
    padding_y = field_height * padding_factor_y
    
    # Apply padding (expand slightly to ensure full text capture)
    x_start_padded = max(0, x_start - padding_x / 2)
    x_end_padded = min(card_width, x_end + padding_x / 2)
    y_start_padded = max(0, y_start - padding_y / 2)
    y_end_padded = min(card_height, y_end + padding_y / 2)
    
    final_width = x_end_padded - x_start_padded
    final_height = y_end_padded - y_start_padded
    
    return (x_start_padded, y_start_padded, final_width, final_height)


def localize_fields_on_canonical_card(
    canonical_card: np.ndarray,
    side: str
) -> LocalizationResult:
    """
    Localize all expected fields on canonical card.
    
    Uses normalized coordinates from configuration for scale-invariant localization.
    
    Args:
        canonical_card: Rectified card image
        side: 'front' or 'back'
        
    Returns:
        LocalizationResult with all field locations
    """
    height, width = canonical_card.shape[:2]
    
    fields = []
    warnings = []
    anchors_used = []
    
    # Select appropriate field definitions based on side
    if side == 'front':
        field_definitions = FIELD_CONFIG.FRONT_FIELDS
    else:
        field_definitions = FIELD_CONFIG.BACK_FIELDS
    
    # Localize each field
    for field_name, field_def in field_definitions.items():
        try:
            x_norm = field_def['x_norm']
            y_norm = field_def['y_norm']
            field_type = field_def['type']
            required = field_def.get('required', False)
            
            # Convert normalized coordinates to absolute bounding box
            bbox = normalize_bbox_from_regions(
                x_norm=x_norm,
                y_norm=y_norm,
                card_width=width,
                card_height=height,
                padding_factor_x=FIELD_CONFIG.CROP_PADDING_X_FACTOR,
                padding_factor_y=FIELD_CONFIG.CROP_PADDING_Y_FACTOR
            )
            
            # Validate bbox is within card bounds
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > width or y + h > height:
                warnings.append(f"Field {field_name} bbox partially outside card bounds")
                # Clamp to valid region
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                bbox = (x, y, w, h)
            
            # Check minimum field size
            if w < 20 or h < 15:
                warnings.append(f"Field {field_name} too small: {w}x{h}")
                continue
            
            # Create field location object
            field_location = FieldLocation(
                field_name=field_name,
                bbox=bbox,
                coordinate_space='canonical_card',
                field_type=field_type,
                required=required,
                confidence=0.9,  # High confidence for template-based localization
                anchors_used=['template_normalized_coordinates']
            )
            
            fields.append(field_location)
            
        except Exception as e:
            if required:
                warnings.append(f"Failed to localize required field {field_name}: {str(e)}")
            else:
                warnings.append(f"Failed to localize optional field {field_name}: {str(e)}")
    
    # Calculate overall localization confidence
    if not fields:
        localization_confidence = 0.0
        error_msg = "No fields localized"
    else:
        # Count required fields that were successfully localized
        required_fields = [f for f in field_definitions.values() if f.get('required', False)]
        localized_required = sum(1 for f in fields if any(rf for rf in required_fields if True))
        
        if required_fields:
            required_coverage = len(fields) / len(required_fields)
        else:
            required_coverage = 1.0
        
        localization_confidence = min(0.9 + (required_coverage * 0.1), 1.0)
        error_msg = None
    
    # Determine template ID based on side and field pattern
    template_id = f"egyptian_id_{side}_v1"
    
    return LocalizationResult(
        side=side,
        fields=fields,
        template_id=template_id,
        localization_confidence=localization_confidence,
        warnings=warnings,
        error_message=error_msg
    )


def extract_field_crop(
    canonical_card: np.ndarray,
    field_location: FieldLocation
) -> Optional[np.ndarray]:
    """
    Extract crop for a specific field from canonical card.
    
    Args:
        canonical_card: Rectified card image
        field_location: FieldLocation with bbox information
        
    Returns:
        Cropped field image, or None if extraction fails
    """
    if field_location.coordinate_space != 'canonical_card':
        return None
    
    x, y, w, h = field_location.bbox
    
    # Ensure coordinates are integers and within bounds
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(canonical_card.shape[1], x + w)), int(min(canonical_card.shape[0], y + h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = canonical_card[y1:y2, x1:x2].copy()
    
    return crop


def get_field_by_name(localization_result: LocalizationResult, field_name: str) -> Optional[FieldLocation]:
    """
    Get specific field by name from localization result.
    
    Args:
        localization_result: Result from localize_fields_on_canonical_card
        field_name: Name of field to retrieve
        
    Returns:
        FieldLocation if found, None otherwise
    """
    for field in localization_result.fields:
        if field.field_name == field_name:
            return field
    return None


def validate_localization(result: LocalizationResult, card_shape: Tuple) -> Tuple[bool, str]:
    """
    Validate localization result.
    
    Args:
        result: LocalizationResult to validate
        card_shape: Shape of canonical card (height, width)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if result.error_message:
        return False, result.error_message
    
    if not result.fields:
        return False, "No fields localized"
    
    height, width = card_shape[:2]
    
    for field in result.fields:
        x, y, w, h = field.bbox
        
        # Check bounds
        if x < 0 or y < 0:
            return False, f"Field {field.field_name} has negative coordinates"
        
        if x + w > width or y + h > height:
            return False, f"Field {field.field_name} exceeds card bounds"
        
        # Check minimum size
        if w < 10 or h < 10:
            return False, f"Field {field.field_name} too small: {w}x{h}"
    
    return True, ""
