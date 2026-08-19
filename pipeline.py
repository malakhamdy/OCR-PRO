"""
Egyptian National ID OCR Pipeline

Main processing pipeline integrating all modules.
Orchestrates the complete document understanding workflow.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Import all pipeline modules
from config import IMAGE_CONFIG, DETECTION_CONFIG, OCR_CONFIG
from image_normalization import normalize_image, ImageNormalizationResult
from card_detection import detect_card, CardDetectionResult, validate_card_detection
from perspective_correction import rectify_card, PerspectiveCorrectionResult, validate_rectification
from side_classification import classify_side, SideClassificationResult
from field_localization import localize_fields_on_canonical_card, LocalizationResult, extract_field_crop, get_field_by_name
from field_preprocessing import generate_preprocessing_variants, assess_crop_quality
from ocr_engine import perform_multi_pass_ocr, extract_text_from_ocr_result, get_ocr_confidence
from arabic_normalization import normalize_text_for_field
from nid_validation import validate_national_id, mask_nid
from cross_field_validation import perform_cross_field_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """Represents an extracted and processed field."""
    field_name: str
    raw_value: str
    normalized_value: str
    ocr_confidence: float
    localization_confidence: float
    validation_status: Optional[str]
    verification_status: str  # EXTRACTED, VERIFIED, CROSS_VALIDATED, LOW_CONFIDENCE
    bbox: Optional[tuple]
    coordinate_space: str
    sources: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_name': self.field_name,
            'raw_value': self.raw_value,
            'normalized_value': self.normalized_value,
            'ocr_confidence': self.ocr_confidence,
            'localization_confidence': self.localization_confidence,
            'validation_status': self.validation_status,
            'verification_status': self.verification_status,
            'bbox': self.bbox,
            'coordinate_space': self.coordinate_space,
            'sources': self.sources
        }


@dataclass
class PipelineResult:
    """Complete pipeline processing result."""
    success: bool
    document_info: Dict[str, Any]
    fields: Dict[str, ExtractedField]
    derived_info: Dict[str, Any]
    cross_validation: Optional[Any]
    debug_info: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'document_info': self.document_info,
            'fields': {k: v.to_dict() for k, v in self.fields.items()},
            'derived_info': self.derived_info,
            'cross_validation': self.cross_validation.to_dict() if self.cross_validation else None,
            'debug_info': self.debug_info,
            'error_message': self.error_message
        }


class EgyptianIDPipeline:
    """
    Complete Egyptian National ID OCR pipeline.
    
    Implements the full workflow from image input to structured output.
    """
    
    def __init__(self):
        self.debug_images = {}
        self.intermediate_results = {}
    
    def process_image(self, image: np.ndarray) -> PipelineResult:
        """
        Process Egyptian National ID image through complete pipeline.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PipelineResult with all extracted information
        """
        debug_info = {}
        fields = {}
        derived_info = {}
        
        try:
            # ========== STEP 1: Image Normalization ==========
            logger.info("Step 1: Image normalization")
            norm_result = normalize_image(image)
            debug_info['normalization'] = norm_result.to_dict()
            self.debug_images['processing_canvas'] = norm_result.processing_image
            
            # ========== STEP 2: Card Detection ==========
            logger.info("Step 2: Card detection")
            detection_result = detect_card(norm_result.processing_image)
            debug_info['card_detection'] = detection_result.to_dict()
            
            if not detection_result.card_detected:
                return PipelineResult(
                    success=False,
                    document_info={},
                    fields={},
                    derived_info={},
                    cross_validation=None,
                    debug_info=debug_info,
                    error_message="Card not detected in image"
                )
            
            # Validate detection
            is_valid_det, det_error = validate_card_detection(
                detection_result, 
                norm_result.processing_image.shape
            )
            if not is_valid_det:
                logger.warning(f"Card detection validation warning: {det_error}")
            
            self.debug_images['card_detection'] = self._draw_card_corners(
                norm_result.processing_image.copy(),
                detection_result.corners
            )
            
            # ========== STEP 3: Perspective Correction ==========
            logger.info("Step 3: Perspective correction")
            rectification_result = rectify_card(
                norm_result.processing_image,
                detection_result
            )
            debug_info['rectification'] = rectification_result.to_dict()
            self.debug_images['canonical_card'] = rectification_result.canonical_card
            
            if not rectification_result.transformation_success:
                return PipelineResult(
                    success=False,
                    document_info={},
                    fields={},
                    derived_info={},
                    cross_validation=None,
                    debug_info=debug_info,
                    error_message="Failed to rectify card"
                )
            
            # ========== STEP 4: Side Classification ==========
            logger.info("Step 4: Side classification")
            side_result = classify_side(rectification_result.canonical_card)
            debug_info['side_classification'] = side_result.to_dict()
            
            # ========== STEP 5: Field Localization ==========
            logger.info("Step 5: Field localization")
            localization_result = localize_fields_on_canonical_card(
                rectification_result.canonical_card,
                side_result.side
            )
            debug_info['localization'] = localization_result.to_dict()
            
            if not localization_result.fields:
                return PipelineResult(
                    success=False,
                    document_info={},
                    fields={},
                    derived_info={},
                    cross_validation=None,
                    debug_info=debug_info,
                    error_message="No fields localized"
                )
            
            # Visualize localization
            self.debug_images['field_localization'] = self._draw_field_boxes(
                rectification_result.canonical_card.copy(),
                localization_result.fields
            )
            
            # ========== STEP 6: Field Extraction & OCR ==========
            logger.info("Step 6: Field extraction and OCR")
            extracted_fields = {}
            
            for field_loc in localization_result.fields:
                field_name = field_loc.field_name
                field_type = field_loc.field_type
                
                # Extract crop
                crop = extract_field_crop(rectification_result.canonical_card, field_loc)
                
                if crop is None or crop.size == 0:
                    logger.warning(f"Failed to extract crop for field: {field_name}")
                    continue
                
                # Assess crop quality
                quality_metrics = assess_crop_quality(crop)
                
                # Generate preprocessing variants
                preprocessed_variants = generate_preprocessing_variants(crop, field_type)
                
                # Perform multi-pass OCR
                ocr_result = perform_multi_pass_ocr(preprocessed_variants, field_name, field_type)
                
                # Extract text
                raw_text = extract_text_from_ocr_result(ocr_result)
                ocr_confidence = get_ocr_confidence(ocr_result)
                
                # Normalize text based on field type
                normalized_text_obj = normalize_text_for_field(raw_text, field_type)
                
                # Create extracted field
                extracted_field = ExtractedField(
                    field_name=field_name,
                    raw_value=normalized_text_obj.raw_text,
                    normalized_value=normalized_text_obj.normalized_text,
                    ocr_confidence=ocr_confidence,
                    localization_confidence=field_loc.confidence,
                    validation_status=None,  # Will be set later
                    verification_status="EXTRACTED",
                    bbox=field_loc.bbox,
                    coordinate_space=field_loc.coordinate_space,
                    sources=[f"ocr_{ocr_result.ocr_engine}"]
                )
                
                extracted_fields[field_name] = extracted_field
            
            # ========== STEP 7: NID Validation ==========
            logger.info("Step 7: NID validation")
            nid_result = None
            if 'national_id' in extracted_fields:
                nid_raw = extracted_fields['national_id'].normalized_value
                if nid_raw:
                    nid_result = validate_national_id(nid_raw)
                    debug_info['nid_validation'] = nid_result.to_dict()
                    
                    # Update NID field status
                    if nid_result.is_valid:
                        extracted_fields['national_id'].validation_status = "VALID"
                        extracted_fields['national_id'].verification_status = "VERIFIED"
                    else:
                        extracted_fields['national_id'].validation_status = nid_result.validation_status
                    
                    # Add derived info
                    if nid_result.derived_info:
                        derived_info['nid_century'] = nid_result.derived_info.get('century')
                        derived_info['nid_birth_date'] = nid_result.derived_info.get('birth_date', {}).get('date_string')
                        derived_info['nid_governorate'] = nid_result.derived_info.get('governorate_name')
            
            # ========== STEP 8: Cross-Field Validation ==========
            logger.info("Step 8: Cross-field validation")
            cross_val_result = perform_cross_field_validation(
                {k: v.to_dict() for k, v in extracted_fields.items()},
                nid_result
            )
            debug_info['cross_validation'] = cross_val_result.to_dict()
            
            # Update field verification statuses based on cross-validation
            if cross_val_result.overall_status == "CROSS_VALIDATED":
                for check in cross_val_result.checks:
                    if check.passed:
                        if 'dob' in check.check_name.lower() and 'date_of_birth' in extracted_fields:
                            extracted_fields['date_of_birth'].verification_status = "CROSS_VALIDATED"
                        if 'governorate' in check.check_name.lower() and 'birth_governorate' in extracted_fields:
                            extracted_fields['birth_governorate'].verification_status = "CROSS_VALIDATED"
            
            # ========== Build Document Info ==========
            document_info = {
                'is_egyptian_id': True,
                'side': side_result.side,
                'side_confidence': side_result.confidence,
                'card_detection_confidence': detection_result.confidence,
                'template': localization_result.template_id,
                'image_quality': {
                    'card_area_ratio': detection_result.card_area_ratio,
                    'aspect_ratio': detection_result.aspect_ratio
                }
            }
            
            # ========== Build Final Result ==========
            result = PipelineResult(
                success=True,
                document_info=document_info,
                fields=extracted_fields,
                derived_info=derived_info,
                cross_validation=cross_val_result,
                debug_info=debug_info
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                document_info={},
                fields={},
                derived_info={},
                cross_validation=None,
                debug_info=debug_info,
                error_message=f"Pipeline error: {str(e)}"
            )
    
    def _draw_card_corners(self, image: np.ndarray, corners: list) -> np.ndarray:
        """Draw detected card corners on image."""
        for i, corner in enumerate(corners):
            cv2.circle(image, (int(corner.x), int(corner.y)), 8, (0, 255, 0), -1)
        
        # Draw contour lines
        if len(corners) == 4:
            pts = np.array([[c.x, c.y] for c in corners], dtype=np.int32)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        
        return image
    
    def _draw_field_boxes(self, image: np.ndarray, fields: list) -> np.ndarray:
        """Draw field bounding boxes on image."""
        colors = {
            'numeric': (255, 0, 0),
            'arabic_text': (0, 255, 0),
            'date': (0, 0, 255),
            'arabic_short': (255, 255, 0),
            'arabic_multiline': (255, 0, 255)
        }
        
        for field in fields:
            x, y, w, h = field.bbox
            color = colors.get(field.field_type, (128, 128, 128))
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Add label
            label = f"{field.field_name}"
            cv2.putText(image, label, (int(x), int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image


# Singleton instance for Streamlit caching
_pipeline_instance = None

def get_pipeline() -> EgyptianIDPipeline:
    """Get or create pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EgyptianIDPipeline()
    return _pipeline_instance
