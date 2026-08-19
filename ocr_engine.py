"""
Arabic OCR Module

Arabic-first OCR using PaddleOCR with Arabic-capable models.
Supports multi-pass OCR with different preprocessing variants.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from config import OCR_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OcrCandidate:
    """Single OCR candidate result."""
    text: str
    confidence: float
    language: str
    preprocessing_variant: str
    bbox: Optional[Tuple[float, float, float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'preprocessing_variant': self.preprocessing_variant,
            'bbox': self.bbox
        }


@dataclass
class OcrResult:
    """Complete OCR result for a field."""
    field_name: str
    best_candidate: OcrCandidate
    all_candidates: List[OcrCandidate]
    num_variants_tried: int
    ocr_engine: str
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_name': self.field_name,
            'best_candidate': self.best_candidate.to_dict(),
            'all_candidates': [c.to_dict() for c in self.all_candidates],
            'num_variants_tried': self.num_variants_tried,
            'ocr_engine': self.ocr_engine,
            'error_message': self.error_message
        }


class PaddleOcrEngine:
    """
    PaddleOCR engine wrapper for Arabic OCR.
    
    Loads model once and reuses for multiple OCR operations.
    Uses Arabic-capable recognition model.
    """
    
    _instance = None
    _ocr_system = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PaddleOcrEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._load_model()
    
    def _load_model(self):
        """Load PaddleOCR model with Arabic support."""
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Loading PaddleOCR with Arabic model...")
            
            # Use Arabic language model with updated PaddleOCR v3.x API
            # Parameters have been updated in newer versions
            self._ocr_system = PaddleOCR(
                lang=OCR_CONFIG.PADDLE_LANG,  # 'ar' for Arabic
                text_det_thresh=OCR_CONFIG.PADDLE_DET_DB_THRESH,
                text_det_box_thresh=OCR_CONFIG.PADDLE_DET_BOX_THRESH,
                text_recognition_batch_size=OCR_CONFIG.PADDLE_REC_BATCH_NUM
            )
            
            logger.info("PaddleOCR loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            self._ocr_system = None
    
    def is_available(self) -> bool:
        """Check if OCR engine is available."""
        return self._ocr_system is not None
    
    def ocr_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform OCR on an image.
        
        Args:
            image: Input image (can be grayscale or BGR)
            
        Returns:
            List of detection results with text, confidence, and bbox
        """
        if self._ocr_system is None:
            return []
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:
                # Grayscale - convert to BGR for PaddleOCR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Run OCR - updated API for PaddleOCR v3.x
            result = self._ocr_system.ocr(image)
            
            # Parse results
            parsed_results = []
            
            if result and result[0]:
                for detection in result[0]:
                    if detection and len(detection) >= 2:
                        bbox_points = detection[0]  # 4 corner points
                        text = detection[1][0] if len(detection[1]) > 0 else ""
                        confidence = detection[1][1] if len(detection[1]) > 1 else 0.0
                        
                        # Convert bbox to (x, y, w, h) format
                        if bbox_points and len(bbox_points) >= 4:
                            x_coords = [p[0] for p in bbox_points]
                            y_coords = [p[1] for p in bbox_points]
                            x_min, y_min = min(x_coords), min(y_coords)
                            x_max, y_max = max(x_coords), max(y_coords)
                            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                        else:
                            bbox = None
                        
                        parsed_results.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []


def perform_ocr_on_crop(
    crop: np.ndarray,
    field_name: str,
    preprocessing_variant_name: str = "default"
) -> List[OcrCandidate]:
    """
    Perform OCR on a preprocessed field crop.
    
    Args:
        crop: Preprocessed field crop image
        field_name: Name of the field being processed
        preprocessing_variant_name: Name of preprocessing variant used
        
    Returns:
        List of OcrCandidate objects
    """
    engine = PaddleOcrEngine()
    
    if not engine.is_available():
        return [
            OcrCandidate(
                text="",
                confidence=0.0,
                language="ar",
                preprocessing_variant=preprocessing_variant_name,
                bbox=None
            )
        ]
    
    # Run OCR
    detections = engine.ocr_image(crop)
    
    candidates = []
    for det in detections:
        candidate = OcrCandidate(
            text=det['text'],
            confidence=det['confidence'],
            language='ar',
            preprocessing_variant=preprocessing_variant_name,
            bbox=det['bbox']
        )
        candidates.append(candidate)
    
    # If no text detected, create empty candidate
    if not candidates:
        candidates.append(
            OcrCandidate(
                text="",
                confidence=0.0,
                language="ar",
                preprocessing_variant=preprocessing_variant_name,
                bbox=None
            )
        )
    
    return candidates


def perform_multi_pass_ocr(
    preprocessed_variants: List[Any],  # List[PreprocessedCrop]
    field_name: str,
    field_type: str
) -> OcrResult:
    """
    Perform multi-pass OCR with different preprocessing variants.
    
    Args:
        preprocessed_variants: List of PreprocessedCrop objects
        field_name: Name of the field
        field_type: Type of field
        
    Returns:
        OcrResult with best candidate and all candidates
    """
    all_candidates = []
    num_variants = len(preprocessed_variants)
    
    for variant in preprocessed_variants:
        try:
            candidates = perform_ocr_on_crop(
                crop=variant.crop,
                field_name=field_name,
                preprocessing_variant_name=variant.variant_name
            )
            all_candidates.extend(candidates)
        except Exception as e:
            logger.warning(f"OCR variant {variant.variant_name} failed: {e}")
    
    # Select best candidate based on confidence
    if all_candidates:
        # Filter out empty candidates
        non_empty = [c for c in all_candidates if c.text.strip()]
        
        if non_empty:
            # Sort by confidence
            non_empty_sorted = sorted(non_empty, key=lambda c: c.confidence, reverse=True)
            best_candidate = non_empty_sorted[0]
        else:
            # All candidates empty
            best_candidate = all_candidates[0]
    else:
        best_candidate = OcrCandidate(
            text="",
            confidence=0.0,
            language="ar",
            preprocessing_variant="none",
            bbox=None
        )
    
    return OcrResult(
        field_name=field_name,
        best_candidate=best_candidate,
        all_candidates=all_candidates,
        num_variants_tried=num_variants,
        ocr_engine="paddleocr_arabic"
    )


def extract_text_from_ocr_result(ocr_result: OcrResult) -> str:
    """
    Extract consolidated text from OCR result.
    
    For single-line fields, returns the highest confidence text.
    For multi-line fields, concatenates all detected lines.
    
    Args:
        ocr_result: Result from multi-pass OCR
        
    Returns:
        Consolidated text string
    """
    if not ocr_result.all_candidates:
        return ""
    
    # Get non-empty candidates
    non_empty = [c for c in ocr_result.all_candidates if c.text.strip()]
    
    if not non_empty:
        return ""
    
    # For now, return best candidate text
    # Can be enhanced to merge multiple detections
    return ocr_result.best_candidate.text


def get_ocr_confidence(ocr_result: OcrResult) -> float:
    """
    Get overall OCR confidence score.
    
    Args:
        ocr_result: Result from multi-pass OCR
        
    Returns:
        Confidence score between 0 and 1
    """
    if not ocr_result.all_candidates:
        return 0.0
    
    non_empty = [c for c in ocr_result.all_candidates if c.text.strip()]
    
    if not non_empty:
        return 0.0
    
    # Return best candidate confidence
    return max(c.confidence for c in non_empty)
