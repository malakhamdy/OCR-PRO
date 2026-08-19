"""
Arabic Text Normalization Module

Normalizes Arabic OCR output while preserving raw text.
Handles Unicode normalization, digit conversion, and common OCR artifacts.
"""

import re
import unicodedata
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NormalizedText:
    """Result of text normalization."""
    raw_text: str
    normalized_text: str
    transformations_applied: list
    confidence_adjustment: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw_text': self.raw_text,
            'normalized_text': self.normalized_text,
            'transformations_applied': self.transformations_applied,
            'confidence_adjustment': self.confidence_adjustment
        }


# Arabic-Indic digits to Western digits mapping
ARABIC_INDIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}

# Extended Arabic digit patterns (Persian/Urdu variants)
EXTENDED_ARABIC_DIGITS = {
    **ARABIC_INDIC_DIGITS
}

# Common Arabic character confusables
ARABIC_CONFUSABLES = {
    'أ': ['ا', 'آ'],  # Alif with hamza variations
    'إ': ['ا', 'آ'],
    'ؤ': ['و'],
    'ئ': ['ي', 'ى'],
    'ة': ['ه'],  # Ta marbuta vs ha
    'ى': ['ي'],  # Alif maqsura vs ya
    'گ': ['ك'],  # Persian gaf vs kaf
    'چ': ['ج'],  # Persian che vs jim
    'پ': ['ب'],  # Persian pe vs ba
}

# Common OCR artifacts in Arabic text
OCR_ARTIFACT_PATTERNS = [
    (r'[ـًٌٍَُِّْٰ]', ''),  # Remove diacritics (tashkeel)
    (r'ـ+', ''),  # Remove tatweel (elongation)
    (r'\s+', ' '),  # Normalize whitespace
    (r'^\s+|\s+$', ''),  # Trim leading/trailing whitespace
]


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode representation of Arabic text.
    
    Uses NFKC normalization for compatibility.
    
    Args:
        text: Input text
        
    Returns:
        Unicode-normalized text
    """
    return unicodedata.normalize('NFKC', text)


def convert_arabic_digits_to_western(text: str) -> str:
    """
    Convert Arabic-Indic digits to Western digits.
    
    Args:
        text: Text containing Arabic-Indic digits
        
    Returns:
        Text with Western digits
    """
    result = text
    for arabic_digit, western_digit in EXTENDED_ARABIC_DIGITS.items():
        result = result.replace(arabic_digit, western_digit)
    return result


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.
    
    Args:
        text: Text with diacritics
        
    Returns:
        Text without diacritics
    """
    # Remove common Arabic diacritical marks
    diacritics_pattern = r'[ًٌٍَُِّْٰٖٜٟٗ٘ٙٚٛٝٞ]'
    return re.sub(diacritics_pattern, '', text)


def remove_tatweel(text: str) -> str:
    """
    Remove tatweel (character elongation) from text.
    
    Args:
        text: Text with tatweel
        
    Returns:
        Text without tatweel
    """
    return re.sub(r'ـ+', '', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text


def normalize_arabic_characters(text: str) -> str:
    """
    Normalize common Arabic character variations.
    
    Standardizes different forms of the same logical character.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized characters
    """
    result = text
    
    # Normalize alif variations to bare alif
    # Note: This is conservative - we don't want to lose semantic meaning
    # Only normalize in specific contexts if needed
    
    # Normalize ta marbuta to ha for matching purposes
    # (Many systems treat them interchangeably)
    result = result.replace('ة', 'ه')
    
    # Normalize alif maqsura to ya
    result = result.replace('ى', 'ي')
    
    return result


def clean_ocr_artifacts(text: str) -> str:
    """
    Clean common OCR artifacts from Arabic text.
    
    Args:
        text: Raw OCR output
        
    Returns:
        Cleaned text
    """
    result = text
    
    for pattern, replacement in OCR_ARTIFACT_PATTERNS:
        result = re.sub(pattern, replacement, result)
    
    return result


def normalize_text_for_field(
    raw_text: str,
    field_type: str
) -> NormalizedText:
    """
    Normalize text based on field type.
    
    Different field types require different normalization strategies.
    
    Args:
        raw_text: Raw OCR output
        field_type: Type of field ('numeric', 'arabic_text', 'date', etc.)
        
    Returns:
        NormalizedText with raw and normalized versions
    """
    transformations = []
    normalized = raw_text
    confidence_adjustment = 0.0
    
    # Always preserve raw text
    raw_preserved = raw_text
    
    # Step 1: Unicode normalization
    normalized = normalize_unicode(normalized)
    transformations.append('unicode_normalize')
    
    # Step 2: Field-specific normalization
    if field_type == 'numeric':
        # For numeric fields (like National ID), aggressively convert digits
        normalized = convert_arabic_digits_to_western(normalized)
        transformations.append('convert_digits')
        
        # Remove any non-digit, non-space characters
        normalized = re.sub(r'[^\d\s]', '', normalized)
        transformations.append('remove_non_numeric')
        
        # Remove all whitespace for pure numeric
        normalized = normalized.replace(' ', '')
        transformations.append('remove_whitespace')
        
    elif field_type == 'date':
        # Convert digits but preserve separators
        normalized = convert_arabic_digits_to_western(normalized)
        transformations.append('convert_digits')
        
        # Normalize date separators
        normalized = re.sub(r'[/\-\.]', '-', normalized)
        transformations.append('normalize_date_separators')
        
    elif field_type in ['arabic_text', 'arabic_short', 'arabic_multiline']:
        # For Arabic text fields, be conservative
        
        # Remove diacritics (usually not semantically important)
        normalized = remove_diacritics(normalized)
        transformations.append('remove_diacritics')
        
        # Remove tatweel
        normalized = remove_tatweel(normalized)
        transformations.append('remove_tatweel')
        
        # Normalize whitespace
        normalized = normalize_whitespace(normalized)
        transformations.append('normalize_whitespace')
        
        # Clean OCR artifacts
        normalized = clean_ocr_artifacts(normalized)
        transformations.append('clean_ocr_artifacts')
        
    else:
        # Default normalization
        normalized = remove_diacritics(normalized)
        normalized = normalize_whitespace(normalized)
        transformations.extend(['remove_diacritics', 'normalize_whitespace'])
    
    # Check if normalization significantly changed the text
    if len(normalized) < len(raw_preserved) * 0.5:
        # Significant reduction - might indicate OCR issues
        confidence_adjustment = -0.1
    
    return NormalizedText(
        raw_text=raw_preserved,
        normalized_text=normalized,
        transformations_applied=transformations,
        confidence_adjustment=confidence_adjustment
    )


def extract_digits_only(text: str) -> str:
    """
    Extract only digits from text.
    
    Useful for National ID extraction.
    
    Args:
        text: Input text
        
    Returns:
        String containing only digits
    """
    # First convert Arabic digits to Western
    converted = convert_arabic_digits_to_western(text)
    # Then extract only digits
    digits = re.sub(r'[^\d]', '', converted)
    return digits


def validate_arabic_text_length(
    normalized_text: str,
    min_length: int,
    max_length: int
) -> Tuple[bool, str]:
    """
    Validate that normalized Arabic text is within expected length range.
    
    Args:
        normalized_text: Normalized text
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    text_length = len(normalized_text.strip())
    
    if text_length < min_length:
        return False, f"Text too short: {text_length} < {min_length}"
    
    if text_length > max_length:
        return False, f"Text too long: {text_length} > {max_length}"
    
    return True, ""
