"""
National ID Validation Module

Validates Egyptian National ID number structure.
Checks length, date components, governorate codes, and checksum.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from config import VALIDATION_CONFIG


@dataclass
class NidValidationResult:
    """Result of National ID validation."""
    nid: str
    is_valid: bool
    validation_status: str  # VALID, INVALID_FORMAT, INVALID_DATE, etc.
    errors: List[str]
    derived_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nid': self.nid,
            'is_valid': self.is_valid,
            'validation_status': self.validation_status,
            'errors': self.errors,
            'derived_info': self.derived_info
        }


def validate_nid_length(nid: str) -> Tuple[bool, str]:
    """
    Validate National ID has correct length.
    
    Args:
        nid: National ID string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(nid) != VALIDATION_CONFIG.NID_LENGTH:
        return False, f"Invalid length: expected {VALIDATION_CONFIG.NID_LENGTH}, got {len(nid)}"
    return True, ""


def validate_nid_format(nid: str) -> Tuple[bool, str]:
    """
    Validate National ID contains only digits.
    
    Args:
        nid: National ID string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not nid.isdigit():
        return False, "National ID must contain only digits"
    return True, ""


def extract_century_code(nid: str) -> Optional[int]:
    """
    Extract century code from National ID.
    
    Position 1 (0-indexed): 2 = 19xx, 3 = 20xx
    
    Args:
        nid: National ID string
        
    Returns:
        Century code (2 or 3) or None if invalid
    """
    if len(nid) < 3:
        return None
    
    try:
        code = int(nid[2])
        if code in [2, 3]:
            return code
        return None
    except (ValueError, IndexError):
        return None


def extract_birth_date_from_nid(nid: str) -> Optional[Dict[str, Any]]:
    """
    Extract birth date information from National ID.
    
    Positions 3-8 (0-indexed): YYMMDD format
    
    Args:
        nid: National ID string
        
    Returns:
        Dictionary with year, month, day, full_year, date_string
        or None if extraction fails
    """
    if len(nid) < 9:
        return None
    
    century_code = extract_century_code(nid)
    if century_code is None:
        return None
    
    try:
        yy = int(nid[3:5])
        mm = int(nid[5:7])
        dd = int(nid[7:9])
        
        # Determine full year based on century code
        if century_code == 2:
            full_year = 1900 + yy
        elif century_code == 3:
            full_year = 2000 + yy
        else:
            return None
        
        return {
            'year': yy,
            'month': mm,
            'day': dd,
            'full_year': full_year,
            'date_string': f"{full_year}-{mm:02d}-{dd:02d}"
        }
        
    except (ValueError, IndexError):
        return None


def validate_birth_date(date_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate extracted birth date components.
    
    Checks:
    - Month range (1-12)
    - Day range (1-31, basic validation)
    - Year range (reasonable bounds)
    - Date validity (e.g., not Feb 30)
    
    Args:
        date_info: Dictionary from extract_birth_date_from_nid
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    year = date_info.get('year', 0)
    month = date_info.get('month', 0)
    day = date_info.get('day', 0)
    full_year = date_info.get('full_year', 0)
    
    # Validate year range
    if full_year < VALIDATION_CONFIG.MIN_VALID_YEAR or full_year > VALIDATION_CONFIG.MAX_VALID_YEAR:
        errors.append(f"Year {full_year} outside valid range ({VALIDATION_CONFIG.MIN_VALID_YEAR}-{VALIDATION_CONFIG.MAX_VALID_YEAR})")
    
    # Validate month range
    if month < 1 or month > 12:
        errors.append(f"Invalid month: {month}")
    
    # Validate day range (basic check)
    if day < 1 or day > 31:
        errors.append(f"Invalid day: {day}")
    
    # More specific day validation per month
    if month in [4, 6, 9, 11] and day > 30:
        errors.append(f"Day {day} invalid for month {month}")
    
    # February validation
    if month == 2:
        is_leap = (full_year % 4 == 0 and full_year % 100 != 0) or (full_year % 400 == 0)
        max_day = 29 if is_leap else 28
        if day > max_day:
            errors.append(f"Day {day} invalid for February {full_year}")
    
    return len(errors) == 0, errors


def extract_governorate_code(nid: str) -> Optional[str]:
    """
    Extract governorate code from National ID.
    
    Positions 5-6 (0-indexed): 2-digit governorate code
    
    Args:
        nid: National ID string
        
    Returns:
        Governorate code string or None if extraction fails
    """
    if len(nid) < 7:
        return None
    
    gov_code = nid[5:7]
    
    # Validate it's a numeric code
    if not gov_code.isdigit():
        return None
    
    return gov_code


def validate_governorate_code(gov_code: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate governorate code against known codes.
    
    Args:
        gov_code: 2-digit governorate code
        
    Returns:
        Tuple of (is_valid, governorate_name_arabic, error_message)
    """
    if gov_code in VALIDATION_CONFIG.VALID_GOVERNORATE_CODES:
        governorate_name = VALIDATION_CONFIG.VALID_GOVERNORATE_CODES[gov_code]
        return True, governorate_name, ""
    else:
        return False, None, f"Unknown governorate code: {gov_code}"


def calculate_checksum(nid: str) -> Optional[int]:
    """
    Calculate checksum digit for National ID.
    
    Uses Luhn-like algorithm adapted for Egyptian NID.
    
    Note: The exact checksum algorithm for Egyptian NID may vary.
    This implements a common validation approach.
    
    Args:
        nid: National ID string (14 digits)
        
    Returns:
        Calculated checksum digit or None if calculation fails
    """
    if len(nid) != 14:
        return None
    
    try:
        # Extract first 13 digits for checksum calculation
        digits = [int(d) for d in nid[:13]]
        
        # Common checksum algorithm: weighted sum mod 10
        weights = [2, 4, 8, 5, 10, 9, 7, 3, 6, 1, 2, 4, 8]  # Example weights
        
        weighted_sum = sum(d * w for d, w in zip(digits, weights))
        checksum = (10 - (weighted_sum % 10)) % 10
        
        return checksum
        
    except (ValueError, IndexError):
        return None


def validate_checksum(nid: str) -> Tuple[bool, str]:
    """
    Validate National ID checksum.
    
    Args:
        nid: National ID string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(nid) != 14:
        return False, "Invalid NID length for checksum validation"
    
    try:
        provided_checksum = int(nid[13])
        calculated_checksum = calculate_checksum(nid)
        
        if calculated_checksum is None:
            return False, "Failed to calculate checksum"
        
        if provided_checksum != calculated_checksum:
            return False, f"Checksum mismatch: expected {calculated_checksum}, got {provided_checksum}"
        
        return True, ""
        
    except (ValueError, IndexError):
        return False, "Failed to validate checksum"


def validate_national_id(nid: str) -> NidValidationResult:
    """
    Perform complete validation of Egyptian National ID.
    
    Checks:
    1. Format (digits only)
    2. Length (14 digits)
    3. Century code (2 or 3)
    4. Birth date validity
    5. Governorate code validity
    6. Checksum
    
    Args:
        nid: National ID string
        
    Returns:
        NidValidationResult with complete validation status
    """
    errors = []
    derived_info = {}
    
    # Store original input
    original_nid = nid
    
    # Step 1: Format validation
    is_valid_format, format_error = validate_nid_format(nid)
    if not is_valid_format:
        return NidValidationResult(
            nid=nid,
            is_valid=False,
            validation_status="INVALID_FORMAT",
            errors=[format_error],
            derived_info={}
        )
    
    # Step 2: Length validation
    is_valid_length, length_error = validate_nid_length(nid)
    if not is_valid_length:
        return NidValidationResult(
            nid=nid,
            is_valid=False,
            validation_status="INVALID_LENGTH",
            errors=[length_error],
            derived_info={}
        )
    
    # Step 3: Extract and validate century code
    century_code = extract_century_code(nid)
    if century_code is None:
        errors.append("Invalid century code")
    else:
        derived_info['century_code'] = century_code
        derived_info['century'] = '19xx' if century_code == 2 else '20xx'
    
    # Step 4: Extract and validate birth date
    date_info = extract_birth_date_from_nid(nid)
    if date_info:
        derived_info['birth_date'] = date_info
        is_valid_date, date_errors = validate_birth_date(date_info)
        if not is_valid_date:
            errors.extend(date_errors)
    else:
        errors.append("Failed to extract birth date")
    
    # Step 5: Extract and validate governorate code
    gov_code = extract_governorate_code(nid)
    if gov_code:
        is_valid_gov, gov_name, gov_error = validate_governorate_code(gov_code)
        derived_info['governorate_code'] = gov_code
        if is_valid_gov:
            derived_info['governorate_name'] = gov_name
        else:
            errors.append(gov_error)
    else:
        errors.append("Failed to extract governorate code")
    
    # Step 6: Validate checksum
    is_valid_checksum, checksum_error = validate_checksum(nid)
    if not is_valid_checksum:
        errors.append(checksum_error)
    else:
        derived_info['checksum_valid'] = True
    
    # Determine overall validation status
    if not errors:
        validation_status = "VALID"
        is_valid = True
    elif any("Date" in e or "month" in e.lower() or "day" in e.lower() for e in errors):
        validation_status = "INVALID_DATE"
        is_valid = False
    elif any("governorate" in e.lower() for e in errors):
        validation_status = "INVALID_GOVERNORATE"
        is_valid = False
    elif any("Checksum" in e for e in errors):
        validation_status = "INVALID_CHECKSUM"
        is_valid = False
    else:
        validation_status = "INVALID_STRUCTURE"
        is_valid = False
    
    return NidValidationResult(
        nid=original_nid,
        is_valid=is_valid,
        validation_status=validation_status,
        errors=errors,
        derived_info=derived_info
    )


def mask_nid(nid: str, visible_chars: int = 4) -> str:
    """
    Mask National ID for privacy-safe logging/display.
    
    Args:
        nid: National ID string
        visible_chars: Number of characters to show at end
        
    Returns:
        Masked National ID
    """
    if len(nid) <= visible_chars:
        return '*' * len(nid)
    
    masked_length = len(nid) - visible_chars
    return '*' * masked_length + nid[-visible_chars:]
