"""
Cross-Field Validation Module

Performs cross-validation between different extracted fields.
Checks consistency between NID-derived data and printed field data.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import VALIDATION_CONFIG


@dataclass
class CrossValidationCheck:
    """Single cross-validation check result."""
    check_name: str
    field_a: str
    field_b: str
    value_a: Any
    value_b: Any
    passed: bool
    confidence: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_name': self.check_name,
            'field_a': self.field_a,
            'field_b': self.field_b,
            'value_a': self.value_a,
            'value_b': self.value_b,
            'passed': self.passed,
            'confidence': self.confidence,
            'message': self.message
        }


@dataclass
class CrossFieldValidationResult:
    """Complete cross-field validation result."""
    checks: List[CrossValidationCheck]
    matches: int
    mismatches: int
    warnings: int
    overall_status: str  # CROSS_VALIDATED, PARTIAL_MATCH, MISMATCH_DETECTED
    critical_issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checks': [c.to_dict() for c in self.checks],
            'matches': self.matches,
            'mismatches': self.mismatches,
            'warnings': self.warnings,
            'overall_status': self.overall_status,
            'critical_issues': self.critical_issues
        }


def compare_dob_from_nid_vs_printed(
    nid_derived_dob: Optional[Dict[str, Any]],
    printed_dob_raw: Optional[str],
    printed_dob_normalized: Optional[str]
) -> CrossValidationCheck:
    """
    Compare date of birth from NID structure vs printed DOB field.
    
    Args:
        nid_derived_dob: DOB extracted from NID structure
        printed_dob_raw: Raw OCR from printed DOB field
        printed_dob_normalized: Normalized printed DOB
        
    Returns:
        CrossValidationCheck result
    """
    check_name = "dob_cross_validation"
    
    # Handle missing data
    if nid_derived_dob is None:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="nid_derived_dob",
            field_b="printed_dob",
            value_a=None,
            value_b=printed_dob_raw,
            passed=False,
            confidence=0.0,
            message="NID-derived DOB not available"
        )
    
    if not printed_dob_normalized:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="nid_derived_dob",
            field_b="printed_dob",
            value_a=nid_derived_dob.get('date_string'),
            value_b=printed_dob_raw,
            passed=False,
            confidence=0.0,
            message="Printed DOB not available or could not be normalized"
        )
    
    # Extract NID DOB string
    nid_dob_string = nid_derived_dob.get('date_string', '')
    
    # Normalize printed DOB for comparison (YYYY-MM-DD format expected)
    printed_dob_clean = printed_dob_normalized.strip()
    
    # Direct comparison
    if nid_dob_string == printed_dob_clean:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="nid_derived_dob",
            field_b="printed_dob",
            value_a=nid_dob_string,
            value_b=printed_dob_clean,
            passed=True,
            confidence=1.0,
            message="DOB values match exactly"
        )
    
    # Try fuzzy matching (handle different formats)
    # E.g., "1990-01-15" vs "15/01/1990" vs "1990/1/15"
    import re
    
    # Extract year, month, day from both strings
    def extract_date_components(date_str):
        digits = re.findall(r'\d+', date_str)
        if len(digits) >= 3:
            # Try to identify year, month, day
            candidates = []
            for d in digits:
                if len(d) == 4:
                    candidates.append(('y', int(d)))
                elif len(d) <= 2:
                    candidates.append(('d', int(d)))
            # This is simplified - real implementation would be more robust
            return digits[:3]
        return []
    
    nid_parts = extract_date_components(nid_dob_string)
    printed_parts = extract_date_components(printed_dob_clean)
    
    if nid_parts and printed_parts:
        # Check if key components match
        if set(nid_parts) == set(printed_parts):
            return CrossValidationCheck(
                check_name=check_name,
                field_a="nid_derived_dob",
                field_b="printed_dob",
                value_a=nid_dob_string,
                value_b=printed_dob_clean,
                passed=True,
                confidence=0.8,
                message="DOB values match with different formatting"
            )
    
    # Mismatch detected
    return CrossValidationCheck(
        check_name=check_name,
        field_a="nid_derived_dob",
        field_b="printed_dob",
        value_a=nid_dob_string,
        value_b=printed_dob_clean,
        passed=False,
        confidence=1.0,
        message=f"DOB mismatch: NID says {nid_dob_string}, printed says {printed_dob_clean}"
    )


def compare_governorate_from_nid_vs_printed(
    nid_governorate_code: Optional[str],
    nid_governorate_name: Optional[str],
    printed_governorate_raw: Optional[str]
) -> CrossValidationCheck:
    """
    Compare governorate from NID structure vs printed governorate field.
    
    Args:
        nid_governorate_code: Governorate code from NID
        nid_governorate_name: Governorate name from NID validation
        printed_governorate_raw: Raw OCR from printed governorate field
        
    Returns:
        CrossValidationCheck result
    """
    check_name = "governorate_cross_validation"
    
    if nid_governorate_code is None:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="nid_governorate",
            field_b="printed_governorate",
            value_a=None,
            value_b=printed_governorate_raw,
            passed=False,
            confidence=0.0,
            message="NID governorate code not available"
        )
    
    if not printed_governorate_raw:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="nid_governorate",
            field_b="printed_governorate",
            value_a=nid_governorate_name or nid_governorate_code,
            value_b=None,
            passed=False,
            confidence=0.0,
            message="Printed governorate not available"
        )
    
    # Check if printed governorate matches NID governorate name
    # This requires Arabic text matching
    if nid_governorate_name:
        # Simple substring match (Arabic)
        if nid_governorate_name in printed_governorate_raw or printed_governorate_raw in nid_governorate_name:
            return CrossValidationCheck(
                check_name=check_name,
                field_a="nid_governorate",
                field_b="printed_governorate",
                value_a=nid_governorate_name,
                value_b=printed_governorate_raw,
                passed=True,
                confidence=0.9,
                message="Governorate names match"
            )
    
    # If no clear match, report as uncertain (not necessarily wrong due to OCR issues)
    return CrossValidationCheck(
        check_name=check_name,
        field_a="nid_governorate",
        field_b="printed_governorate",
        value_a=nid_governorate_name or nid_governorate_code,
        value_b=printed_governorate_raw,
        passed=False,
        confidence=0.5,
        message="Governorate comparison inconclusive - possible OCR variance"
    )


def validate_gender_consistency(
    printed_gender: Optional[str],
    nid_derived_gender: Optional[str]
) -> CrossValidationCheck:
    """
    Validate consistency between printed gender and NID-derived gender.
    
    Note: Egyptian NID may encode gender in certain positions.
    This check compares printed gender field with any derived signal.
    
    Args:
        printed_gender: Gender from printed field
        nid_derived_gender: Gender derived from NID structure (if applicable)
        
    Returns:
        CrossValidationCheck result
    """
    check_name = "gender_consistency"
    
    if not printed_gender:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="printed_gender",
            field_b="nid_derived_gender",
            value_a=None,
            value_b=nid_derived_gender,
            passed=False,
            confidence=0.0,
            message="Printed gender not available"
        )
    
    if not nid_derived_gender:
        # NID-derived gender not available - can't cross-validate
        return CrossValidationCheck(
            check_name=check_name,
            field_a="printed_gender",
            field_b="nid_derived_gender",
            value_a=printed_gender,
            value_b=None,
            passed=False,
            confidence=0.0,
            message="NID-derived gender not available for cross-validation"
        )
    
    # Normalize gender values for comparison
    gender_map = {
        'ذكر': 'M', 'male': 'M', 'm': 'M', 'Male': 'M',
        'أنثى': 'F', 'female': 'F', 'f': 'F', 'Female': 'F'
    }
    
    printed_norm = gender_map.get(printed_gender.strip().lower(), printed_gender)
    derived_norm = gender_map.get(nid_derived_gender.strip().lower(), nid_derived_gender)
    
    if printed_norm == derived_norm:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="printed_gender",
            field_b="nid_derived_gender",
            value_a=printed_gender,
            value_b=nid_derived_gender,
            passed=True,
            confidence=1.0,
            message="Gender values consistent"
        )
    else:
        return CrossValidationCheck(
            check_name=check_name,
            field_a="printed_gender",
            field_b="nid_derived_gender",
            value_a=printed_gender,
            value_b=nid_derived_gender,
            passed=False,
            confidence=1.0,
            message=f"Gender mismatch: printed={printed_gender}, derived={nid_derived_gender}"
        )


def perform_cross_field_validation(
    extracted_fields: Dict[str, Any],
    nid_validation_result: Optional[Any]
) -> CrossFieldValidationResult:
    """
    Perform complete cross-field validation.
    
    Args:
        extracted_fields: Dictionary of all extracted fields
        nid_validation_result: Result from NID validation
        
    Returns:
        CrossFieldValidationResult with all checks
    """
    checks = []
    critical_issues = []
    
    # Get NID-derived information
    nid_derived_dob = None
    nid_gov_code = None
    nid_gov_name = None
    
    if nid_validation_result and nid_validation_result.is_valid:
        nid_derived_dob = nid_validation_result.derived_info.get('birth_date')
        nid_gov_code = nid_validation_result.derived_info.get('governorate_code')
        nid_gov_name = nid_validation_result.derived_info.get('governorate_name')
    
    # Get printed field values
    printed_dob_raw = extracted_fields.get('date_of_birth', {}).get('raw')
    printed_dob_normalized = extracted_fields.get('date_of_birth', {}).get('normalized')
    printed_gov_raw = extracted_fields.get('birth_governorate', {}).get('raw')
    printed_gender = extracted_fields.get('gender', {}).get('raw')
    
    # Run DOB cross-validation
    dob_check = compare_dob_from_nid_vs_printed(
        nid_derived_dob,
        printed_dob_raw,
        printed_dob_normalized
    )
    checks.append(dob_check)
    
    # Run governorate cross-validation
    gov_check = compare_governorate_from_nid_vs_printed(
        nid_gov_code,
        nid_gov_name,
        printed_gov_raw
    )
    checks.append(gov_check)
    
    # Count results
    matches = sum(1 for c in checks if c.passed)
    mismatches = sum(1 for c in checks if not c.passed and c.confidence > 0.7)
    warnings = sum(1 for c in checks if not c.passed and c.confidence <= 0.7)
    
    # Identify critical issues
    for check in checks:
        if not check.passed and check.confidence > 0.8:
            critical_issues.append(check.message)
    
    # Determine overall status
    if mismatches > 0:
        overall_status = "MISMATCH_DETECTED"
    elif warnings > 0:
        overall_status = "PARTIAL_MATCH"
    elif matches > 0:
        overall_status = "CROSS_VALIDATED"
    else:
        overall_status = "NO_CROSS_VALIDATION_AVAILABLE"
    
    return CrossFieldValidationResult(
        checks=checks,
        matches=matches,
        mismatches=mismatches,
        warnings=warnings,
        overall_status=overall_status,
        critical_issues=critical_issues
    )
