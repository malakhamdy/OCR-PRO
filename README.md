# Egyptian National ID OCR System

🇪🇬 **Arabic-First Document Understanding Pipeline for Egyptian National IDs**

A production-oriented computer vision and document AI system that understands Egyptian National ID cards as structured Arabic documents.

## Features

### Core Capabilities

- **Card Detection**: Automatically detects Egyptian National ID cards in photographs with varying backgrounds, rotations, and perspectives
- **Perspective Correction**: Rectifies card images to canonical representation using homography transformation
- **Side Classification**: Automatically determines whether uploaded image shows front or back side
- **Dynamic Field Localization**: Scale-invariant field detection using normalized coordinates (not fixed pixels)
- **Field-Specific Preprocessing**: Multiple preprocessing variants per field type optimized for Arabic text
- **Arabic-First OCR**: PaddleOCR with Arabic-capable models as primary OCR engine
- **Multi-Pass OCR**: Generates multiple candidates using different preprocessing approaches
- **NID Validation**: Complete structural validation including century code, date, governorate, and checksum
- **Cross-Field Validation**: Independent verification between NID-derived data and printed fields
- **Uncertainty Exposure**: Clear indication of confidence levels and verification status

### Non-Negotiable Principles

1. ✅ Arabic is the canonical language of the document
2. ✅ No translation of Arabic identity data into English
3. ✅ No dependence on fixed pixel coordinates
4. ✅ No assumption of uniform image dimensions/scale/aspect ratio
5. ✅ Field-specific preprocessing (not one-size-fits-all)
6. ✅ OCR confidence ≠ validation correctness
7. ✅ No hallucination of missing data
8. ✅ Preservation of raw OCR separately from normalized output
9. ✅ Every field has provenance
10. ✅ Critical fields validated independently

## Architecture

```
INPUT IMAGE
    ↓
Image Validation & Quality Assessment
    ↓
Input Normalization (aspect-ratio preserving)
    ↓
Card Detection (contours, edges, quadrilateral)
    ↓
Corner Detection & Perspective Correction
    ↓
Canonical Card Representation (1200×760)
    ↓
Front/Back Side Classification
    ↓
Dynamic Field Localization (normalized coordinates)
    ↓
Field Crop Extraction
    ↓
Field-Specific Preprocessing (multiple variants)
    ↓
Arabic-First OCR (PaddleOCR)
    ↓
OCR Candidate Generation & Ranking
    ↓
Arabic Text Normalization
    ↓
Field Parsing & Validation
    ↓
Cross-Field Validation (NID vs printed DOB/governorate)
    ↓
Independent Verification
    ↓
Final Structured Result with Confidence Scores
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

- `paddlepaddle` - Deep learning framework
- `paddleocr` - Arabic-capable OCR engine
- `opencv-python-headless` - Image processing
- `numpy` - Numerical operations
- `streamlit` - Web UI
- `pyzbar` - Barcode decoding (for back side)
- `Pillow` - Image handling
- `shapely` - Geometric operations

## Usage

### Running the Streamlit Application

```bash
streamlit run app.py --server.headless true --server.port 8501
```

Then open your browser to `http://localhost:8501`

### Programmatic Usage

```python
from pipeline import get_pipeline
import cv2

# Load pipeline
pipeline = get_pipeline()

# Load image (BGR format)
image = cv2.imread('egyptian_id.jpg')

# Process
result = pipeline.process_image(image)

# Access results
if result.success:
    # Document info
    print(f"Side: {result.document_info['side']}")
    print(f"Card detection confidence: {result.document_info['card_detection_confidence']:.2f}")
    
    # Extracted fields
    for field_name, field_obj in result.fields.items():
        print(f"\n{field_name}:")
        print(f"  Raw: {field_obj.raw_value}")
        print(f"  Normalized: {field_obj.normalized_value}")
        print(f"  Verification: {field_obj.verification_status}")
        print(f"  OCR Confidence: {field_obj.ocr_confidence:.2f}")
    
    # Cross-validation
    if result.cross_validation:
        print(f"\nCross-validation status: {result.cross_validation.overall_status}")
        print(f"Matches: {result.cross_validation.matches}")
        print(f"Mismatches: {result.cross_validation.mismatches}")
```

## Module Structure

```
/workspace
├── config.py                    # Centralized configuration
├── image_normalization.py       # Input normalization & coordinate tracking
├── card_detection.py            # Card detection & corner estimation
├── perspective_correction.py    # Homography & rectification
├── side_classification.py       # Front/back side detection
├── field_localization.py        # Dynamic field localization
├── field_preprocessing.py       # Field-specific preprocessing
├── ocr_engine.py                # Arabic OCR with PaddleOCR
├── arabic_normalization.py      # Arabic text normalization
├── nid_validation.py            # National ID structural validation
├── cross_field_validation.py    # Cross-field consistency checks
├── pipeline.py                  # Main processing pipeline
├── app.py                       # Streamlit web application
├── tests.py                     # Comprehensive test suite
└── requirements.txt             # Python dependencies
```

## Supported Fields

### Front Side
- **National ID Number** (numeric, verified via structural validation)
- **Name** (Arabic text)
- **Date of Birth** (cross-validated with NID-derived DOB)
- **Gender** (Arabic short text)
- **Birth Governorate** (cross-validated with NID governorate code)
- **Profession** (Arabic text)
- **Address** (Arabic multi-line text)

### Back Side
- **Barcode/PDF417** (separate barcode decoder)
- **Additional Information** (Arabic multi-line text)

## Verification Status Levels

| Status | Meaning |
|--------|---------|
| `VERIFIED` | Mathematically validated (e.g., NID passes all structural checks) |
| `CROSS_VALIDATED` | Confirmed by independent source (e.g., printed DOB matches NID-derived DOB) |
| `EXTRACTED` | Successfully extracted but not independently verified |
| `LOW_CONFIDENCE` | Extraction uncertain, multiple conflicting candidates |

## NID Validation Checks

The National ID validator performs:

1. **Format Check**: Digits only
2. **Length Check**: Exactly 14 digits
3. **Century Code**: Position 2 (2=19xx, 3=20xx)
4. **Date Validation**: Positions 3-8 (YYMMDD), validates month/day/year
5. **Governorate Code**: Positions 5-6, validates against 34 governorates
6. **Checksum**: Position 13, Luhn-like algorithm

Validation statuses:
- `VALID` - All checks passed
- `INVALID_FORMAT` - Contains non-digit characters
- `INVALID_LENGTH` - Not 14 digits
- `INVALID_DATE` - Invalid birth date components
- `INVALID_GOVERNORATE` - Unknown governorate code
- `INVALID_CHECKSUM` - Checksum mismatch
- `INVALID_STRUCTURE` - Other structural issues

## Coordinate Systems

The system maintains explicit separation of coordinate spaces:

1. **Original Image Coordinates**: Pixel positions in uploaded image
2. **Processing Canvas Coordinates**: After normalization/padding
3. **Canonical Card Coordinates**: After perspective rectification (source of truth for localization)
4. **Field Crop Coordinates**: Individual field regions

All bounding boxes specify their coordinate space:
```json
{
  "bbox": [420, 215, 380, 45],
  "coordinate_space": "canonical_card"
}
```

## Testing

Run the comprehensive test suite:

```bash
python tests.py
```

Tests cover:
- Image normalization (different resolutions, aspect ratios)
- Card detection (synthetic images)
- Perspective correction (mock detections)
- Side classification
- Field localization (front and back)
- Arabic normalization (digit conversion, text cleaning)
- NID validation (valid, invalid length, invalid format)
- Cross-field validation
- Pipeline integration

## Privacy & Security

- **NID Masking**: Option to mask National ID numbers in UI (`7704********77`)
- **No Persistent Storage**: Images processed in-memory, not stored
- **Debug Logging**: Sensitive identifiers masked in logs
- **Local Processing**: All computation runs locally, no cloud APIs

## Limitations & Considerations

### OCR Engine
- PaddleOCR Arabic model requires proper installation
- GPU support available but not required (CPU fallback works)
- Model loading is cached to avoid reloading on each request

### Image Quality
- Minimum recommended resolution: 640×480
- Card should occupy at least 15% of image area
- Extreme blur, glare, or shadows may reduce accuracy

### Arabic Text
- Diacritics (tashkeel) are removed during normalization
- Tatweel (elongation) is normalized
- Character variations (ة/ه, ى/ي) handled conservatively

## Troubleshooting

### OCR Engine Not Loading
```
ERROR: Failed to load PaddleOCR: Unknown argument: use_gpu
```
Solution: The OCR engine gracefully handles this and continues with CPU mode.

### Card Not Detected
- Ensure card occupies sufficient portion of image (>15%)
- Improve lighting and reduce shadows
- Try uploading higher resolution image

### Low OCR Confidence
- Check field crop quality in debug visualization
- Verify card was properly rectified
- Consider re-taking photo with better lighting

## License

This project is provided as-is for educational and research purposes.

## Contributing

When contributing:
1. Preserve Arabic as canonical language
2. Maintain coordinate system separation
3. Add tests for new functionality
4. Document uncertainty and confidence levels
5. Never silently correct uncertain identity information

---

**Built following production-oriented document AI principles for Arabic document understanding.**
