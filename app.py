"""
Egyptian National ID OCR System - Streamlit Application

Production-ready UI for Egyptian National ID document understanding.
Visualizes all processing steps and exposes uncertainty.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional

# Import pipeline modules
from config import IMAGE_CONFIG
from pipeline import get_pipeline, PipelineResult
from nid_validation import mask_nid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Egyptian National ID OCR",
    page_icon="🇪🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better visualization
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .status-verified {
        color: #28a745;
        font-weight: bold;
    }
    .status-extracted {
        color: #ffc107;
        font-weight: bold;
    }
    .status-low-confidence {
        color: #dc3545;
        font-weight: bold;
    }
    .field-table {
        width: 100%;
        border-collapse: collapse;
    }
    .field-table th, .field-table td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load pipeline with caching to avoid reloading on each rerun."""
    return get_pipeline()


def convert_to_opencv_format(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR format."""
    rgb_image = np.array(pil_image)
    if len(rgb_image.shape) == 2:
        # Grayscale to BGR
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        # RGB to BGR
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def resize_for_display(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image for display while preserving aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🇪🇬 Egyptian National ID OCR System</p>', unsafe_allow_html=True)
    st.markdown("""
    **Arabic-First Document Understanding Pipeline**
    
    This system processes Egyptian National ID cards through a complete pipeline:
    card detection → perspective correction → field localization → Arabic OCR → validation → cross-field verification
    """)
    
    # Sidebar for settings and info
    with st.sidebar:
        st.header("Settings")
        
        show_debug = st.checkbox("Show Debug Info", value=False)
        display_masked_nid = st.checkbox("Mask NID for Privacy", value=True)
        
        st.divider()
        
        st.subheader("Pipeline Status")
        try:
            pipeline = load_pipeline()
            st.success("✅ Pipeline loaded successfully")
            
            # Check OCR engine availability
            from ocr_engine import PaddleOcrEngine
            ocr_engine = PaddleOcrEngine()
            if ocr_engine.is_available():
                st.success("✅ Arabic OCR engine ready")
            else:
                st.warning("⚠️ OCR engine not fully loaded")
        except Exception as e:
            st.error(f"❌ Pipeline error: {str(e)}")
        
        st.divider()
        
        st.markdown("""
        ### Supported Fields
        
        **Front Side:**
        - National ID Number
        - Name (Arabic)
        - Date of Birth
        - Gender
        - Birth Governorate
        - Profession
        - Address
        
        **Back Side:**
        - Barcode/PDF417
        - Additional Information
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Egyptian National ID Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supports front or back side images"
        )
    
    with col2:
        st.markdown("""
        ### Upload Guidelines
        
        - Ensure good lighting
        - Minimize shadows and glare
        - Keep card within frame
        - Avoid extreme angles
        - Recommended resolution: 1280×720 or higher
        """)
    
    # Process uploaded image
    if uploaded_file is not None:
        try:
            # Load and convert image
            pil_image = Image.open(uploaded_file)
            opencv_image = convert_to_opencv_format(pil_image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(
                cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB),
                caption=f"Uploaded: {pil_image.size[0]}×{pil_image.size[1]} pixels",
                use_container_width=True
            )
            
            # Run pipeline
            with st.spinner("Processing document..."):
                pipeline = load_pipeline()
                result = pipeline.process_image(opencv_image)
            
            # Display results
            if result.success:
                st.success("✅ Document processed successfully!")
                
                # ========== Document Info ==========
                st.markdown('<p class="section-header">Document Information</p>', unsafe_allow_html=True)
                
                doc_info = result.document_info
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric(
                        "Detected Side",
                        doc_info.get('side', 'Unknown').upper(),
                        f"Confidence: {doc_info.get('side_confidence', 0):.0%}"
                    )
                
                with info_col2:
                    st.metric(
                        "Card Detection",
                        "Detected" if doc_info.get('card_detection_confidence', 0) > 0.7 else "Uncertain",
                        f"Confidence: {doc_info.get('card_detection_confidence', 0):.0%}"
                    )
                
                with info_col3:
                    quality = doc_info.get('image_quality', {})
                    st.metric(
                        "Card Area Ratio",
                        f"{quality.get('card_area_ratio', 0):.1%}",
                        "Good" if quality.get('card_area_ratio', 0) > 0.15 else "Low"
                    )
                
                # ========== Extracted Fields ==========
                st.markdown('<p class="section-header">Extracted Fields</p>', unsafe_allow_html=True)
                
                if result.fields:
                    # Create fields table
                    field_data = []
                    for field_name, field_obj in result.fields.items():
                        display_value = field_obj.normalized_value
                        
                        # Mask NID if requested
                        if display_masked_nid and field_name == 'national_id' and display_value:
                            display_value = mask_nid(display_value)
                        
                        # Determine status indicator
                        if field_obj.verification_status == "VERIFIED":
                            status_indicator = "✅ VERIFIED"
                        elif field_obj.verification_status == "CROSS_VALIDATED":
                            status_indicator = "🔄 CROSS-VALIDATED"
                        elif field_obj.verification_status == "LOW_CONFIDENCE":
                            status_indicator = "⚠️ LOW CONFIDENCE"
                        else:
                            status_indicator = "📝 EXTRACTED"
                        
                        field_data.append({
                            "Field": field_name.replace('_', ' ').title(),
                            "Value": display_value or "-",
                            "OCR Confidence": f"{field_obj.ocr_confidence:.0%}" if field_obj.ocr_confidence else "-",
                            "Status": status_indicator
                        })
                    
                    st.table(field_data)
                    
                    # Detailed field view
                    with st.expander("View Field Details"):
                        for field_name, field_obj in result.fields.items():
                            st.markdown(f"**{field_name.replace('_', ' ').title()}**")
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.write("**Raw Value:**")
                                st.code(field_obj.raw_value or "Not detected")
                            
                            with detail_col2:
                                st.write("**Normalized Value:**")
                                st.code(field_obj.normalized_value or "Not detected")
                            
                            st.write(f"**Verification Status:** `{field_obj.verification_status}`")
                            st.write(f"**OCR Confidence:** `{field_obj.ocr_confidence:.2f}`")
                            st.write(f"**Localization Confidence:** `{field_obj.localization_confidence:.2f}`")
                            st.divider()
                
                # ========== Derived Information ==========
                if result.derived_info:
                    st.markdown('<p class="section-header">Derived Information</p>', unsafe_allow_html=True)
                    
                    derived_col1, derived_col2 = st.columns(2)
                    
                    with derived_col1:
                        if 'nid_birth_date' in result.derived_info:
                            st.info(f"**NID-Derived DOB:** {result.derived_info['nid_birth_date']}")
                    
                    with derived_col2:
                        if 'nid_governorate' in result.derived_info:
                            st.info(f"**Birth Governorate:** {result.derived_info['nid_governorate']}")
                
                # ========== Cross-Validation Results ==========
                if result.cross_validation:
                    st.markdown('<p class="section-header">Cross-Field Validation</p>', unsafe_allow_html=True)
                    
                    cv_status = result.cross_validation.overall_status
                    
                    if cv_status == "CROSS_VALIDATED":
                        st.success(f"✅ {cv_status}")
                    elif cv_status == "PARTIAL_MATCH":
                        st.warning(f"⚠️ {cv_status}")
                    elif cv_status == "MISMATCH_DETECTED":
                        st.error(f"❌ {cv_status}")
                    else:
                        st.info(f"ℹ️ {cv_status}")
                    
                    st.write(f"**Matches:** {result.cross_validation.matches} | **Mismatches:** {result.cross_validation.mismatches} | **Warnings:** {result.cross_validation.warnings}")
                    
                    if result.cross_validation.critical_issues:
                        st.warning("**Critical Issues:**")
                        for issue in result.cross_validation.critical_issues:
                            st.write(f"- {issue}")
                
                # ========== Debug Visualization ==========
                if show_debug and result.debug_info:
                    st.markdown('<p class="section-header">Debug Visualization</p>', unsafe_allow_html=True)
                    
                    debug_col1, debug_col2 = st.columns(2)
                    
                    with debug_col1:
                        if 'card_detection' in pipeline.debug_images:
                            debug_img = pipeline.debug_images['card_detection']
                            st.image(
                                cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                                caption="Card Detection",
                                use_container_width=True
                            )
                    
                    with debug_col2:
                        if 'field_localization' in pipeline.debug_images:
                            debug_img = pipeline.debug_images['field_localization']
                            st.image(
                                cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                                caption="Field Localization",
                                use_container_width=True
                            )
                    
                    if 'canonical_card' in pipeline.debug_images:
                        canonical = pipeline.debug_images['canonical_card']
                        st.image(
                            cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB),
                            caption="Rectified Canonical Card",
                            use_container_width=True
                        )
                    
                    # Show detailed debug info
                    with st.expander("Raw Debug Data"):
                        st.json(result.debug_info)
                
            else:
                # Pipeline failed
                st.error(f"❌ Processing failed: {result.error_message}")
                
                if result.debug_info and show_debug:
                    with st.expander("Debug Information"):
                        st.json(result.debug_info)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            logger.exception("Processing error")
    
    else:
        # No file uploaded - show placeholder
        st.info("👆 Please upload an Egyptian National ID image to begin processing")
        
        # Show example output structure
        with st.expander("See Example Output Structure"):
            st.markdown("""
            ```json
            {
              "document_info": {
                "side": "front",
                "side_confidence": 0.95,
                "card_detection_confidence": 0.92
              },
              "fields": {
                "national_id": {
                  "raw_value": "29001011234567",
                  "normalized_value": "29001011234567",
                  "verification_status": "VERIFIED",
                  "ocr_confidence": 0.94
                },
                "name": {
                  "raw_value": "محمد أحمد",
                  "normalized_value": "محمد احمد",
                  "verification_status": "EXTRACTED",
                  "ocr_confidence": 0.89
                },
                "date_of_birth": {
                  "raw_value": "1990-01-01",
                  "normalized_value": "1990-01-01",
                  "verification_status": "CROSS_VALIDATED",
                  "ocr_confidence": 0.91
                }
              }
            }
            ```
            """)


if __name__ == "__main__":
    main()
