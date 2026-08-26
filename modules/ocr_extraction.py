"""
Step 2 — OCR Extraction

Reads a document image and extracts structured fields such as:

- Name
- Date of Birth
- Document Number
- Nationality
- Issue Date
- Expiry Date
- MRZ information

Uses a lightweight OCR pipeline based on:

OpenCV preprocessing
        +
Tesseract OCR

The preprocessing improves OCR accuracy while keeping the system
fast and lightweight enough for deployment.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from utils.mrz_utils import parse_mrz, MRZResult

logger = logging.getLogger(__name__)


# ============================================================
# TESSERACT AVAILABILITY
# ============================================================

_TESSERACT_AVAILABLE = None


def _tesseract_available() -> bool:
    """
    Check whether Tesseract is installed and available.
    """

    global _TESSERACT_AVAILABLE

    if _TESSERACT_AVAILABLE is None:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()

            _TESSERACT_AVAILABLE = True

            logger.info("Tesseract is available.")

        except Exception as e:
            logger.warning("Tesseract unavailable: %s", e)

            _TESSERACT_AVAILABLE = False

    return _TESSERACT_AVAILABLE


# ============================================================
# OCR RESULT
# ============================================================

@dataclass
class OCRResult:

    engine_used: str

    raw_lines: list = field(default_factory=list)

    full_text: str = ""

    name: Optional[str] = None

    date_of_birth: Optional[str] = None

    document_number: Optional[str] = None

    nationality: Optional[str] = None

    issue_date: Optional[str] = None

    expiry_date: Optional[str] = None

    mrz: Optional[MRZResult] = None

    error: Optional[str] = None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _preprocess_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Create multiple lightweight image variants for OCR.

    Different document images respond better to different
    preprocessing techniques, so we generate a few versions
    and let OCR choose the best result.
    """

    if image is None or image.size == 0:
        return []

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    if image.ndim == 3:

        if image.shape[2] == 4:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2GRAY
            )
        else:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

    else:
        gray = image.copy()

    # --------------------------------------------------------
    # Upscale small images
    # --------------------------------------------------------

    height, width = gray.shape[:2]

    # Documents with small text benefit from upscaling
    if max(height, width) < 1800:

        scale = 2

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # --------------------------------------------------------
    # CLAHE contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # Light denoising
    # --------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # --------------------------------------------------------
    # Sharpening
    # --------------------------------------------------------

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        denoised,
        -1,
        kernel
    )

    # --------------------------------------------------------
    # OTSU threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        gray,
        enhanced,
        sharpened,
        otsu,
        adaptive
    ]


# ============================================================
# OCR SCORING
# ============================================================

def _score_ocr_text(text: str) -> float:
    """
    Give a simple quality score to OCR output.

    Higher score means the text is more likely to contain
    meaningful document information.
    """

    if not text:
        return 0.0

    clean = text.strip()

    if not clean:
        return 0.0

    # Count useful characters
    alphanumeric = sum(
        char.isalnum()
        for char in clean
    )

    # Count alphabetic words
    words = re.findall(
        r"[A-Za-z]{2,}",
        clean
    )

    # Count numbers
    digits = sum(
        char.isdigit()
        for char in clean
    )

    score = (
        len(clean) * 0.2
        + alphanumeric * 0.5
        + len(words) * 3
        + digits * 0.5
    )

    # Bonus for document-related keywords
    keywords = [
        "passport",
        "name",
        "surname",
        "nationality",
        "birth",
        "date",
        "document",
        "identity",
        "india",
        "government",
        "valid",
        "expiry"
    ]

    lower_text = clean.lower()

    for keyword in keywords:

        if keyword in lower_text:
            score += 10

    return score


# ============================================================
# TESSERACT OCR
# ============================================================

def _run_tesseract(image: np.ndarray) -> list[str]:
    """
    Run Tesseract on multiple preprocessing variants.

    The best OCR result is selected automatically.
    """

    import pytesseract
    from PIL import Image

    processed_images = _preprocess_image(image)

    if not processed_images:
        return []

    best_text = ""
    best_score = 0.0

    # Try different page segmentation modes
    psm_modes = [
        3,   # Automatic page segmentation
        6,   # Uniform block of text
        11   # Sparse text
    ]

    for image_index, processed in enumerate(processed_images):

        pil_img = Image.fromarray(processed)

        for psm in psm_modes:

            config = (
                f"--oem 3 "
                f"--psm {psm}"
            )

            try:

                text = pytesseract.image_to_string(
                    pil_img,
                    config=config
                )

                score = _score_ocr_text(text)

                logger.info(
                    "OCR variant %d | PSM %d | Score %.2f",
                    image_index,
                    psm,
                    score
                )

                if score > best_score:

                    best_score = score

                    best_text = text

            except Exception as e:

                logger.warning(
                    "Tesseract failed on variant %d PSM %d: %s",
                    image_index,
                    psm,
                    e
                )

    lines = [
        line.strip()
        for line in best_text.splitlines()
        if line.strip()
    ]

    logger.info(
        "Best OCR result detected %d lines with score %.2f.",
        len(lines),
        best_score
    )

    return lines


# ============================================================
# FIELD EXTRACTION
# ============================================================

_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}"
    r"|"
    r"\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2}"
    r")\b"
)


_DOC_NUM_RE = re.compile(
    r"\b("
    r"[A-Z]{1,2}\d{6,9}"
    r"|"
    r"\d{9,12}"
    r")\b"
)


_NAME_KEYWORDS = (
    "name",
    "surname",
    "given name",
    "given names"
)


_DOB_KEYWORDS = (
    "birth",
    "dob",
    "date of birth"
)


_NATIONALITY_KEYWORDS = (
    "nationality",
    "nation"
)


_DOCNUM_KEYWORDS = (
    "passport no",
    "document no",
    "id no",
    "number",
    "no."
)


_ISSUE_KEYWORDS = (
    "issue",
    "date of issue"
)


_EXPIRY_KEYWORDS = (
    "expiry",
    "expiration",
    "date of expiry",
    "valid until"
)


def _extract_field_near_keyword(
    lines: list[str],
    keywords: tuple[str, ...],
    pattern: Optional[re.Pattern] = None
) -> Optional[str]:
    """
    Extract a field value located near a known keyword.
    """

    sorted_kws = sorted(
        keywords,
        key=len,
        reverse=True
    )

    lowered = [
        line.lower()
        for line in lines
    ]

    for i, line_lower in enumerate(lowered):

        matching_kw = None

        for keyword in sorted_kws:

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                line_lower
            ):
                matching_kw = keyword
                break

        if not matching_kw:
            continue

        original_line = lines[i]

        # ---------------------------------------------
        # Look for regex value on same line
        # ---------------------------------------------

        if pattern:

            match = pattern.search(
                original_line
            )

            if match:
                return match.group(0)

        # ---------------------------------------------
        # Remove keyword and return remaining text
        # ---------------------------------------------

        position = line_lower.find(
            matching_kw
        )

        if position != -1:

            remainder = original_line[
                position + len(matching_kw):
            ].strip(" :.-")

            if remainder:

                return remainder

        # ---------------------------------------------
        # Check next line
        # ---------------------------------------------

        if i + 1 < len(lines):

            next_line = lines[i + 1].strip()

            if pattern:

                match = pattern.search(
                    next_line
                )

                if match:
                    return match.group(0)

            elif next_line:

                return next_line

    return None


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_fields(
    image: np.ndarray,
    prefer_engine: str = "tesseract"
) -> OCRResult:
    """
    Extract OCR text and structured fields from a document image.

    This implementation intentionally uses Tesseract as the
    primary lightweight OCR engine to avoid loading large
    neural-network OCR models during deployment.
    """

    logger.info(
        "Starting OCR extraction..."
    )

    if image is None or image.size == 0:

        return OCRResult(
            engine_used="none",
            error="Invalid or empty image."
        )

    if not _tesseract_available():

        return OCRResult(
            engine_used="none",
            error=(
                "Tesseract is not available. "
                "Install pytesseract and the Tesseract binary."
            )
        )

    try:

        lines = _run_tesseract(
            image
        )

        logger.info(
            "Tesseract returned %d lines.",
            len(lines)
        )

    except Exception as e:

        logger.exception(
            "OCR extraction failed."
        )

        return OCRResult(
            engine_used="Tesseract",
            error=str(e)
        )

    full_text = "\n".join(lines)

    logger.info(
        "OCR completed using Tesseract."
    )

    # --------------------------------------------------------
    # MRZ PARSING
    # --------------------------------------------------------

    try:

        mrz = parse_mrz(
            lines
        )

    except Exception as e:

        logger.warning(
            "MRZ parsing failed: %s",
            e
        )

        mrz = None

    result = OCRResult(

        engine_used="Tesseract",

        raw_lines=lines,

        full_text=full_text,

        mrz=mrz

    )

    # --------------------------------------------------------
    # Prefer MRZ fields
    # --------------------------------------------------------

    if mrz and mrz.found:

        result.document_number = (
            mrz.document_number
        )

        result.date_of_birth = (
            mrz.date_of_birth
        )

        result.expiry_date = (
            mrz.expiry_date
        )

        result.nationality = (
            mrz.nationality
        )

        result.name = " ".join(
            filter(
                None,
                [
                    mrz.given_names,
                    mrz.surname
                ]
            )
        ) or None

    # --------------------------------------------------------
    # Fallback keyword extraction
    # --------------------------------------------------------

    if not result.name:

        result.name = (
            _extract_field_near_keyword(
                lines,
                _NAME_KEYWORDS
            )
        )

    if not result.date_of_birth:

        result.date_of_birth = (
            _extract_field_near_keyword(
                lines,
                _DOB_KEYWORDS,
                _DATE_RE
            )
        )

    if not result.document_number:

        result.document_number = (
            _extract_field_near_keyword(
                lines,
                _DOCNUM_KEYWORDS,
                _DOC_NUM_RE
            )
        )

    if not result.nationality:

        result.nationality = (
            _extract_field_near_keyword(
                lines,
                _NATIONALITY_KEYWORDS
            )
        )

    if not result.issue_date:

        result.issue_date = (
            _extract_field_near_keyword(
                lines,
                _ISSUE_KEYWORDS,
                _DATE_RE
            )
        )

    if not result.expiry_date:

        result.expiry_date = (
            _extract_field_near_keyword(
                lines,
                _EXPIRY_KEYWORDS,
                _DATE_RE
            )
        )

    logger.info(
        "OCR extraction finished successfully."
    )

    return result