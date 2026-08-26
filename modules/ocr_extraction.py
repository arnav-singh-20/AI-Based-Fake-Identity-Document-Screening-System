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

The OCR pipeline:

1. Preprocesses the document image
2. Tries Tesseract and EasyOCR
3. Uses the result containing the most detected text
4. Extracts MRZ information
5. Extracts structured fields using keywords and regex
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.mrz_utils import parse_mrz, MRZResult


logger = logging.getLogger(__name__)


_EASYOCR_READER = None
_EASYOCR_AVAILABLE = None
_TESSERACT_AVAILABLE = None


# ============================================================
# ENGINE AVAILABILITY
# ============================================================

def _easyocr_available() -> bool:

    global _EASYOCR_AVAILABLE

    if _EASYOCR_AVAILABLE is None:

        try:

            import easyocr  # noqa: F401

            _EASYOCR_AVAILABLE = True

            logger.info("EasyOCR is available.")

        except Exception as e:

            logger.warning(
                "EasyOCR unavailable: %s",
                e
            )

            _EASYOCR_AVAILABLE = False

    return _EASYOCR_AVAILABLE


def _tesseract_available() -> bool:

    global _TESSERACT_AVAILABLE

    if _TESSERACT_AVAILABLE is None:

        try:

            import pytesseract

            pytesseract.get_tesseract_version()

            _TESSERACT_AVAILABLE = True

            logger.info("Tesseract is available.")

        except Exception as e:

            logger.warning(
                "Tesseract unavailable: %s",
                e
            )

            _TESSERACT_AVAILABLE = False

    return _TESSERACT_AVAILABLE


# ============================================================
# EASY OCR
# ============================================================

def _get_easyocr_reader():

    global _EASYOCR_READER

    if _EASYOCR_READER is None:

        import easyocr

        logger.info(
            "Loading EasyOCR model..."
        )

        _EASYOCR_READER = easyocr.Reader(
            ["en"],
            gpu=False
        )

    return _EASYOCR_READER


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class OCRResult:

    engine_used: str

    raw_lines: list = field(
        default_factory=list
    )

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

def _preprocess_image(
    image: np.ndarray
) -> np.ndarray:

    """
    Preprocess image before OCR.

    Improves OCR performance for:
    - Low contrast documents
    - Small text
    - Slightly dark images
    """

    import cv2

    if image is None:

        raise ValueError(
            "OCR received an empty image."
        )

    if image.size == 0:

        raise ValueError(
            "OCR received an invalid image."
        )

    # Convert RGB to grayscale
    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image.copy()

    # Upscale small images
    height, width = gray.shape[:2]

    if width < 1200:

        scale = 2

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Mild denoising
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    processed = clahe.apply(
        gray
    )

    return processed


# ============================================================
# TESSERACT
# ============================================================

def _run_tesseract(
    image: np.ndarray
) -> list[str]:

    import pytesseract
    from PIL import Image

    pil_img = Image.fromarray(
        image
    )

    # PSM 6 = assume block of text
    text = pytesseract.image_to_string(
        pil_img,
        config="--oem 3 --psm 6"
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:

            lines.append(
                line
            )

    logger.info(
        "Tesseract detected %d lines.",
        len(lines)
    )

    return lines


# ============================================================
# EASY OCR
# ============================================================

def _run_easyocr(
    image: np.ndarray
) -> list[str]:

    reader = _get_easyocr_reader()

    results = reader.readtext(
        image,
        detail=0,
        paragraph=False
    )

    lines = [
        str(result).strip()
        for result in results
        if str(result).strip()
    ]

    logger.info(
        "EasyOCR detected %d lines.",
        len(lines)
    )

    return lines


# ============================================================
# REGEX PATTERNS
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
    "full name",
    "given names",
    "given name",
    "surname",
    "name"
)


_DOB_KEYWORDS = (
    "date of birth",
    "birth date",
    "birth",
    "dob"
)


_NATIONALITY_KEYWORDS = (
    "nationality",
    "nation"
)


_DOCNUM_KEYWORDS = (
    "passport number",
    "passport no",
    "document number",
    "document no",
    "id number",
    "id no",
    "number",
    "no."
)


_ISSUE_KEYWORDS = (
    "date of issue",
    "issue date",
    "issued",
    "issue"
)


_EXPIRY_KEYWORDS = (
    "date of expiry",
    "expiry date",
    "expiration date",
    "expiry",
    "expiration"
)


# ============================================================
# FIELD EXTRACTION
# ============================================================

def _extract_field_near_keyword(
    lines: list[str],
    keywords: tuple[str, ...],
    pattern: Optional[re.Pattern] = None
) -> Optional[str]:

    sorted_keywords = sorted(
        keywords,
        key=len,
        reverse=True
    )

    lowered_lines = [
        line.lower()
        for line in lines
    ]

    for i, lowered_line in enumerate(
        lowered_lines
    ):

        matched_keyword = None

        for keyword in sorted_keywords:

            if keyword in lowered_line:

                matched_keyword = keyword

                break

        if not matched_keyword:

            continue

        original_line = lines[i]

        # Try regex on same line
        if pattern:

            match = pattern.search(
                original_line
            )

            if match:

                return match.group(0)

        # Extract text after keyword
        position = lowered_line.find(
            matched_keyword
        )

        if position != -1:

            remainder = original_line[
                position + len(matched_keyword):
            ]

            remainder = remainder.strip(
                " :-|."
            )

            if remainder:

                return remainder

        # Try next line
        if i + 1 < len(lines):

            next_line = lines[
                i + 1
            ].strip()

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
    prefer_engine: str = "auto"
) -> OCRResult:

    """
    Run OCR on document image.

    The image is preprocessed first.

    Both OCR engines are attempted when possible,
    and the engine producing the most text is selected.
    """

    logger.info(
        "Starting OCR extraction..."
    )

    try:

        processed_image = _preprocess_image(
            image
        )

    except Exception as e:

        logger.exception(
            "Image preprocessing failed."
        )

        return OCRResult(
            engine_used="none",
            error=f"Image preprocessing failed: {e}"
        )


    # --------------------------------------------------------
    # Decide engine order
    # --------------------------------------------------------

    if prefer_engine == "easyocr":

        engines = [
            "easyocr",
            "tesseract"
        ]

    elif prefer_engine == "tesseract":

        engines = [
            "tesseract",
            "easyocr"
        ]

    else:

        engines = [
            "tesseract",
            "easyocr"
        ]


    best_lines = []

    best_engine = None

    last_error = None


    # --------------------------------------------------------
    # Run OCR engines
    # --------------------------------------------------------

    for engine in engines:

        try:

            if (
                engine == "tesseract"
                and _tesseract_available()
            ):

                lines = _run_tesseract(
                    processed_image
                )

                logger.info(
                    "Tesseract returned %d lines.",
                    len(lines)
                )


            elif (
                engine == "easyocr"
                and _easyocr_available()
            ):

                lines = _run_easyocr(
                    processed_image
                )

                logger.info(
                    "EasyOCR returned %d lines.",
                    len(lines)
                )

            else:

                continue


            # Keep the OCR result with more text
            if len(lines) > len(best_lines):

                best_lines = lines

                best_engine = engine


            # If OCR found useful text,
            # continue checking the other engine
            # for potentially better results.

        except Exception as e:

            logger.warning(
                "%s OCR failed: %s",
                engine,
                e
            )

            last_error = str(e)


    # --------------------------------------------------------
    # No OCR text found
    # --------------------------------------------------------

    if not best_lines:

        return OCRResult(

            engine_used="none",

            raw_lines=[],

            full_text="",

            error=(
                last_error
                or
                "OCR completed but no readable text "
                "was detected."
            )
        )


    # --------------------------------------------------------
    # Prepare OCR result
    # --------------------------------------------------------

    full_text = "\n".join(
        best_lines
    )


    logger.info(
        "OCR completed using %s. "
        "Detected %d lines.",
        best_engine,
        len(best_lines)
    )


    # --------------------------------------------------------
    # MRZ EXTRACTION
    # --------------------------------------------------------

    mrz = parse_mrz(
        best_lines
    )


    engine_display = (
        "EasyOCR"
        if best_engine == "easyocr"
        else "Tesseract"
    )


    result = OCRResult(

        engine_used=engine_display,

        raw_lines=best_lines,

        full_text=full_text,

        mrz=mrz
    )


    # --------------------------------------------------------
    # MRZ FIELDS
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
    # KEYWORD / REGEX FALLBACK
    # --------------------------------------------------------

    if not result.name:

        result.name = (
            _extract_field_near_keyword(
                best_lines,
                _NAME_KEYWORDS
            )
        )


    if not result.date_of_birth:

        result.date_of_birth = (
            _extract_field_near_keyword(
                best_lines,
                _DOB_KEYWORDS,
                _DATE_RE
            )
        )


    if not result.document_number:

        result.document_number = (
            _extract_field_near_keyword(
                best_lines,
                _DOCNUM_KEYWORDS,
                _DOC_NUM_RE
            )
        )


    if not result.nationality:

        result.nationality = (
            _extract_field_near_keyword(
                best_lines,
                _NATIONALITY_KEYWORDS
            )
        )


    if not result.issue_date:

        result.issue_date = (
            _extract_field_near_keyword(
                best_lines,
                _ISSUE_KEYWORDS,
                _DATE_RE
            )
        )


    if not result.expiry_date:

        result.expiry_date = (
            _extract_field_near_keyword(
                best_lines,
                _EXPIRY_KEYWORDS,
                _DATE_RE
            )
        )


    return result