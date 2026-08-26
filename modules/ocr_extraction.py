"""
Step 2 — OCR Extraction

Lightweight OCR extraction optimized for Streamlit Cloud.

Uses Tesseract as the primary OCR engine.
EasyOCR is only attempted if Tesseract fails completely.

This prevents unnecessary EasyOCR/PyTorch model downloads and
reduces memory usage during deployment.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.mrz_utils import parse_mrz, MRZResult


logger = logging.getLogger(__name__)


# --------------------------------------------------
# ENGINE STATE
# --------------------------------------------------

_EASYOCR_READER = None
_EASYOCR_AVAILABLE = None
_TESSERACT_AVAILABLE = None


# --------------------------------------------------
# ENGINE AVAILABILITY
# --------------------------------------------------

def _tesseract_available() -> bool:
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


def _easyocr_available() -> bool:
    global _EASYOCR_AVAILABLE

    if _EASYOCR_AVAILABLE is None:
        try:
            import easyocr  # noqa: F401

            _EASYOCR_AVAILABLE = True
            logger.info("EasyOCR is available.")

        except Exception as e:
            logger.warning("EasyOCR unavailable: %s", e)
            _EASYOCR_AVAILABLE = False

    return _EASYOCR_AVAILABLE


def _get_easyocr_reader():
    global _EASYOCR_READER

    if _EASYOCR_READER is None:
        import easyocr

        logger.info("Loading EasyOCR model...")

        _EASYOCR_READER = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

    return _EASYOCR_READER


# --------------------------------------------------
# RESULT CLASS
# --------------------------------------------------

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


# --------------------------------------------------
# OCR ENGINES
# --------------------------------------------------

def _run_tesseract(image: np.ndarray) -> list[str]:

    import pytesseract
    from PIL import Image

    if image is None or image.size == 0:
        return []

    try:
        pil_image = Image.fromarray(image)

        text = pytesseract.image_to_string(
            pil_image,
            config="--oem 3 --psm 6"
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        logger.info(
            "Tesseract detected %d lines.",
            len(lines)
        )

        return lines

    except Exception as e:

        logger.exception(
            "Tesseract OCR failed."
        )

        raise e


def _run_easyocr(image: np.ndarray) -> list[str]:

    reader = _get_easyocr_reader()

    results = reader.readtext(
        image,
        detail=0,
        paragraph=False
    )

    lines = [
        str(item).strip()
        for item in results
        if str(item).strip()
    ]

    logger.info(
        "EasyOCR detected %d lines.",
        len(lines)
    )

    return lines


# --------------------------------------------------
# FIELD EXTRACTION
# --------------------------------------------------

_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}"
    r"|"
    r"\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2}"
    r")\b"
)


_DOC_NUM_RE = re.compile(
    r"\b([A-Z]{1,2}\d{6,9}|\d{9,12})\b"
)


_NAME_KEYWORDS = (
    "name",
    "surname",
    "given name",
    "given names",
)

_DOB_KEYWORDS = (
    "date of birth",
    "birth",
    "dob",
)

_NATIONALITY_KEYWORDS = (
    "nationality",
    "nation",
)

_DOCNUM_KEYWORDS = (
    "passport no",
    "document no",
    "document number",
    "id no",
    "number",
)

_ISSUE_KEYWORDS = (
    "date of issue",
    "issue date",
    "issue",
)

_EXPIRY_KEYWORDS = (
    "date of expiry",
    "expiry date",
    "expiration",
    "expiry",
)


def _extract_field_near_keyword(
    lines: list[str],
    keywords: tuple[str, ...],
    pattern: Optional[re.Pattern] = None,
) -> Optional[str]:

    sorted_keywords = sorted(
        keywords,
        key=len,
        reverse=True,
    )

    lowered_lines = [
        line.lower()
        for line in lines
    ]

    for i, lower_line in enumerate(lowered_lines):

        matched_keyword = None

        for keyword in sorted_keywords:

            if keyword in lower_line:

                matched_keyword = keyword
                break

        if not matched_keyword:
            continue

        original_line = lines[i]

        # Try extracting regex match
        # from the same line.

        if pattern is not None:

            match = pattern.search(
                original_line
            )

            if match:

                return match.group(0)

        # Extract text after keyword.

        index = lower_line.find(
            matched_keyword
        )

        if index != -1:

            value = original_line[
                index + len(matched_keyword):
            ].strip(" :-")

            if value:

                return value

        # Try next line.

        if i + 1 < len(lines):

            next_line = lines[
                i + 1
            ].strip()

            if pattern is not None:

                match = pattern.search(
                    next_line
                )

                if match:

                    return match.group(0)

            elif next_line:

                return next_line

    return None


# --------------------------------------------------
# MAIN OCR PIPELINE
# --------------------------------------------------

def extract_fields(
    image: np.ndarray,
    prefer_engine: str = "auto",
) -> OCRResult:

    logger.info(
        "Starting OCR extraction..."
    )

    lines: list[str] = []

    engine_used = "none"

    last_error = None


    # ==============================================
    # STEP 1: TRY TESSERACT FIRST
    # ==============================================

    if (
        prefer_engine in ("auto", "tesseract")
        and _tesseract_available()
    ):

        try:

            lines = _run_tesseract(
                image
            )

            logger.info(
                "Tesseract returned %d lines.",
                len(lines)
            )

            # IMPORTANT:
            # If Tesseract got text, STOP HERE.
            # Do NOT load EasyOCR.

            if lines:

                engine_used = "Tesseract"

        except Exception as e:

            last_error = str(e)

            logger.warning(
                "Tesseract failed: %s",
                e,
            )


    # ==============================================
    # STEP 2: EASYOCR ONLY IF TESSERACT GOT NOTHING
    # ==============================================

    if (
        not lines
        and prefer_engine != "tesseract"
        and _easyocr_available()
    ):

        try:

            logger.info(
                "Tesseract returned no text. "
                "Trying EasyOCR fallback..."
            )

            lines = _run_easyocr(
                image
            )

            if lines:

                engine_used = "EasyOCR"

        except Exception as e:

            last_error = str(e)

            logger.warning(
                "EasyOCR failed: %s",
                e,
            )


    # ==============================================
    # NO OCR RESULT
    # ==============================================

    if not lines:

        return OCRResult(
            engine_used="none",
            error=(
                last_error
                or "OCR could not extract text."
            ),
        )


    # ==============================================
    # PROCESS OCR RESULT
    # ==============================================

    full_text = "\n".join(
        lines
    )

    logger.info(
        "OCR completed using %s.",
        engine_used,
    )


    # MRZ extraction

    try:

        mrz = parse_mrz(
            lines
        )

    except Exception as e:

        logger.warning(
            "MRZ parsing failed: %s",
            e,
        )

        mrz = None


    result = OCRResult(

        engine_used=engine_used,

        raw_lines=lines,

        full_text=full_text,

        mrz=mrz,
    )


    # ==============================================
    # PREFER MRZ DATA
    # ==============================================

    if mrz is not None and mrz.found:

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
                    mrz.surname,
                ],
            )
        ) or None


    # ==============================================
    # FALLBACK FIELD EXTRACTION
    # ==============================================

    if not result.name:

        result.name = (
            _extract_field_near_keyword(
                lines,
                _NAME_KEYWORDS,
            )
        )


    if not result.date_of_birth:

        result.date_of_birth = (
            _extract_field_near_keyword(
                lines,
                _DOB_KEYWORDS,
                _DATE_RE,
            )
        )


    if not result.document_number:

        result.document_number = (
            _extract_field_near_keyword(
                lines,
                _DOCNUM_KEYWORDS,
                _DOC_NUM_RE,
            )
        )


    if not result.nationality:

        result.nationality = (
            _extract_field_near_keyword(
                lines,
                _NATIONALITY_KEYWORDS,
            )
        )


    if not result.issue_date:

        result.issue_date = (
            _extract_field_near_keyword(
                lines,
                _ISSUE_KEYWORDS,
                _DATE_RE,
            )
        )


    if not result.expiry_date:

        result.expiry_date = (
            _extract_field_near_keyword(
                lines,
                _EXPIRY_KEYWORDS,
                _DATE_RE,
            )
        )


    logger.info(
        "OCR extraction finished successfully."
    )

    return result