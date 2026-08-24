"""
Step 2 — OCR Extraction

Reads a document image and extracts structured fields (name, DOB,
document number, nationality, issue/expiry dates) plus the raw MRZ
lines if present. Tries EasyOCR first (better accuracy on ID-style
fonts); falls back to Tesseract if EasyOCR isn't installed/available
or fails at runtime. Nothing here is trained — both are pretrained,
pip-installable OCR engines.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.mrz_utils import parse_mrz, MRZResult

logger = logging.getLogger(__name__)

# --- Engine availability (checked lazily so the app runs even if one
#     of the two OCR backends isn't installed) -----------------------

_EASYOCR_READER = None
_EASYOCR_AVAILABLE = None
_TESSERACT_AVAILABLE = None


def _easyocr_available() -> bool:
    global _EASYOCR_AVAILABLE
    if _EASYOCR_AVAILABLE is None:
        try:
            import easyocr  # noqa: F401
            _EASYOCR_AVAILABLE = True
        except Exception:
            _EASYOCR_AVAILABLE = False
    return _EASYOCR_AVAILABLE


def _tesseract_available() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        try:
            import pytesseract  # noqa: F401
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


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


def _run_easyocr(image: np.ndarray) -> list[str]:
    reader = _get_easyocr_reader()
    results = reader.readtext(image, detail=0, paragraph=False)
    return [str(r) for r in results]


def _run_tesseract(image: np.ndarray) -> list[str]:
    import pytesseract
    from PIL import Image

    if image.ndim == 3:
        pil_img = Image.fromarray(image)
    else:
        pil_img = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_img)
    return [line for line in text.splitlines() if line.strip()]


# --- Lightweight field extraction heuristics -------------------------
# These are intentionally simple, transparent regex/keyword rules —
# consistent with the doc's "zero-training" design principle. Tune the
# keyword lists to whatever document types you actually demo with.

_DATE_RE = re.compile(r"\b(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}|\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2})\b")
_DOC_NUM_RE = re.compile(r"\b([A-Z]{1,2}\d{6,9}|\d{9,12})\b")

_NAME_KEYWORDS = ("name", "surname", "given name", "given names")
_DOB_KEYWORDS = ("birth", "dob", "date of birth")
_NATIONALITY_KEYWORDS = ("nationality", "nation")
_DOCNUM_KEYWORDS = ("passport no", "document no", "id no", "no.", "number")
_ISSUE_KEYWORDS = ("issue", "date of issue")
_EXPIRY_KEYWORDS = ("expiry", "expiration", "date of expiry")


def _extract_field_near_keyword(lines: list[str], keywords: tuple[str, ...], pattern: Optional[re.Pattern] = None) -> Optional[str]:
    """
    Look for a keyword line, then pull the value either from the same
    line (after the keyword) or the next line. If a regex pattern is
    given, prefer a value in the line matching that pattern.
    """
    # Sort keywords by length descending (most-specific first)
    sorted_kws = sorted(keywords, key=len, reverse=True)
    
    lowered = [l.lower() for l in lines]
    for i, l in enumerate(lowered):
        # Match using word boundaries to avoid matching substrings like "name" in "surname"
        matching_kw = None
        for kw in sorted_kws:
            if re.search(r'\b' + re.escape(kw) + r'\b', l):
                matching_kw = kw
                break
                
        if matching_kw:
            same_line = lines[i]
            if pattern:
                m = pattern.search(same_line)
                if m:
                    return m.group(0)
            # try stripping the keyword prefix off the same line
            idx = l.find(matching_kw)
            if idx != -1:
                remainder = lines[i][idx + len(matching_kw):].strip(" :.-")
                if remainder:
                    return remainder
            # else check the next line
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if pattern:
                    m = pattern.search(nxt)
                    if m:
                        return m.group(0)
                elif nxt:
                    return nxt
    return None


def extract_fields(image: np.ndarray, prefer_engine: str = "auto") -> OCRResult:
    """
    Run OCR on `image` (a numpy array, RGB) and extract structured
    fields. `prefer_engine`: "easyocr", "tesseract", or "auto"
    (try EasyOCR first, fall back to Tesseract).
    """
    engine_used = None
    lines: list[str] = []
    error = None

    order = []
    if prefer_engine == "easyocr":
        order = ["easyocr", "tesseract"]
    elif prefer_engine == "tesseract":
        order = ["tesseract", "easyocr"]
    else:
        order = ["easyocr", "tesseract"]

    for engine in order:
        try:
            if engine == "easyocr" and _easyocr_available():
                lines = _run_easyocr(image)
                engine_used = "EasyOCR"
                break
            if engine == "tesseract" and _tesseract_available():
                lines = _run_tesseract(image)
                engine_used = "Tesseract"
                break
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR engine %s failed: %s", engine, e)
            error = str(e)
            continue

    if engine_used is None:
        return OCRResult(
            engine_used="none",
            error=error or "No OCR engine available. Install easyocr or pytesseract.",
        )

    full_text = "\n".join(lines)
    mrz = parse_mrz(lines)

    result = OCRResult(
        engine_used=engine_used,
        raw_lines=lines,
        full_text=full_text,
        mrz=mrz,
    )

    # Prefer MRZ-derived fields when available (far more reliable than
    # free-text field matching), fall back to keyword heuristics.
    if mrz.found:
        result.document_number = mrz.document_number
        result.date_of_birth = mrz.date_of_birth
        result.expiry_date = mrz.expiry_date
        result.nationality = mrz.nationality
        result.name = " ".join(filter(None, [mrz.given_names, mrz.surname])) or None

    if not result.name:
        result.name = _extract_field_near_keyword(lines, _NAME_KEYWORDS)
    if not result.date_of_birth:
        result.date_of_birth = _extract_field_near_keyword(lines, _DOB_KEYWORDS, _DATE_RE)
    if not result.document_number:
        result.document_number = _extract_field_near_keyword(lines, _DOCNUM_KEYWORDS, _DOC_NUM_RE)
    if not result.nationality:
        result.nationality = _extract_field_near_keyword(lines, _NATIONALITY_KEYWORDS)
    if not result.issue_date:
        result.issue_date = _extract_field_near_keyword(lines, _ISSUE_KEYWORDS, _DATE_RE)
    if not result.expiry_date:
        result.expiry_date = _extract_field_near_keyword(lines, _EXPIRY_KEYWORDS, _DATE_RE)

    return result
