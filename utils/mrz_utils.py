"""
MRZ (Machine-Readable Zone) parsing and ICAO 9303 checksum validation.

Passports use a 2-line, 44-character-per-line MRZ (TD3 format).
ID cards / visas often use a 3-line, 30-character-per-line MRZ (TD1) —
both are handled here.

No ML involved: this is pure rule-based / arithmetic validation, exactly
as described in the design doc (Step 3 — Document Validation).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ICAO 9303 check-digit weighting pattern, applied cyclically.
_WEIGHTS = [7, 3, 1]


def _char_value(ch: str) -> int:
    """Map an MRZ character to its numeric value for checksum purposes."""
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    # Any unexpected character is treated as 0 (fails safe -> mismatch likely)
    return 0


def compute_check_digit(data: str) -> int:
    """Compute the ICAO 9303 check digit for a given MRZ data string."""
    total = 0
    for i, ch in enumerate(data):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def verify_check_digit(data: str, check_digit: str) -> bool:
    """Return True if the printed check digit matches the computed one."""
    if not check_digit or not check_digit.isdigit():
        return False
    return compute_check_digit(data) == int(check_digit)


@dataclass
class MRZResult:
    found: bool
    format: Optional[str] = None          # "TD3" (passport) or "TD1" (ID card)
    raw_lines: list = field(default_factory=list)
    document_number: Optional[str] = None
    document_number_valid: Optional[bool] = None
    date_of_birth: Optional[str] = None    # YYMMDD
    dob_valid: Optional[bool] = None
    expiry_date: Optional[str] = None      # YYMMDD
    expiry_valid: Optional[bool] = None
    nationality: Optional[str] = None
    sex: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    composite_valid: Optional[bool] = None
    all_checks_passed: Optional[bool] = None
    low_confidence: bool = False


_MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{26,44}$")


def find_mrz_lines(ocr_text_lines: list[str]) -> list[str]:
    """
    Scan raw OCR line output for MRZ-looking lines: long runs of
    uppercase letters, digits and '<' fillers. Tolerates some noise.
    """
    candidates = []
    for line in ocr_text_lines:
        cleaned = line.strip().upper().replace(" ", "")
        # Remove any stray punctuation that OCR might have inserted at start/end
        cleaned = re.sub(r"[^A-Z0-9<]", "", cleaned)
        if _MRZ_LINE_RE.match(cleaned):
            candidates.append(cleaned)
    return candidates


def parse_mrz(ocr_text_lines: list[str]) -> MRZResult:
    """
    Attempt to locate and parse an MRZ block from raw OCR line output.
    Supports TD3 (passport, 2x44) and TD1 (ID card, 3x30).
    Tolerates up to 4 dropped/misread characters.
    """
    lines = find_mrz_lines(ocr_text_lines)
    low_confidence = False

    # TD3: 2 lines of 44 chars, tolerate length down to 40 chars
    td3_lines = [l for l in lines if len(l) >= 40]
    if len(td3_lines) >= 2:
        l1 = td3_lines[0][:44]
        l2 = td3_lines[1][:44]
        if len(l1) < 44 or len(l2) < 44:
            low_confidence = True
        l1 = l1.ljust(44, "<")
        l2 = l2.ljust(44, "<")
        res = _parse_td3(l1, l2)
        res.low_confidence = low_confidence
        return res

    # TD1: 3 lines of 30 chars, tolerate length between 26 and 34 chars
    td1_lines = [l for l in lines if 26 <= len(l) <= 34]
    if len(td1_lines) >= 3:
        l1 = td1_lines[0][:30]
        l2 = td1_lines[1][:30]
        l3 = td1_lines[2][:30]
        if len(l1) < 30 or len(l2) < 30 or len(l3) < 30:
            low_confidence = True
        l1 = l1.ljust(30, "<")
        l2 = l2.ljust(30, "<")
        l3 = l3.ljust(30, "<")
        res = _parse_td1(l1, l2, l3)
        res.low_confidence = low_confidence
        return res

    return MRZResult(found=False)


def _parse_td3(line1: str, line2: str) -> MRZResult:
    """Parse a TD3 (passport) MRZ: line1 = names, line2 = data + checks."""
    doc_num = line2[0:9].replace("<", "")
    doc_num_check = line2[9]
    dob = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_check = line2[27]
    nationality = line2[10:13]

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    composite_check = line2[43] if len(line2) > 43 else ""

    names_part = line1[5:] if len(line1) > 5 else ""
    surname, _, given = names_part.partition("<<")
    surname = surname.replace("<", " ").strip()
    given_names = given.replace("<", " ").strip()

    doc_valid = verify_check_digit(line2[0:9], doc_num_check)
    dob_valid = verify_check_digit(dob, dob_check)
    expiry_valid = verify_check_digit(expiry, expiry_check)
    composite_valid = verify_check_digit(composite_data, composite_check) if composite_check else None

    checks = [doc_valid, dob_valid, expiry_valid]
    if composite_valid is not None:
        checks.append(composite_valid)

    return MRZResult(
        found=True,
        format="TD3",
        raw_lines=[line1, line2],
        document_number=doc_num,
        document_number_valid=doc_valid,
        date_of_birth=dob,
        dob_valid=dob_valid,
        expiry_date=expiry,
        expiry_valid=expiry_valid,
        nationality=nationality,
        sex=sex,
        surname=surname,
        given_names=given_names,
        composite_valid=composite_valid,
        all_checks_passed=all(checks),
    )


def _parse_td1(line1: str, line2: str, line3: str) -> MRZResult:
    """Parse a TD1 (ID card) MRZ: 3 lines of 30 chars."""
    doc_num = line1[5:14].replace("<", "")
    doc_num_check = line1[14]
    dob = line2[0:6]
    dob_check = line2[6]
    sex = line2[7]
    expiry = line2[8:14]
    expiry_check = line2[14]
    nationality = line2[15:18]

    names_part = line3
    surname, _, given = names_part.partition("<<")
    surname = surname.replace("<", " ").strip()
    given_names = given.replace("<", " ").strip()

    doc_valid = verify_check_digit(line1[5:14], doc_num_check)
    dob_valid = verify_check_digit(dob, dob_check)
    expiry_valid = verify_check_digit(expiry, expiry_check)

    checks = [doc_valid, dob_valid, expiry_valid]

    return MRZResult(
        found=True,
        format="TD1",
        raw_lines=[line1, line2, line3],
        document_number=doc_num,
        document_number_valid=doc_valid,
        date_of_birth=dob,
        dob_valid=dob_valid,
        expiry_date=expiry,
        expiry_valid=expiry_valid,
        nationality=nationality,
        sex=sex,
        surname=surname,
        given_names=given_names,
        composite_valid=None,
        all_checks_passed=all(checks),
    )
