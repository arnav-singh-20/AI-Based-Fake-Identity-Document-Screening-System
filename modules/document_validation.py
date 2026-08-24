"""
Step 3 — Document Validation

Pure rule-based checks — no ML involved:
  - Date logic (expiry after issue, issue not in the future)
  - Field format checks (document number pattern, nationality vs ISO codes)
  - MRZ checksum validation (recompute check digits, compare to printed)

Outputs a Document Validation Score (0-100) = internal consistency of
the document's own data.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from modules.ocr_extraction import OCRResult
from utils.iso_codes import is_valid_country_code

# Passport/ID document numbers vary by issuing country (e.g. "L898902C3"),
# so this checks a broad but still meaningful shape: 6-9 alphanumeric
# characters, at least one digit, no spaces or symbols — not a strict
# per-country format (that's what the MRZ checksum check is for).
_DOC_NUM_FORMAT_RE = re.compile(r"^(?=[A-Z0-9]{6,9}$)(?=.*\d)[A-Z0-9]{6,9}$")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationResult:
    checks: list = field(default_factory=list)  # list[CheckResult]
    score: float = 0.0  # 0-100, higher = more internally consistent

    def add(self, name: str, passed: bool, detail: str):
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))


def _parse_yymmdd(s: str) -> Optional[date]:
    """Parse an MRZ-style YYMMDD date, with a simple century pivot."""
    if not s or not re.match(r"^\d{6}$", s):
        return None
    yy, mm, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
    # Pivot: 00-30 -> 2000s, 31-99 -> 1900s (typical passport convention,
    # since expiry dates rarely exceed +10 years from issue and DOBs
    # rarely predate the 1930s for a living traveler).
    year = 2000 + yy if yy <= 30 else 1900 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def _parse_free_date(s: str) -> Optional[date]:
    """Best-effort parse of a free-text date string in several common formats."""
    if not s:
        return None
    s = s.strip()
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d",
               "%d/%m/%y", "%d-%m-%y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_any_date(s: str) -> Optional[date]:
    if not s:
        return None
    if re.match(r"^\d{6}$", s):
        return _parse_yymmdd(s)
    return _parse_free_date(s)


def validate_document(ocr: OCRResult) -> ValidationResult:
    result = ValidationResult()

    issue_dt = _parse_any_date(ocr.issue_date) if ocr.issue_date else None
    expiry_dt = _parse_any_date(ocr.expiry_date) if ocr.expiry_date else None
    today = date.today()

    # --- Date logic -----------------------------------------------
    if issue_dt and expiry_dt:
        result.add(
            "Expiry after issue date",
            expiry_dt > issue_dt,
            f"Issue: {issue_dt.isoformat()}, Expiry: {expiry_dt.isoformat()}",
        )
    else:
        result.add(
            "Expiry after issue date",
            True,  # can't fail a check we can't evaluate; don't penalize
            "Skipped — issue and/or expiry date not extracted",
        )

    if issue_dt:
        result.add(
            "Issue date not in the future",
            issue_dt <= today,
            f"Issue date: {issue_dt.isoformat()}",
        )
    else:
        result.add("Issue date not in the future", True, "Skipped — issue date not extracted")

    if expiry_dt:
        result.add(
            "Document not expired",
            expiry_dt >= today,
            f"Expiry date: {expiry_dt.isoformat()}",
        )
    else:
        result.add("Document not expired", True, "Skipped — expiry date not extracted")

    # --- Field format checks ----------------------------------------
    if ocr.document_number:
        fmt_ok = bool(_DOC_NUM_FORMAT_RE.match(ocr.document_number.upper()))
        result.add(
            "Document number format",
            fmt_ok,
            f"'{ocr.document_number}' {'matches' if fmt_ok else 'does not match'} expected pattern",
        )
    else:
        result.add("Document number format", False, "No document number extracted")

    if ocr.nationality:
        nat_ok = is_valid_country_code(ocr.nationality)
        result.add(
            "Nationality code is a valid ISO code",
            nat_ok,
            f"'{ocr.nationality}'",
        )
    else:
        result.add("Nationality code is a valid ISO code", True, "Skipped — no nationality extracted")

    # --- MRZ checksum validation --------------------------------------
    if ocr.mrz and ocr.mrz.found:
        result.add(
            "MRZ document number checksum",
            bool(ocr.mrz.document_number_valid),
            "Recomputed check digit vs printed digit",
        )
        result.add(
            "MRZ date-of-birth checksum",
            bool(ocr.mrz.dob_valid),
            "Recomputed check digit vs printed digit",
        )
        result.add(
            "MRZ expiry-date checksum",
            bool(ocr.mrz.expiry_valid),
            "Recomputed check digit vs printed digit",
        )
        if ocr.mrz.composite_valid is not None:
            result.add(
                "MRZ composite checksum",
                bool(ocr.mrz.composite_valid),
                "Recomputed composite check digit vs printed digit — "
                "the strongest single tamper signal (forgers routinely "
                "forget to recompute this after editing a field)",
            )
    else:
        result.add("MRZ checksum validation", False, "No MRZ block detected on document")

    # --- Score: percentage of checks passed --------------------------
    if result.checks:
        result.score = 100.0 * sum(1 for c in result.checks if c.passed) / len(result.checks)
    else:
        result.score = 0.0

    return result
