"""
Comprehensive test suite for the Document Screening System.

Covers:
  - MRZ checksum computation & verification
  - MRZ noise tolerance (TD3 and TD1)
  - MRZ checksum NOT auto-corrected (forgery signal preservation)
  - OCR keyword word-boundary regression (name vs surname)
  - OCR most-specific-first keyword matching
  - Document validation (passing and failing cases)
  - Risk scoring (pass, flag, edge cases)
  - Tamper detection (smoke test)
  - Copy-move detection (smoke test)
  - Screening history (SQLite audit log)
  - Config loading
  - Face verification padding
"""

import sys
import os
import json
import tempfile
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.mrz_utils import parse_mrz, compute_check_digit, verify_check_digit, find_mrz_lines, MRZResult
from modules.ocr_extraction import _extract_field_near_keyword, _NAME_KEYWORDS, _DOB_KEYWORDS
from modules.document_validation import validate_document
from modules.risk_scoring import compute_risk_score
from modules.tamper_detection import detect_tampering, compute_ela, _detect_copy_move, _localized_ela_score
from modules.face_verification import _pad_to_min_size
from utils.screening_history import log_screening, get_history, clear_history


# ==============================================================================
# 1. MRZ Checksum Tests
# ==============================================================================

def test_mrz_checksum_computation():
    assert compute_check_digit("520727") == 3
    assert verify_check_digit("520727", "3") is True
    assert verify_check_digit("520727", "4") is False


def test_mrz_checksum_not_autocorrected():
    """Checksums are a forgery signal — they must NEVER be auto-corrected."""
    # Deliberately wrong check digit
    assert verify_check_digit("520727", "9") is False
    # Empty or non-digit check digit
    assert verify_check_digit("520727", "") is False
    assert verify_check_digit("520727", "X") is False


def test_mrz_char_value_edge_cases():
    from utils.mrz_utils import _char_value
    assert _char_value("<") == 0
    assert _char_value("0") == 0
    assert _char_value("9") == 9
    assert _char_value("A") == 10
    assert _char_value("Z") == 35
    # Unexpected chars treated as 0
    assert _char_value("!") == 0
    assert _char_value(" ") == 0


# ==============================================================================
# 2. MRZ Noise Tolerance Tests
# ==============================================================================

def test_mrz_noise_tolerance_td3():
    line1 = "P<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    line2 = "L898902C<3USA7408122F1204159ZE184226B<<<<<10"
    # Truncated to 40 chars (4 dropped)
    res = parse_mrz([line1[:40], line2[:40]])
    assert res.found is True
    assert res.format == "TD3"
    assert res.low_confidence is True
    assert res.document_number == "L898902C"


def test_mrz_noise_tolerance_td1():
    line1 = "I<UTD1234567819<<<<<<<<<<<<<<"
    line2 = "6508122F1204159UTD<<<<<<<<<<<"
    line3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<"
    res = parse_mrz([line1[:27], line2[:27], line3[:27]])
    assert res.found is True
    assert res.format == "TD1"
    assert res.low_confidence is True


def test_mrz_full_length_not_low_confidence():
    line1 = "P<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    line2 = "L898902C<3USA7408122F1204159ZE184226B<<<<<10"
    res = parse_mrz([line1, line2])
    assert res.found is True
    assert res.low_confidence is False


def test_mrz_not_found_garbage():
    res = parse_mrz(["hello world", "this is not mrz", "short"])
    assert res.found is False


def test_find_mrz_lines_strips_punctuation():
    """OCR sometimes inserts stray punctuation — it should be cleaned."""
    lines = ["P<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<..", "L898902C<3USA7408122F1204159ZE184226B<<<<<10!"]
    cleaned = find_mrz_lines(lines)
    assert len(cleaned) == 2
    assert all(c.isalpha() or c.isdigit() or c == "<" for line in cleaned for c in line)


# ==============================================================================
# 3. OCR Keyword Extraction & Regression Tests
# ==============================================================================

def test_ocr_keyword_word_boundary_regression():
    """Bug Regression: 'name' must NOT match inside 'surname'."""
    lines = ["Surname: Smith", "Given Names: John"]
    assert _extract_field_near_keyword(lines, ("name",)) is None
    assert _extract_field_near_keyword(lines, ("given names", "given name")) == "John"


def test_ocr_keyword_most_specific_first():
    lines = ["Given Name: John"]
    extracted = _extract_field_near_keyword(lines, ("name", "given name"))
    assert extracted == "John"


def test_ocr_keyword_empty_lines():
    assert _extract_field_near_keyword([], ("name",)) is None


def test_ocr_keyword_value_on_next_line():
    lines = ["Name:", "Alice Bob"]
    assert _extract_field_near_keyword(lines, ("name",)) == "Alice Bob"


# ==============================================================================
# 4. Document Validation Tests
# ==============================================================================

class MockOCRResult:
    def __init__(self, name="John Doe", dob="800101", doc_num="L898902C", nat="USA",
                 issue="2010-01-01", expiry="2030-01-01", mrz=None):
        self.engine_used = "EasyOCR"
        self.name = name
        self.date_of_birth = dob
        self.document_number = doc_num
        self.nationality = nat
        self.issue_date = issue
        self.expiry_date = expiry
        self.mrz = mrz
        self.error = None
        self.full_text = "Mock Text"
        self.raw_lines = ["Mock"]


def test_document_validation_all_pass():
    mrz = MRZResult(
        found=True, format="TD3", document_number="L898902C",
        document_number_valid=True, date_of_birth="800102", dob_valid=True,
        expiry_date="300102", expiry_valid=True, nationality="USA", sex="M",
        surname="Smith", given_names="John", composite_valid=True,
        all_checks_passed=True
    )
    ocr = MockOCRResult(dob="800102", mrz=mrz)
    val_res = validate_document(ocr)
    assert val_res.score == 100.0
    assert len([c for c in val_res.checks if not c.passed]) == 0


def test_document_validation_no_mrz():
    ocr = MockOCRResult(mrz=MRZResult(found=False))
    val_res = validate_document(ocr)
    # Should lose points for no MRZ but not crash
    assert val_res.score < 100.0
    assert any("MRZ" in c.name for c in val_res.checks)


def test_document_validation_expired():
    ocr = MockOCRResult(expiry="2020-01-01", mrz=MRZResult(found=False))
    val_res = validate_document(ocr)
    failed = [c for c in val_res.checks if not c.passed]
    assert any("expired" in c.name.lower() or "expir" in c.name.lower() for c in failed)


# ==============================================================================
# 5. Risk Scoring Tests
# ==============================================================================

def test_risk_scoring_pass():
    weights = {"validation": 0.30, "tamper": 0.10, "face": 0.60}
    res = compute_risk_score(100.0, 17.82, 89.18, weights=weights, threshold=25.0)
    assert res.verdict == "PASS"
    assert res.risk_score < 25.0


def test_risk_scoring_flag():
    weights = {"validation": 0.30, "tamper": 0.10, "face": 0.60}
    res = compute_risk_score(66.67, 7.13, 61.04, weights=weights, threshold=25.0)
    assert res.verdict == "FLAG FOR REVIEW"
    assert res.risk_score >= 25.0


def test_risk_scoring_perfect_document():
    res = compute_risk_score(100.0, 0.0, 100.0, threshold=25.0)
    assert res.risk_score == 0.0
    assert res.verdict == "PASS"


def test_risk_scoring_worst_document():
    res = compute_risk_score(0.0, 100.0, 0.0, threshold=25.0)
    assert res.risk_score == 100.0
    assert res.verdict == "FLAG FOR REVIEW"


def test_risk_scoring_boundary():
    """Exactly at threshold should be FLAG."""
    weights = {"validation": 1.0, "tamper": 0.0, "face": 0.0}
    res = compute_risk_score(75.0, 0.0, 100.0, weights=weights, threshold=25.0)
    assert res.risk_score == 25.0
    assert res.verdict == "FLAG FOR REVIEW"


# ==============================================================================
# 6. Tamper Detection Tests
# ==============================================================================

def test_detect_tampering_smoke():
    """Smoke test: detect_tampering should not crash on a random image."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = detect_tampering(img)
    assert 0 <= result.tamper_score <= 100
    assert result.heatmap_overlay.shape[:2] == img.shape[:2]
    assert hasattr(result, "ela_score")
    assert hasattr(result, "copy_move_score")


def test_compute_ela_output_shape():
    img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    ela = compute_ela(img)
    assert ela.shape == img.shape
    assert ela.dtype == np.uint8


def test_localized_ela_score_uniform():
    """Uniform gray should give a consistent score."""
    gray = np.full((128, 128), 50, dtype=np.uint8)
    score = _localized_ela_score(gray, block_size=64)
    assert abs(score - 50.0) < 1.0


def test_copy_move_on_clean_image():
    """A clean gradient image should have very few copy-move matches."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    for i in range(200):
        img[i, :, :] = i  # gradient
    score, clusters, mask = _detect_copy_move(img)
    assert score < 50  # shouldn't flag a clean gradient


# ==============================================================================
# 7. Face Verification Padding Tests
# ==============================================================================

def test_pad_to_min_size_small_image():
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    padded = _pad_to_min_size(img, target_size=640)
    assert padded.shape[0] >= 640
    assert padded.shape[1] >= 640


def test_pad_to_min_size_large_image():
    """Large images should pass through unchanged."""
    img = np.zeros((800, 900, 3), dtype=np.uint8)
    padded = _pad_to_min_size(img, target_size=640)
    assert padded.shape == img.shape


# ==============================================================================
# 8. Screening History Tests
# ==============================================================================

def test_screening_history_roundtrip():
    """Log a result and read it back."""
    clear_history()
    log_screening(
        filename="test_doc.jpg", validation_score=85.0, tamper_score=12.0,
        face_match_score=95.0, risk_score=8.0, verdict="PASS",
        notes="unit test", ela_score=10.0, copy_move_score=2.0,
    )
    rows = get_history(limit=10)
    assert len(rows) >= 1
    latest = rows[0]
    assert latest["filename"] == "test_doc.jpg"
    assert latest["verdict"] == "PASS"
    assert abs(latest["risk_score"] - 8.0) < 0.01
    clear_history()


def test_screening_history_clear():
    log_screening(filename="to_delete.jpg", validation_score=0, tamper_score=0,
                  face_match_score=0, risk_score=50, verdict="FLAG FOR REVIEW")
    clear_history()
    assert len(get_history()) == 0


# ==============================================================================
# 9. Config Loading Tests
# ==============================================================================

def test_config_json_exists_and_valid():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert "weights" in cfg
    assert abs(sum(cfg["weights"].values()) - 1.0) < 0.01
    assert "threshold" in cfg
    assert isinstance(cfg["threshold"], (int, float))
