"""
Step 6 — Risk Scoring Engine

Combines the three independent signals — Document Validation, Tamper,
and Face-Match — into a single weighted Risk Score (0-100).

Face-match and tamper detection are weighted more heavily than field
validation, since they're harder to fake convincingly, while a
validation-only failure (e.g. a wrong date format) alone is a weaker
signal.

    Risk Score = w_val * (100 - ValidationScore)
               + w_tamper * TamperScore
               + w_face * (100 - FaceMatchScore)

Weights are configurable (exposed as sliders in the Streamlit UI) —
this is a parameter to justify and tune, not a fixed law.
"""

import os
import json
from dataclasses import dataclass

DEFAULT_WEIGHTS = {
    "validation": 0.30,
    "tamper": 0.10,
    "face": 0.60,
}

DEFAULT_THRESHOLD = 25.0  # >= threshold -> FLAGGED FOR REVIEW

# Load from config.json if present
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
if os.path.exists(_config_path):
    try:
        with open(_config_path, "r", encoding="utf-8") as _f:
            _config = json.load(_f)
            if "weights" in _config:
                DEFAULT_WEIGHTS = _config["weights"]
            if "threshold" in _config:
                DEFAULT_THRESHOLD = _config["threshold"]
    except Exception:
        pass


@dataclass
class RiskResult:
    risk_score: float
    verdict: str  # "PASS" or "FLAG FOR REVIEW"
    validation_score: float
    tamper_score: float
    face_match_score: float
    weights: dict
    threshold: float


def compute_risk_score(
    validation_score: float,
    tamper_score: float,
    face_match_score: float,
    weights: dict | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> RiskResult:
    w = weights or DEFAULT_WEIGHTS

    risk = (
        w["validation"] * (100 - validation_score)
        + w["tamper"] * tamper_score
        + w["face"] * (100 - face_match_score)
    )
    risk = max(0.0, min(100.0, risk))

    verdict = "FLAG FOR REVIEW" if risk >= threshold else "PASS"

    return RiskResult(
        risk_score=risk,
        verdict=verdict,
        validation_score=validation_score,
        tamper_score=tamper_score,
        face_match_score=face_match_score,
        weights=w,
        threshold=threshold,
    )
