"""
Step 5 — Face Verification

Compares the face extracted from the document with the uploaded/live
traveler photo.

Uses DeepFace with a lightweight model and lazy initialization.

Designed to run on CPU-only environments such as Streamlit Cloud.

Outputs a Face-Match Score (0-100).
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reduce TensorFlow logs and unnecessary GPU initialization noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger(__name__)

_DEEPFACE_AVAILABLE = None


@dataclass
class FaceMatchResult:
    engine_used: str

    match_score: Optional[float] = None

    face_found_in_document: bool = False

    face_found_in_live_photo: bool = False

    raw_distance: Optional[float] = None

    error: Optional[str] = None


def _deepface_available() -> bool:
    """
    Check whether DeepFace is installed.
    """

    global _DEEPFACE_AVAILABLE

    if _DEEPFACE_AVAILABLE is None:

        try:
            from deepface import DeepFace  # noqa: F401

            _DEEPFACE_AVAILABLE = True

        except Exception as e:

            logger.warning(
                "DeepFace unavailable: %s",
                e
            )

            _DEEPFACE_AVAILABLE = False

    return _DEEPFACE_AVAILABLE


def _calculate_score(
    distance: float,
    threshold: float
) -> float:
    """
    Convert DeepFace distance into a 0-100 similarity score.

    Lower distance means higher similarity.
    """

    if threshold <= 0:
        threshold = 0.4

    score = 1.0 - (
        distance / (threshold * 2)
    )

    score = np.clip(
        score * 100,
        0,
        100
    )

    return float(score)


def _verify_deepface(
    doc_face: np.ndarray,
    live_face: np.ndarray
) -> FaceMatchResult:
    """
    Verify two faces using DeepFace.

    GhostFaceNet is used because it is lighter than Facenet512
    and better suited for limited-memory deployments.
    """

    try:

        from deepface import DeepFace

        logger.info(
            "Starting DeepFace verification using GhostFaceNet..."
        )

        result = DeepFace.verify(

            img1_path=doc_face,

            img2_path=live_face,

            model_name="GhostFaceNet",

            detector_backend="opencv",

            enforce_detection=False,

            align=True,

            silent=True
        )

        distance = float(
            result.get(
                "distance",
                1.0
            )
        )

        threshold = float(
            result.get(
                "threshold",
                0.4
            )
        )

        verified = bool(
            result.get(
                "verified",
                False
            )
        )

        match_score = _calculate_score(
            distance,
            threshold
        )

        logger.info(
            "Face verification completed | "
            "Distance: %.4f | "
            "Threshold: %.4f | "
            "Score: %.2f | "
            "Verified: %s",
            distance,
            threshold,
            match_score,
            verified
        )

        return FaceMatchResult(

            engine_used="DeepFace (GhostFaceNet)",

            match_score=match_score,

            face_found_in_document=True,

            face_found_in_live_photo=True,

            raw_distance=distance
        )

    except Exception as e:

        logger.exception(
            "DeepFace verification failed"
        )

        return FaceMatchResult(

            engine_used="DeepFace (GhostFaceNet)",

            error=str(e)
        )


def verify_faces(
    doc_face: np.ndarray,
    live_face: np.ndarray,
    prefer_engine: str = "auto"
) -> FaceMatchResult:
    """
    Compare the document face with the uploaded/live photo.

    Parameters
    ----------
    doc_face:
        RGB numpy image containing the face from the document.

    live_face:
        RGB numpy image containing the uploaded/live face.

    prefer_engine:
        Kept for compatibility with the rest of the pipeline.

    Returns
    -------
    FaceMatchResult
    """

    logger.info(
        "Running face verification..."
    )

    if doc_face is None:

        return FaceMatchResult(

            engine_used="none",

            error="Document face image is missing."
        )

    if live_face is None:

        return FaceMatchResult(

            engine_used="none",

            error="Live/uploaded face image is missing."
        )

    if not isinstance(
        doc_face,
        np.ndarray
    ):

        return FaceMatchResult(

            engine_used="none",

            error="Invalid document face format."
        )

    if not isinstance(
        live_face,
        np.ndarray
    ):

        return FaceMatchResult(

            engine_used="none",

            error="Invalid live face format."
        )

    if not _deepface_available():

        return FaceMatchResult(

            engine_used="none",

            error=(
                "DeepFace is unavailable. "
                "Please check the requirements.txt file."
            )
        )

    return _verify_deepface(
        doc_face,
        live_face
    )