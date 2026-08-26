"""
Step 5 — Face Verification

Lightweight face verification designed for Streamlit Cloud.

Uses OpenCV Haar Cascade to detect faces and compares normalized
face regions using histogram correlation and structural similarity.

This implementation avoids DeepFace, TensorFlow, InsightFace,
and large model downloads.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceMatchResult:
    engine_used: str

    match_score: Optional[float] = None

    face_found_in_document: bool = False

    face_found_in_live_photo: bool = False

    raw_distance: Optional[float] = None

    error: Optional[str] = None


def _get_face_detector():
    """
    Load OpenCV's built-in Haar Cascade face detector.
    """

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Unable to load OpenCV face detector."
        )

    return detector


def _extract_largest_face(image: np.ndarray):
    """
    Detect and return the largest face from an RGB image.

    Returns:
        face_image, face_found
    """

    if image is None or image.size == 0:
        return None, False

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    detector = _get_face_detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None, False

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    face = gray[y:y + h, x:x + w]

    return face, True


def _normalize_face(face: np.ndarray) -> np.ndarray:
    """
    Resize and normalize a face for comparison.
    """

    face = cv2.resize(
        face,
        (128, 128)
    )

    face = cv2.equalizeHist(face)

    return face


def _compare_faces(
    face1: np.ndarray,
    face2: np.ndarray
):
    """
    Compare two detected face regions.

    Uses:
    1. Histogram correlation
    2. Mean Squared Error
    3. Normalized pixel correlation

    Returns a score between 0 and 100.
    """

    face1 = _normalize_face(face1)
    face2 = _normalize_face(face2)

    # Histogram comparison
    hist1 = cv2.calcHist(
        [face1],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist2 = cv2.calcHist(
        [face2],
        [0],
        None,
        [256],
        [0, 256]
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    histogram_similarity = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_CORREL
    )

    # Convert -1...1 to 0...100
    histogram_score = (
        (histogram_similarity + 1) / 2
    ) * 100

    # Pixel correlation
    correlation = np.corrcoef(
        face1.flatten(),
        face2.flatten()
    )[0, 1]

    if np.isnan(correlation):
        correlation = 0

    correlation_score = (
        (correlation + 1) / 2
    ) * 100

    # Mean squared error
    mse = np.mean(
        (
            face1.astype("float")
            - face2.astype("float")
        ) ** 2
    )

    # Convert MSE to similarity
    mse_score = max(
        0,
        100 - (mse / 65025) * 100
    )

    # Weighted final score
    final_score = (
        0.45 * histogram_score
        + 0.45 * correlation_score
        + 0.10 * mse_score
    )

    final_score = float(
        np.clip(
            final_score,
            0,
            100
        )
    )

    raw_distance = float(
        1 - (final_score / 100)
    )

    return final_score, raw_distance


def verify_faces(
    doc_face: np.ndarray,
    live_face: np.ndarray,
    prefer_engine: str = "auto"
) -> FaceMatchResult:
    """
    Compare the face from the identity document with
    the uploaded/live photo.

    Lightweight OpenCV implementation suitable for
    Streamlit Cloud.
    """

    try:

        logger.info(
            "Starting lightweight face verification..."
        )

        document_face, document_found = (
            _extract_largest_face(doc_face)
        )

        live_detected_face, live_found = (
            _extract_largest_face(live_face)
        )

        # Face detection failed
        if not document_found or not live_found:

            missing = []

            if not document_found:
                missing.append(
                    "document image"
                )

            if not live_found:
                missing.append(
                    "live/uploaded photo"
                )

            return FaceMatchResult(

                engine_used="OpenCV Haar Cascade",

                face_found_in_document=
                    document_found,

                face_found_in_live_photo=
                    live_found,

                error=(
                    "Face not detected in: "
                    + ", ".join(missing)
                )
            )

        # Compare detected faces
        score, distance = _compare_faces(
            document_face,
            live_detected_face
        )

        logger.info(
            "Face verification completed. "
            "Score: %.2f",
            score
        )

        return FaceMatchResult(

            engine_used="OpenCV Haar Cascade + Feature Comparison",

            match_score=score,

            face_found_in_document=True,

            face_found_in_live_photo=True,

            raw_distance=distance
        )

    except Exception as e:

        logger.exception(
            "Face verification failed"
        )

        return FaceMatchResult(

            engine_used="OpenCV",

            error=str(e)
        )