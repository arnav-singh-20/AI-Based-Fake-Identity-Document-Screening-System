"""
Step 5 — Face Verification

Compares the face extracted from the identity document with a
live/uploaded traveler photo.

Uses InsightFace as the primary engine because it is significantly
lighter and more reliable than DeepFace/TensorFlow on CPU-based
deployment environments such as Streamlit Cloud.

DeepFace is kept as an optional fallback.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_INSIGHTFACE_AVAILABLE = None
_INSIGHTFACE_APP = None
_DEEPFACE_AVAILABLE = None


def _insightface_available() -> bool:
    global _INSIGHTFACE_AVAILABLE

    if _INSIGHTFACE_AVAILABLE is None:
        try:
            import insightface  # noqa: F401
            _INSIGHTFACE_AVAILABLE = True
        except Exception as e:
            logger.warning("InsightFace unavailable: %s", e)
            _INSIGHTFACE_AVAILABLE = False

    return _INSIGHTFACE_AVAILABLE


def _deepface_available() -> bool:
    global _DEEPFACE_AVAILABLE

    if _DEEPFACE_AVAILABLE is None:
        try:
            from deepface import DeepFace  # noqa: F401
            _DEEPFACE_AVAILABLE = True
        except Exception as e:
            logger.warning("DeepFace unavailable: %s", e)
            _DEEPFACE_AVAILABLE = False

    return _DEEPFACE_AVAILABLE


def _get_insightface_app():
    global _INSIGHTFACE_APP

    if _INSIGHTFACE_APP is None:
        from insightface.app import FaceAnalysis

        logger.info("Loading InsightFace model...")

        app = FaceAnalysis(
            providers=["CPUExecutionProvider"]
        )

        # IMPORTANT:
        # -1 means CPU mode.
        # Using 0 incorrectly requests GPU context.
        app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        _INSIGHTFACE_APP = app

    return _INSIGHTFACE_APP


@dataclass
class FaceMatchResult:
    engine_used: str
    match_score: Optional[float] = None
    face_found_in_document: bool = False
    face_found_in_live_photo: bool = False
    raw_distance: Optional[float] = None
    error: Optional[str] = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def _largest_face(faces):
    """
    Return the face with the largest bounding-box area.
    """

    return max(
        faces,
        key=lambda face: (
            (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1])
        )
    )


def _verify_insightface(
    doc_face: np.ndarray,
    live_face: np.ndarray
) -> FaceMatchResult:

    import cv2

    app = _get_insightface_app()

    # Convert RGB → BGR because OpenCV / InsightFace expects BGR.
    doc_bgr = cv2.cvtColor(
        doc_face,
        cv2.COLOR_RGB2BGR
    )

    live_bgr = cv2.cvtColor(
        live_face,
        cv2.COLOR_RGB2BGR
    )

    logger.info("Detecting face in document...")

    doc_faces = app.get(doc_bgr)

    logger.info("Detecting face in live photo...")

    live_faces = app.get(live_bgr)

    if not doc_faces or not live_faces:

        return FaceMatchResult(
            engine_used="InsightFace",
            face_found_in_document=bool(doc_faces),
            face_found_in_live_photo=bool(live_faces),
            error="Face not detected in one or both images."
        )

    # Select the largest face if multiple faces are detected.
    doc_detected_face = _largest_face(doc_faces)
    live_detected_face = _largest_face(live_faces)

    doc_embedding = doc_detected_face.normed_embedding
    live_embedding = live_detected_face.normed_embedding

    cosine_similarity = _cosine_similarity(
        doc_embedding,
        live_embedding
    )

    # Cosine similarity is approximately -1 to 1.
    # Convert it to a 0–100 score.
    match_score = float(
        np.clip(
            (cosine_similarity + 1) / 2 * 100,
            0,
            100
        )
    )

    return FaceMatchResult(
        engine_used="InsightFace",
        match_score=match_score,
        face_found_in_document=True,
        face_found_in_live_photo=True,
        raw_distance=float(1 - cosine_similarity)
    )


def _verify_deepface(
    doc_face: np.ndarray,
    live_face: np.ndarray
) -> FaceMatchResult:

    from deepface import DeepFace

    try:

        result = DeepFace.verify(
            img1_path=doc_face,
            img2_path=live_face,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )

        distance = float(
            result.get("distance", 1.0)
        )

        threshold = float(
            result.get("threshold", 0.4)
        )

        similarity = max(
            0.0,
            1.0 - distance / (threshold * 2)
        )

        match_score = float(
            np.clip(
                similarity * 100,
                0,
                100
            )
        )

        return FaceMatchResult(
            engine_used="DeepFace (Facenet512)",
            match_score=match_score,
            face_found_in_document=True,
            face_found_in_live_photo=True,
            raw_distance=distance
        )

    except Exception as e:

        return FaceMatchResult(
            engine_used="DeepFace (Facenet512)",
            error=str(e)
        )


def verify_faces(
    doc_face: np.ndarray,
    live_face: np.ndarray,
    prefer_engine: str = "auto"
) -> FaceMatchResult:
    """
    Compare the face on the document against the uploaded/live photo.

    InsightFace is preferred by default because it performs much better
    on CPU-only cloud deployments.
    """

    # IMPORTANT:
    # InsightFace is now tried FIRST.

    if prefer_engine == "deepface":

        order = ["deepface", "insightface"]

    else:

        order = ["insightface", "deepface"]

    last_error = None

    for engine in order:

        try:

            if (
                engine == "insightface"
                and _insightface_available()
            ):

                logger.info(
                    "Running face verification using InsightFace..."
                )

                result = _verify_insightface(
                    doc_face,
                    live_face
                )

                if result.match_score is not None:
                    return result

                last_error = result.error

            elif (
                engine == "deepface"
                and _deepface_available()
            ):

                logger.info(
                    "Running face verification using DeepFace..."
                )

                result = _verify_deepface(
                    doc_face,
                    live_face
                )

                if result.match_score is not None:
                    return result

                last_error = result.error

        except Exception as e:

            logger.exception(
                "Face verification engine %s failed",
                engine
            )

            last_error = str(e)

    return FaceMatchResult(
        engine_used="none",
        error=(
            last_error
            or "No face verification engine available."
        )
    )