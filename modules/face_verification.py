"""
Step 5 — Face Verification

Compares the face from the document with the uploaded/live photo.

Uses DeepFace with lazy loading and CPU-friendly settings.
InsightFace is used as an optional fallback if available.

Outputs a Face-Match Score (0-100).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEEPFACE_AVAILABLE = None
_INSIGHTFACE_AVAILABLE = None


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


@dataclass
class FaceMatchResult:
    engine_used: str
    match_score: Optional[float] = None
    face_found_in_document: bool = False
    face_found_in_live_photo: bool = False
    raw_distance: Optional[float] = None
    error: Optional[str] = None


def _verify_deepface(
    doc_face: np.ndarray,
    live_face: np.ndarray
) -> FaceMatchResult:

    from deepface import DeepFace

    try:
        logger.info("Starting DeepFace verification...")

        result = DeepFace.verify(
            img1_path=doc_face,
            img2_path=live_face,

            # Lighter model than Facenet512
            model_name="VGG-Face",

            # Fastest/simple detector
            detector_backend="opencv",

            enforce_detection=False,

            # Explicitly disable optional features
            align=True,
            normalization="base",
        )

        distance = float(result.get("distance", 1.0))
        threshold = float(result.get("threshold", 0.4))

        verified = bool(result.get("verified", False))

        # Convert distance into a display score
        similarity = max(
            0.0,
            min(
                1.0,
                1.0 - (distance / max(threshold * 2, 0.0001))
            )
        )

        match_score = similarity * 100

        logger.info(
            "DeepFace verification completed. "
            "Distance: %.4f | Score: %.2f",
            distance,
            match_score
        )

        return FaceMatchResult(
            engine_used="DeepFace (VGG-Face)",
            match_score=float(match_score),
            face_found_in_document=True,
            face_found_in_live_photo=True,
            raw_distance=distance,
        )

    except Exception as e:

        logger.exception("DeepFace verification failed")

        return FaceMatchResult(
            engine_used="DeepFace (VGG-Face)",
            error=str(e),
        )


def _cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    a = a.flatten()
    b = b.flatten()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def _verify_insightface(
    doc_face: np.ndarray,
    live_face: np.ndarray
) -> FaceMatchResult:

    try:
        import cv2
        from insightface.app import FaceAnalysis

        logger.info("Starting InsightFace verification...")

        app = FaceAnalysis(
            providers=["CPUExecutionProvider"]
        )

        app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        doc_bgr = cv2.cvtColor(
            doc_face,
            cv2.COLOR_RGB2BGR
        )

        live_bgr = cv2.cvtColor(
            live_face,
            cv2.COLOR_RGB2BGR
        )

        doc_faces = app.get(doc_bgr)
        live_faces = app.get(live_bgr)

        if not doc_faces or not live_faces:

            return FaceMatchResult(
                engine_used="InsightFace",
                face_found_in_document=bool(doc_faces),
                face_found_in_live_photo=bool(live_faces),
                error="Face not detected in one or both images.",
            )

        doc = max(
            doc_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0])
            * (f.bbox[3] - f.bbox[1])
        )

        live = max(
            live_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0])
            * (f.bbox[3] - f.bbox[1])
        )

        similarity = _cosine_similarity(
            doc.normed_embedding,
            live.normed_embedding
        )

        match_score = float(
            np.clip(similarity * 100, 0, 100)
        )

        return FaceMatchResult(
            engine_used="InsightFace",
            match_score=match_score,
            face_found_in_document=True,
            face_found_in_live_photo=True,
            raw_distance=float(1 - similarity),
        )

    except Exception as e:

        logger.exception("InsightFace verification failed")

        return FaceMatchResult(
            engine_used="InsightFace",
            error=str(e),
        )


def verify_faces(
    doc_face: np.ndarray,
    live_face: np.ndarray,
    prefer_engine: str = "auto"
) -> FaceMatchResult:

    logger.info(
        "Running face verification using preference: %s",
        prefer_engine
    )

    if prefer_engine == "insightface":
        engines = ["insightface", "deepface"]

    elif prefer_engine == "deepface":
        engines = ["deepface", "insightface"]

    else:
        engines = ["deepface", "insightface"]

    last_error = None

    for engine in engines:

        try:

            if (
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

            elif (
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

        except Exception as e:

            logger.exception(
                "Face verification engine %s failed",
                engine
            )

            last_error = str(e)

    return FaceMatchResult(
        engine_used="none",
        error=last_error
        or "No face verification engine is available.",
    )