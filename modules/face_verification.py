"""
Step 5 — Face Verification

Extracts a face embedding from the photo printed on the document and
from a separate live/uploaded photo of the traveler, then computes a
similarity score. Both DeepFace and InsightFace are pretrained
face-recognition toolkits — no training required.

Tries DeepFace first (simpler pip install, good default accuracy);
falls back to InsightFace if DeepFace isn't installed or fails.

Outputs a Face-Match Score (0-100).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEEPFACE_AVAILABLE = None
_INSIGHTFACE_AVAILABLE = None
_INSIGHTFACE_APP = None


def _deepface_available() -> bool:
    global _DEEPFACE_AVAILABLE
    if _DEEPFACE_AVAILABLE is None:
        try:
            from deepface import DeepFace  # noqa: F401
            _DEEPFACE_AVAILABLE = True
        except Exception:
            _DEEPFACE_AVAILABLE = False
    return _DEEPFACE_AVAILABLE


def _insightface_available() -> bool:
    global _INSIGHTFACE_AVAILABLE
    if _INSIGHTFACE_AVAILABLE is None:
        try:
            import insightface  # noqa: F401
            _INSIGHTFACE_AVAILABLE = True
        except Exception:
            _INSIGHTFACE_AVAILABLE = False
    return _INSIGHTFACE_AVAILABLE


def _get_insightface_app():
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHTFACE_APP = app
    return _INSIGHTFACE_APP


@dataclass
class FaceMatchResult:
    engine_used: str
    match_score: Optional[float] = None   # 0-100, higher = more likely same person
    face_found_in_document: bool = False
    face_found_in_live_photo: bool = False
    raw_distance: Optional[float] = None
    error: Optional[str] = None


def _verify_deepface(doc_face: np.ndarray, live_face: np.ndarray) -> FaceMatchResult:
    from deepface import DeepFace

    try:
        res = DeepFace.verify(
            img1_path=doc_face,
            img2_path=live_face,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )
        distance = float(res.get("distance", 1.0))
        threshold = float(res.get("threshold", 0.4)) or 0.4
        # Convert distance -> a 0-100 "match score" where 0 distance = 100,
        # and the model's own verification threshold maps to ~50.
        similarity = max(0.0, 1.0 - (distance / (threshold * 2)))
        match_score = float(np.clip(similarity * 100, 0, 100))
        return FaceMatchResult(
            engine_used="DeepFace (Facenet512)",
            match_score=match_score,
            face_found_in_document=True,
            face_found_in_live_photo=True,
            raw_distance=distance,
        )
    except ValueError as e:
        # DeepFace raises ValueError when a face can't be detected in
        # one of the two images.
        msg = str(e).lower()
        doc_ok = "img1" not in msg
        live_ok = "img2" not in msg
        return FaceMatchResult(
            engine_used="DeepFace (Facenet512)",
            face_found_in_document=doc_ok,
            face_found_in_live_photo=live_ok,
            error=str(e),
        )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _verify_insightface(doc_face: np.ndarray, live_face: np.ndarray) -> FaceMatchResult:
    import cv2

    app = _get_insightface_app()

    doc_bgr = cv2.cvtColor(doc_face, cv2.COLOR_RGB2BGR)
    live_bgr = cv2.cvtColor(live_face, cv2.COLOR_RGB2BGR)

    doc_faces = app.get(doc_bgr)
    live_faces = app.get(live_bgr)

    if not doc_faces or not live_faces:
        return FaceMatchResult(
            engine_used="InsightFace",
            face_found_in_document=bool(doc_faces),
            face_found_in_live_photo=bool(live_faces),
            error="Face not detected in one or both images.",
        )

    # Use the largest detected face in each image.
    doc_emb = max(doc_faces, key=lambda f: f.bbox[2] * f.bbox[3]).normed_embedding
    live_emb = max(live_faces, key=lambda f: f.bbox[2] * f.bbox[3]).normed_embedding

    cos_sim = _cosine_similarity(doc_emb, live_emb)  # roughly -1..1
    match_score = float(np.clip((cos_sim + 1) / 2 * 100, 0, 100))

    return FaceMatchResult(
        engine_used="InsightFace",
        match_score=match_score,
        face_found_in_document=True,
        face_found_in_live_photo=True,
        raw_distance=1 - cos_sim,
    )


def _pad_to_min_size(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    if h >= target_size and w >= target_size:
        return img
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def verify_faces(doc_face: np.ndarray, live_face: np.ndarray, prefer_engine: str = "auto") -> FaceMatchResult:
    """
    Compare the face on the document (`doc_face`) against the
    live/uploaded traveler photo (`live_face`). Both are RGB numpy
    arrays. `prefer_engine`: "deepface", "insightface", or "auto".
    """
    # Pad images to minimum size to ensure face detection succeeds on small crops
    doc_face_padded = _pad_to_min_size(doc_face)
    live_face_padded = _pad_to_min_size(live_face)

    order = []
    if prefer_engine == "insightface":
        order = ["insightface", "deepface"]
    elif prefer_engine == "deepface":
        order = ["deepface", "insightface"]
    else:
        order = ["deepface", "insightface"]

    last_error = None
    for engine in order:
        try:
            if engine == "deepface" and _deepface_available():
                result = _verify_deepface(doc_face_padded, live_face_padded)
                if result.match_score is not None:
                    return result
                last_error = result.error
                continue
            if engine == "insightface" and _insightface_available():
                result = _verify_insightface(doc_face_padded, live_face_padded)
                if result.match_score is not None:
                    return result
                last_error = result.error
                continue
        except Exception as e:  # noqa: BLE001
            logger.warning("Face engine %s failed: %s", engine, e)
            last_error = str(e)
            continue

    return FaceMatchResult(
        engine_used="none",
        error=last_error or "No face-verification engine available. Install deepface or insightface.",
    )
