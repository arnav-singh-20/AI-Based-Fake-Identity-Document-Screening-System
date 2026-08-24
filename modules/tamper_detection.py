"""
Step 4 — Tampering Detection (Error Level Analysis + Copy-Move Detection)

Two complementary classical image forensics techniques, no ML:

1. **ELA (Error Level Analysis)**: Re-save the image at a fixed JPEG
   quality, diff it pixel-by-pixel against the original, and amplify.
   Digitally edited regions compress differently from untouched regions.
   Uses **localized block-based scoring** so small tampered areas aren't
   diluted by a large clean background.

2. **Copy-Move Detection**: Uses ORB keypoint matching to find duplicated
   regions within the same image — a telltale sign of clone-stamp edits
   (e.g. blanking out an MRZ by copying a nearby background patch).

Outputs a Tamper Score (0-100) plus a heatmap image for the dashboard.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageChops
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TamperResult:
    tamper_score: float                # 0-100, higher = more likely tampered
    heatmap_overlay: np.ndarray         # RGB image: original + heatmap overlay
    heatmap_raw: np.ndarray             # grayscale ELA diff map
    hotspot_fraction: float             # fraction of pixels flagged as anomalous
    mean_diff: float
    max_diff: float
    ela_score: float = 0.0              # ELA sub-score (0-100)
    copy_move_score: float = 0.0        # Copy-move sub-score (0-100)
    copy_move_matches: int = 0          # number of suspicious match clusters found


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return Image.fromarray(image)


def compute_ela(image: np.ndarray, jpeg_quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """
    Re-compress `image` (RGB numpy array) as JPEG at `jpeg_quality`,
    reload it, and return the per-pixel absolute difference from the
    original, amplified by `scale` for visibility.
    """
    pil_img = _to_pil(image).convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(pil_img, recompressed)
    diff_arr = np.asarray(diff).astype(np.float32)

    amplified = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)
    return amplified


def _localized_ela_score(gray_ela: np.ndarray, block_size: int = 64) -> float:
    """
    Instead of averaging over the entire image (which dilutes small tampered
    areas), divide into blocks and use the **95th-percentile block mean**.
    This highlights localized anomalies even in mostly-clean images.
    """
    h, w = gray_ela.shape
    if h < block_size or w < block_size:
        return float(np.mean(gray_ela))

    block_means = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray_ela[i:i + block_size, j:j + block_size]
            block_means.append(float(np.mean(block)))

    if not block_means:
        return float(np.mean(gray_ela))

    block_means.sort()
    # Use 95th percentile of block means — catches localized hot regions
    idx_95 = int(len(block_means) * 0.95)
    top_blocks = block_means[idx_95:]
    return float(np.mean(top_blocks)) if top_blocks else block_means[-1]


def _detect_copy_move(image: np.ndarray, min_matches: int = 8, distance_threshold: float = 30.0) -> tuple[float, int, np.ndarray]:
    """
    ORB keypoint-based copy-move forgery detection.

    Detects duplicated regions within the same image by finding keypoint
    matches that are spatially displaced (i.e. not matching themselves).
    Returns (score 0-100, number of suspicious clusters, visualization mask).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    # Resize for speed if the image is very large
    max_dim = 1024
    h, w = gray.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)))

    # ORB feature detection
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < min_matches * 2:
        return 0.0, 0, np.zeros((h, w), dtype=np.uint8)

    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=5)

    # Filter: keep matches where the descriptors are similar but the
    # keypoints are spatially far apart (not self-matches)
    suspicious_pairs = []
    min_spatial_dist = 20.0 / scale_factor  # minimum pixel displacement

    for match_group in matches:
        for m in match_group:
            if m.queryIdx == m.trainIdx:
                continue  # skip self-match
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            spatial_dist = np.linalg.norm(pt1 - pt2)
            if spatial_dist > min_spatial_dist and m.distance < distance_threshold:
                suspicious_pairs.append((m.queryIdx, m.trainIdx, pt1, pt2))

    # Build a mask highlighting suspicious regions
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    for _, _, pt1, pt2 in suspicious_pairs:
        cv2.circle(mask, (int(pt1[0]), int(pt1[1])), 8, 255, -1)
        cv2.circle(mask, (int(pt2[0]), int(pt2[1])), 8, 255, -1)

    if scale_factor < 1.0:
        mask = cv2.resize(mask, (w, h))

    num_clusters = len(suspicious_pairs) // 2  # rough cluster count
    # Score: saturates around 40+ suspicious pairs
    score = min(100.0, (len(suspicious_pairs) / 40.0) * 100.0)

    return float(score), num_clusters, mask


def _build_heatmap_overlay(original: np.ndarray, ela_map: np.ndarray, copy_move_mask: Optional[np.ndarray] = None, alpha: float = 0.55) -> np.ndarray:
    """Overlay a JET colormap of the (grayscale) ELA intensity onto the original image."""
    gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY) if ela_map.ndim == 3 else ela_map
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Merge copy-move mask into ELA heatmap if available
    if copy_move_mask is not None:
        cm_resized = cv2.resize(copy_move_mask, (gray.shape[1], gray.shape[0])) if copy_move_mask.shape[:2] != gray.shape[:2] else copy_move_mask
        gray_blurred = np.maximum(gray_blurred, cm_resized)

    heat_color = cv2.applyColorMap(gray_blurred, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    original_resized = cv2.resize(original, (gray.shape[1], gray.shape[0])) if original.shape[:2] != gray.shape[:2] else original
    if original_resized.ndim == 2:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)

    mask = (gray_blurred.astype(np.float32) / 255.0) ** 0.6
    mask = mask[..., None]
    overlay = (original_resized.astype(np.float32) * (1 - alpha * mask) +
               heat_color.astype(np.float32) * (alpha * mask))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def detect_tampering(
    image: np.ndarray,
    jpeg_quality: int = 90,
    scale: float = 15.0,
    hotspot_threshold: int = 60,
) -> TamperResult:
    """
    Run ELA + copy-move detection on `image` and derive a 0-100 tamper
    score from the combined signals.
    """
    ela_map = compute_ela(image, jpeg_quality=jpeg_quality, scale=scale)
    gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY) if ela_map.ndim == 3 else ela_map

    mean_diff = float(np.mean(gray))
    max_diff = float(np.max(gray))
    hotspot_fraction = float(np.mean(gray > hotspot_threshold))

    # --- Localized ELA score (block-based) ---
    localized_intensity = _localized_ela_score(gray)
    spread_component = min(hotspot_fraction * 400, 100)
    intensity_component = min((localized_intensity / 60.0) * 100, 100)
    ela_score = float(np.clip(0.65 * spread_component + 0.35 * intensity_component, 0, 100))

    # --- Copy-move detection ---
    try:
        cm_score, cm_clusters, cm_mask = _detect_copy_move(image)
    except Exception as e:
        logger.warning("Copy-move detection failed: %s", e)
        cm_score, cm_clusters, cm_mask = 0.0, 0, np.zeros(gray.shape[:2], dtype=np.uint8)

    # --- Combined tamper score ---
    # ELA and copy-move are complementary; take the max so either signal
    # alone can flag a document, weighted slightly toward ELA.
    tamper_score = float(np.clip(max(ela_score, 0.4 * ela_score + 0.6 * cm_score), 0, 100))

    overlay = _build_heatmap_overlay(image, ela_map, cm_mask)

    return TamperResult(
        tamper_score=tamper_score,
        heatmap_overlay=overlay,
        heatmap_raw=gray,
        hotspot_fraction=hotspot_fraction,
        mean_diff=mean_diff,
        max_diff=max_diff,
        ela_score=ela_score,
        copy_move_score=cm_score,
        copy_move_matches=cm_clusters,
    )
