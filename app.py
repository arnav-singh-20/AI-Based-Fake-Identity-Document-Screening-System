import time

import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

from modules.ocr_extraction import extract_fields
from modules.document_validation import validate_document
from modules.tamper_detection import detect_tampering
from modules.face_verification import verify_faces
from modules.risk_scoring import compute_risk_score, DEFAULT_WEIGHTS, DEFAULT_THRESHOLD
from scripts.evaluate_pipeline import run_batch_evaluation, compute_evaluation_metrics
from utils.screening_history import log_screening, get_history, clear_history

st.set_page_config(
    page_title="Document Screening System",
    page_icon="🛂",
    layout="wide",
)


def load_image(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def verdict_badge(verdict: str, risk_score: float):
    if verdict == "PASS":
        st.success(f"**PASS** — Risk Score: {risk_score:.1f} / 100")
    else:
        st.error(f"🚩 **FLAGGED FOR REVIEW** — Risk Score: {risk_score:.1f} / 100")


def score_bar(label: str, value: float, help_text: str = ""):
    st.metric(label, f"{value:.1f} / 100", help=help_text)
    st.progress(min(max(value / 100, 0.0), 1.0))

with st.sidebar:
    st.title("Screening Config")
    st.caption("AI-Based Fake Identity & Document Screening System")

    st.subheader("Engines")
    ocr_engine = st.selectbox("OCR engine", ["auto (EasyOCR → Tesseract)", "easyocr", "tesseract"], index=0)
    face_engine = st.selectbox("Face verification engine",["Lightweight OpenCV"],index=0)

    st.subheader("Risk Scoring Weights")
    st.caption("Face-match and tamper detection are weighted more heavily — they're harder to fake convincingly than field validation.")
    w_val = st.slider("Validation weight", 0.0, 1.0, DEFAULT_WEIGHTS["validation"], 0.05)
    w_tamper = st.slider("Tamper weight", 0.0, 1.0, DEFAULT_WEIGHTS["tamper"], 0.05)
    w_face = st.slider("Face-match weight", 0.0, 1.0, DEFAULT_WEIGHTS["face"], 0.05)
    total_w = w_val + w_tamper + w_face
    if total_w > 0:
        weights = {"validation": w_val / total_w, "tamper": w_tamper / total_w, "face": w_face / total_w}
    else:
        weights = DEFAULT_WEIGHTS
    st.caption(f"Normalized: validation={weights['validation']:.2f}, tamper={weights['tamper']:.2f}, face={weights['face']:.2f}")

    threshold = st.slider("FLAG threshold", 0, 100, int(DEFAULT_THRESHOLD), 1,
                           help="Risk score at or above this is flagged for human review.")

    st.subheader("ELA Parameters")
    jpeg_quality = st.slider("Re-compression JPEG quality", 50, 99, 90)
    ela_scale = st.slider("ELA amplification", 1.0, 30.0, 15.0, 0.5)

    st.divider()

_ocr_pref = {"auto (EasyOCR → Tesseract)": "auto", "easyocr": "easyocr", "tesseract": "tesseract"}[ocr_engine]
_face_pref = "opencv"

# Main UI
st.title("AI-Based Fake Identity & Document Screening System")
st.caption("Real-time forgery detection for passports, visas & IDs at border checkpoints · Ministry of Home Affairs — Sashastra Seema Bal")

tab1, tab2 = st.tabs(["Single Document Screening", "Screening History"])

with tab1:
    input_mode = st.radio(
        "Choose input method",
        ["Upload Files", "Use Camera"],
        horizontal=True,
        help="Upload images from disk, or scan documents and capture face photos live using your device camera.",
    )

    doc_file = None
    face_file = None

    if input_mode == "Upload Files":
        col_upload1, col_upload2 = st.columns(2)
        with col_upload1:
            doc_file = st.file_uploader("Upload travel document (passport / visa / ID)", type=["jpg", "jpeg", "png"])
            if doc_file:
                st.image(doc_file, caption="Uploaded document")

        with col_upload2:
            face_file = st.file_uploader("Upload traveler's photo", type=["jpg", "jpeg", "png"])
            if face_file:
                st.image(face_file, caption="Traveler photo")

    else: 
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.markdown("#### Scan Travel Document")
            st.caption("Hold the passport / visa / ID card in front of your camera and take a photo.")
            doc_file = st.camera_input("Capture document image", key="cam_doc")

        with col_cam2:
            st.markdown("#### Capture Traveler's Face")
            st.caption("Position the traveler in front of the camera and take a photo.")
            face_file = st.camera_input("Capture traveler face", key="cam_face")

    run = st.button("🔍 Run Screening", type="primary", use_container_width=True, disabled=not (doc_file and face_file))

    if not (doc_file and face_file):
        st.info("Provide both a document image and a traveler photo (via upload or camera) to run the pipeline.")


    # Pipeline execution
    if run:
        doc_image = load_image(doc_file)
        face_image = load_image(face_file)

        progress = st.progress(0, text="Starting pipeline...")

        # Step 2 — OCR
        progress.progress(10, text="Step 2/6 — Running OCR extraction...")
        t0 = time.time()
        ocr_result = extract_fields(doc_image, prefer_engine=_ocr_pref)
        ocr_time = time.time() - t0

        # Step 3 — Validation
        progress.progress(35, text="Step 3/6 — Validating extracted fields...")
        t0 = time.time()
        validation_result = validate_document(ocr_result)
        val_time = time.time() - t0

        # Step 4 — Tamper detection
        progress.progress(55, text="Step 4/6 — Running Error Level Analysis...")
        t0 = time.time()
        tamper_result = detect_tampering(doc_image, jpeg_quality=jpeg_quality, scale=ela_scale)
        ela_time = time.time() - t0

        # Step 5 — Face verification
        progress.progress(80, text="Step 5/6 — Verifying face match...")
        t0 = time.time()
        face_result = verify_faces(doc_image, face_image, prefer_engine=_face_pref)
        face_time = time.time() - t0

        # Step 6 — Risk scoring
        progress.progress(95, text="Step 6/6 — Computing combined risk score...")
        t0 = time.time()
        face_score_for_risk = face_result.match_score if face_result.match_score is not None else 0.0
        risk_result = compute_risk_score(
            validation_score=validation_result.score,
            tamper_score=tamper_result.tamper_score,
            face_match_score=face_score_for_risk,
            weights=weights,
            threshold=threshold,
        )
        scoring_time = time.time() - t0
        
        progress.progress(100, text="Done.")
        time.sleep(0.5)
        progress.empty()

        # Log to screening history
        log_screening(
            filename=doc_file.name if hasattr(doc_file, 'name') else 'camera_capture',
            validation_score=validation_result.score,
            tamper_score=tamper_result.tamper_score,
            face_match_score=face_score_for_risk,
            risk_score=risk_result.risk_score,
            verdict=risk_result.verdict,
            notes=f"OCR: {ocr_result.engine_used}, Face: {face_result.engine_used}",
            ela_score=getattr(tamper_result, 'ela_score', tamper_result.tamper_score),
            copy_move_score=getattr(tamper_result, 'copy_move_score', 0.0),
            weights=weights,
            threshold=threshold,
        )

        st.divider()
        st.header("Verdict")
        verdict_badge(risk_result.verdict, risk_result.risk_score)

        verdict_cols = st.columns(3)
        with verdict_cols[0]:
            score_bar("Document Validation", validation_result.score, "% of rule-based checks passed")
        with verdict_cols[1]:
            score_bar("Tamper Score", tamper_result.tamper_score, "Higher = more likely digitally edited (ELA)")
        with verdict_cols[2]:
            if face_result.match_score is not None:
                score_bar("Face-Match Score", face_result.match_score, "Higher = more likely same person")
            else:
                st.metric("Face-Match Score", "N/A")
                st.caption(f"{face_result.error}")

        st.divider()

        # Extracted fields 
        st.header("Extracted Fields")
        st.caption(f"OCR engine used: **{ocr_result.engine_used}** · {ocr_time:.2f}s")
        if ocr_result.error:
            st.warning(ocr_result.error)
        else:
            fcols = st.columns(3)
            fields = [
                ("Name", ocr_result.name),
                ("Date of Birth", ocr_result.date_of_birth),
                ("Document Number", ocr_result.document_number),
                ("Nationality", ocr_result.nationality),
                ("Issue Date", ocr_result.issue_date),
                ("Expiry Date", ocr_result.expiry_date),
            ]
            for i, (label, value) in enumerate(fields):
                with fcols[i % 3]:
                    st.metric(label, value or "—")

            if ocr_result.mrz and ocr_result.mrz.found:
                st.success(f"MRZ detected ({ocr_result.mrz.format} format)")
                with st.expander("Raw MRZ lines"):
                    for line in ocr_result.mrz.raw_lines:
                        st.code(line)
            else:
                st.info("No MRZ block detected on this document.")

            with st.expander("Raw OCR text"):
                st.text(ocr_result.full_text)

        st.divider()

        #  Validation checks
        st.header("Document Validation Checks")
        for check in validation_result.checks:
            icon = "✅" if check.passed else "❌"
            st.write(f"{icon} **{check.name}** — {check.detail}")

        st.divider()

        #  Tamper heatmap 
        st.header("Tampering Detection (Error Level Analysis)")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.image(doc_image, caption="Original document", use_container_width=True)
        with tcol2:
            st.image(tamper_result.heatmap_overlay, caption="ELA tamper heatmap overlay", use_container_width=True)
        st.caption(
            f"Hotspot fraction: {tamper_result.hotspot_fraction*100:.1f}% of pixels · "
            f"Mean diff: {tamper_result.mean_diff:.1f} · Max diff: {tamper_result.max_diff:.1f} · "
            f"Computed in {ela_time:.2f}s"
        )
        if tamper_result.tamper_score >= 50:
            st.error("Significant localized anomalies detected — consistent with digital editing.")
        elif tamper_result.tamper_score >= 25:
            st.warning("Some anomalies detected — review recommended.")
        else:
            st.success("Diff map is mostly flat/uniform — no strong tampering signal.")

        st.divider()

        #  Face verification 
        st.header("Face Verification")
        st.caption(f"Engine used: **{face_result.engine_used}** · {face_time:.2f}s")
        fvcol1, fvcol2 = st.columns(2)
        with fvcol1:
            st.image(doc_image, caption="Document photo", use_container_width=True)
        with fvcol2:
            st.image(face_image, caption="Live / uploaded photo", use_container_width=True)

        if face_result.match_score is not None:
            if face_result.match_score >= 70:
                st.success(f"High similarity ({face_result.match_score:.1f}/100) — likely same person.")
            elif face_result.match_score >= 40:
                st.warning(f"Moderate similarity ({face_result.match_score:.1f}/100) — review recommended.")
            else:
                st.error(f"Low similarity ({face_result.match_score:.1f}/100) — possible photo-swap or impersonation.")
        else:
            st.error(f"Face verification could not complete: {face_result.error}")
            st.write(f"Face detected in document photo: {face_result.face_found_in_document}")
            st.write(f"Face detected in live photo: {face_result.face_found_in_live_photo}")

        st.divider()

        #  Risk breakdown 
        st.header("Risk Score Breakdown")
        st.latex(
            r"\text{Risk} = %.2f \times (100 - \text{Validation}) + %.2f \times \text{Tamper} + %.2f \times (100 - \text{FaceMatch})"
            % (weights["validation"], weights["tamper"], weights["face"])
        )
        breakdown_cols = st.columns(4)
        breakdown_cols[0].metric("Validation contribution", f"{weights['validation'] * (100 - validation_result.score):.1f}")
        breakdown_cols[1].metric("Tamper contribution", f"{weights['tamper'] * tamper_result.tamper_score:.1f}")
        breakdown_cols[2].metric("Face-match contribution", f"{weights['face'] * (100 - face_score_for_risk):.1f}")
        breakdown_cols[3].metric("Total Risk Score", f"{risk_result.risk_score:.1f}", delta=f"Threshold: {threshold}")

    else:
        st.markdown("""
        
        """)

with tab2:
    st.header("Screening History")
    st.markdown("A persistent audit log of all past screenings, stored in a local SQLite database.")

    history = get_history(limit=200)

    if history:
        st.caption(f"Showing {len(history)} most recent screening(s).")
        df_hist = pd.DataFrame(history)
        # Reorder columns for readability
        display_cols = ["id", "timestamp", "filename", "verdict", "risk_score",
                        "validation_score", "tamper_score", "face_match_score",
                        "ela_score", "copy_move_score", "notes", "ground_truth"]
        display_cols = [c for c in display_cols if c in df_hist.columns]
        df_hist = df_hist[display_cols]

        # Color-code verdict
        st.dataframe(df_hist, use_container_width=True)

        # Summary stats
        st.subheader("Summary")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Total Screenings", len(history))
        flagged = sum(1 for r in history if r.get("verdict") == "FLAG FOR REVIEW")
        passed = sum(1 for r in history if r.get("verdict") == "PASS")
        s_col2.metric("Passed", passed)
        s_col3.metric("Flagged", flagged)
        s_col4.metric("Flag Rate", f"{flagged / len(history) * 100:.1f}%" if history else "0%")

        if st.button("🗑️ Clear History", type="secondary"):
            clear_history()
            st.rerun()
    else:
        st.info("No screening history yet. Run a screening from the first tab to start building the audit log.")
