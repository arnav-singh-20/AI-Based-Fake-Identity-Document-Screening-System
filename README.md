# AI-Based Fake Identity & Document Screening System

**SIH26188** · Ministry of Home Affairs — Sashastra Seema Bal (Border Security)
Real-time forgery detection for passports, visas & IDs at border checkpoints.

A five-stage pipeline — **OCR → Validation → Tamper Detection → Face Verification → Risk Scoring** —
built entirely from pretrained models and classical computer-vision techniques.
Nothing is trained from scratch, so it's fully working and demoable out of the box.

---

## 1. Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt
```

### Tesseract system binary (only needed if you want the Tesseract fallback)

EasyOCR needs no system install — it downloads its own models on first run.
Tesseract needs the OS binary too:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# and add it to PATH
```

If you only install `easyocr` (skip `pytesseract`/tesseract), the app still runs fine —
the OCR module falls back automatically and just uses whichever engine is available.
Same logic applies to `deepface` vs `insightface` for face verification.

### First-run model downloads

EasyOCR, DeepFace, and InsightFace all download their pretrained weights the
**first time each is used** (a few hundred MB total). Do this once, ahead of
time, with an internet connection, well before your demo:

```bash
python -c "import easyocr; easyocr.Reader(['en'])"
python -c "from deepface import DeepFace; import numpy as np; DeepFace.build_model('Facenet512')"
```

---

## 2. Sample data

Place your document images under `sample_data/`:

```
sample_data/
  genuine/    <- unmodified MIDV-2020 documents
  forged/     <- matching FMIDV (pre-forged) versions
```

Pick ~10 clean pairs spanning 2–3 document types (passport, ID card,
driving license) — see the design doc's build roadmap (Section 6) for
the recommended selection process. These aren't wired into the UI
directly; upload them through the Streamlit file-uploader when you
run the app.

---

## 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a document image and a
traveler photo, tune the sidebar config if needed, and click
**Run Screening**.

---

## 4. Project structure

```
app.py                        Streamlit dashboard — wires all 5 stages together
modules/
  ocr_extraction.py           Step 2 — EasyOCR/Tesseract field + MRZ extraction
  document_validation.py      Step 3 — rule-based date/format/checksum checks
  tamper_detection.py         Step 4 — Error Level Analysis + heatmap
  face_verification.py        Step 5 — DeepFace/InsightFace similarity
  risk_scoring.py             Step 6 — weighted risk score + PASS/FLAG verdict
utils/
  mrz_utils.py                ICAO 9303 MRZ parsing + check-digit computation
  iso_codes.py                ISO alpha-3 country code lookup
sample_data/
  genuine/ , forged/          Put your MIDV-2020 / FMIDV pairs here
requirements.txt
.streamlit/config.toml        UI theme
```

Every module is standalone and independently testable — matches the
build order in the design doc's Section 6.1:

```python
from modules.ocr_extraction import extract_fields
import cv2

image = cv2.cvtColor(cv2.imread("sample_data/genuine/passport_01.jpg"), cv2.COLOR_BGR2RGB)
result = extract_fields(image)
print(result.name, result.document_number, result.mrz.all_checks_passed)
```

---

## 5. How each stage works

| Stage | Module | Technique |
|---|---|---|
| OCR Extraction | `ocr_extraction.py` | EasyOCR (primary) / Tesseract (fallback) — pretrained text recognition, plus MRZ line detection |
| Document Validation | `document_validation.py` | Pure rule-based: date logic, doc-number regex, ISO nationality check, MRZ checksum recomputation |
| Tampering Detection | `tamper_detection.py` | Error Level Analysis — recompress at fixed JPEG quality, diff against original, heatmap the anomalies |
| Face Verification | `face_verification.py` | DeepFace (Facenet512, primary) / InsightFace (fallback) — pretrained embeddings + similarity |
| Risk Scoring | `risk_scoring.py` | Weighted combination: `0.25×(100−Validation) + 0.35×Tamper + 0.40×(100−FaceMatch)`, threshold-based verdict |

All weights and the PASS/FLAG threshold are adjustable live from the
Streamlit sidebar — the design doc calls this out explicitly as a
parameter you should be ready to justify and tune in front of the jury.

---

## 6. Demo script (3–5 min)

1. Open with the problem: high checkpoint volume, slow/inconsistent manual checks.
2. Briefly show the architecture — five stages, all pretrained/classical.
3. Upload a **genuine** document + matching live photo live: fields populate,
   validation checks pass, ELA heatmap stays flat, face-match score is high → **PASS**.
4. Upload a **forged** document (from FMIDV) live: same walkthrough, but the ELA
   heatmap visibly lights up over the tampered region and/or the face-match score
   drops, pushing the risk score above threshold → **FLAGGED FOR REVIEW**.
5. Close with future scope: NFC/chip cross-verification, liveness detection,
   edge deployment at checkpoint kiosks, multilingual OCR.

---

## 7. Notes / known simplifications

- Field-extraction regex/keyword heuristics in `ocr_extraction.py` are
  intentionally simple and transparent (per the design doc's "zero-training,
  auditable" principle) — tune the keyword lists to whatever document
  templates you actually demo with.
- `document_validation.py` does not penalize a check it couldn't evaluate
  (e.g. a date field OCR failed to read) — it's excluded rather than
  counted as a failure, so a validation score reflects consistency among
  the fields actually extracted.
- ELA thresholds (`hotspot_threshold`, JPEG quality, amplification) are
  tunable from the sidebar; defaults are reasonable starting points but
  benchmark them against your actual genuine/forged sample pairs.
