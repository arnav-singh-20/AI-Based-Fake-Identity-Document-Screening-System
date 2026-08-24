import os
import sys
import csv
import json
import logging
import pandas as pd
import numpy as np
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import project modules
sys.path.append(os.getcwd())
from modules.ocr_extraction import extract_fields
from modules.document_validation import validate_document
from modules.tamper_detection import detect_tampering
from modules.face_verification import verify_faces
from modules.risk_scoring import compute_risk_score, DEFAULT_WEIGHTS, DEFAULT_THRESHOLD

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    # Convert BGR (OpenCV) to RGB (which the modules expect)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading config.json: {e}")
    return {}

def run_pipeline_on_image(img_path_or_arr, gt_label="unknown", config=None):
    """
    Runs the full 5-stage pipeline on a single image.
    Accepts either an absolute file path or an RGB numpy array.
    """
    if config is None:
        config = load_config()
        
    weights = config.get("weights", DEFAULT_WEIGHTS)
    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    engines = config.get("engines", {})
    ela_config = config.get("ela", {})
    
    ocr_pref = engines.get("ocr", "auto")
    face_pref = engines.get("face", "auto")
    
    jpeg_quality = ela_config.get("jpeg_quality", 90)
    ela_scale = ela_config.get("scale", 15.0)
    hotspot_threshold = ela_config.get("hotspot_threshold", 60)
    
    notes = ["Self face verification baseline"]
    
    # Default values in case of failures
    ocr_engine = "N/A"
    val_score = 0.0
    tamper_score = 0.0
    face_score = 0.0
    risk_score = 0.0
    verdict = "FLAG FOR REVIEW"
    
    # 1. Load / Read Image
    if isinstance(img_path_or_arr, str):
        try:
            img = load_image(img_path_or_arr)
        except Exception as e:
            logger.error(f"Failed to load image {img_path_or_arr}: {e}")
            notes.append(f"Image load failed: {e}")
            return {
                "filename": os.path.basename(img_path_or_arr),
                "ground_truth": gt_label,
                "validation_score": val_score,
                "tamper_score": tamper_score,
                "face_match_score": face_score,
                "risk_score": risk_score,
                "verdict": verdict,
                "notes": "; ".join(notes)
            }
    else:
        img = img_path_or_arr
        
    filename = os.path.basename(img_path_or_arr) if isinstance(img_path_or_arr, str) else "uploaded_doc"
    
    # 2. OCR Extraction
    ocr_res = None
    try:
        ocr_res = extract_fields(img, prefer_engine=ocr_pref)
        ocr_engine = ocr_res.engine_used
        if ocr_res.error:
            notes.append(f"OCR error: {ocr_res.error}")
        if not ocr_res.mrz or not ocr_res.mrz.found:
            notes.append("No MRZ found")
    except Exception as e:
        logger.error(f"OCR stage crashed for {filename}: {e}")
        notes.append(f"OCR crashed: {e}")
        
    # 3. Document Validation
    if ocr_res:
        try:
            val_res = validate_document(ocr_res)
            val_score = val_res.score
        except Exception as e:
            logger.error(f"Validation stage crashed for {filename}: {e}")
            notes.append(f"Validation crashed: {e}")
    else:
        notes.append("Validation skipped (no OCR)")
        
    # 4. Tamper Detection (ELA)
    try:
        tamper_res = detect_tampering(img, jpeg_quality=jpeg_quality, scale=ela_scale, hotspot_threshold=hotspot_threshold)
        tamper_score = tamper_res.tamper_score
    except Exception as e:
        logger.error(f"Tamper stage crashed for {filename}: {e}")
        notes.append(f"Tamper detection crashed: {e}")
        
    # 5. Face Verification (Self-verification)
    try:
        face_res = verify_faces(img, img, prefer_engine=face_pref)
        if face_res.match_score is not None:
            face_score = face_res.match_score
        else:
            face_score = 0.0
            notes.append(f"Face match failed: {face_res.error or 'No face detected'}")
    except Exception as e:
        logger.error(f"Face verification crashed for {filename}: {e}")
        notes.append(f"Face verification crashed: {e}")
        
    # 6. Risk Scoring
    try:
        risk_res = compute_risk_score(val_score, tamper_score, face_score, weights=weights, threshold=threshold)
        risk_score = risk_res.risk_score
        verdict = risk_res.verdict
    except Exception as e:
        logger.error(f"Risk scoring crashed for {filename}: {e}")
        notes.append(f"Risk scoring crashed: {e}")
        
    return {
        "filename": filename,
        "ground_truth": gt_label,
        "validation_score": val_score,
        "tamper_score": tamper_score,
        "face_match_score": face_score,
        "risk_score": risk_score,
        "verdict": verdict,
        "notes": "; ".join(notes)
    }

def run_batch_evaluation(genuine_dir="sample_data/genuine", forged_dir="sample_data/forged", config=None):
    """
    Runs the pipeline on all files in the genuine and forged folders.
    """
    if config is None:
        config = load_config()
        
    results = []
    folders = [("genuine", genuine_dir), ("forged", forged_dir)]
    
    for gt_label, folder_path in folders:
        if not os.path.exists(folder_path):
            logger.warning(f"Directory {folder_path} not found.")
            continue
        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        for fn in filenames:
            path = os.path.join(folder_path, fn)
            res = run_pipeline_on_image(path, gt_label=gt_label, config=config)
            results.append(res)
            
    return results

def compute_evaluation_metrics(results, config=None):
    """
    Computes accuracy, precision, recall, F1 and score distributions from results.
    """
    if config is None:
        config = load_config()
        
    weights = config.get("weights", DEFAULT_WEIGHTS)
    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    
    df = pd.DataFrame(results)
    if len(df) == 0:
        return {}
        
    df["gt_positive"] = df["ground_truth"] == "forged"
    df["pred_positive"] = df["verdict"] == "FLAG FOR REVIEW"
    
    tp = sum(df["gt_positive"] & df["pred_positive"])
    fp = sum(~df["gt_positive"] & df["pred_positive"])
    tn = sum(~df["gt_positive"] & ~df["pred_positive"])
    fn = sum(df["gt_positive"] & ~df["pred_positive"])
    
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    distributions = {}
    for gt in ["genuine", "forged"]:
        subset = df[df["ground_truth"] == gt]
        if len(subset) > 0:
            distributions[gt] = {
                "validation": {"mean": subset["validation_score"].mean(), "std": subset["validation_score"].std()},
                "tamper": {"mean": subset["tamper_score"].mean(), "std": subset["tamper_score"].std()},
                "face": {"mean": subset["face_match_score"].mean(), "std": subset["face_match_score"].std()},
                "risk": {"mean": subset["risk_score"].mean(), "std": subset["risk_score"].std()}
            }
            
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "distributions": distributions,
        "weights": weights,
        "threshold": threshold
    }

def run_grid_search(df):
    """
    Run grid search optimization for weights and threshold.
    """
    best_f1 = 0.0
    best_weights = DEFAULT_WEIGHTS
    best_threshold = DEFAULT_THRESHOLD
    
    df["gt_positive"] = df["ground_truth"] == "forged"
    
    # Grid search ranges (weights sum to 1.0)
    for w_val in np.arange(0.0, 1.01, 0.05):
        for w_tamper in np.arange(0.0, 1.01 - w_val, 0.05):
            w_face = 1.0 - w_val - w_tamper
            if w_face < 0.0:
                continue
            
            temp_risk = (
                w_val * (100 - df["validation_score"]) +
                w_tamper * df["tamper_score"] +
                w_face * (100 - df["face_match_score"])
            )
            
            for th in np.arange(10, 91, 1.0):
                pred = temp_risk >= th
                gt = df["gt_positive"]
                
                t_tp = sum(gt & pred)
                t_fp = sum(~gt & pred)
                t_fn = sum(gt & ~pred)
                
                t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
                t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
                t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
                
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    best_weights = {"validation": w_val, "tamper": w_tamper, "face": w_face}
                    best_threshold = th
                    
    return best_weights, best_threshold, best_f1

def evaluate_pipeline():
    config = load_config()
    results = run_batch_evaluation(config=config)
    
    # Save to CSV
    csv_path = "evaluation_results.csv"
    keys = ["filename", "ground_truth", "validation_score", "tamper_score", "face_match_score", "risk_score", "verdict", "notes"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE. Results saved to {csv_path}")
    print("=" * 80)
    
    metrics = compute_evaluation_metrics(results, config=config)
    
    print("\nMetrics summary:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
    print("-" * 80)
    
    cm = metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"  True Positives (Forged correctly flagged):   {cm['tp']}")
    print(f"  False Positives (Genuine falsely flagged):  {cm['fp']}")
    print(f"  True Negatives (Genuine correctly passed):  {cm['tn']}")
    print(f"  False Negatives (Forged falsely passed):   {cm['fn']}")
    print("=" * 80)
    
    print("\nScore Distributions (Mean ± Std Dev):")
    for gt, dist in metrics["distributions"].items():
        print(f"  Ground Truth: {gt.upper()}")
        print(f"    Validation Score: {dist['validation']['mean']:.2f} ± {dist['validation']['std']:.2f}")
        print(f"    Tamper Score:     {dist['tamper']['mean']:.2f} ± {dist['tamper']['std']:.2f}")
        print(f"    Face Match Score: {dist['face']['mean']:.2f} ± {dist['face']['std']:.2f}")
        print(f"    Final Risk Score: {dist['risk']['mean']:.2f} ± {dist['risk']['std']:.2f}")
        print("-" * 50)
        
    df = pd.DataFrame(results)
    best_weights, best_threshold, best_f1 = run_grid_search(df)
    
    print("\nOptimization Check:")
    print(f"  Current Weights: validation={metrics['weights']['validation']:.2f}, tamper={metrics['weights']['tamper']:.2f}, face={metrics['weights']['face']:.2f}")
    print(f"  Current Threshold: {metrics['threshold']}")
    
    print(f"\nGrid Search Recommendation:")
    if best_f1 > metrics["f1"]:
        print(f"  Optimal Weights: validation={best_weights['validation']:.2f}, tamper={best_weights['tamper']:.2f}, face={best_weights['face']:.2f}")
        print(f"  Optimal Threshold: {best_threshold:.1f}")
        print(f"  Expected F1-Score: {best_f1*100:.1f}%")
        print("\nACTION REQUIRED: Update config.json with these values if they differ.")
    else:
        print("  Current weights and threshold are already optimal.")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_pipeline()
