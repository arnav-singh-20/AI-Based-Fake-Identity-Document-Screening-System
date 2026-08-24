import os
import sys
import json
import argparse
import shutil
import numpy as np
import cv2
from PIL import Image

try:
    import datasets
except ImportError:
    print("Error: 'datasets' package is required. Install it using 'pip install datasets'")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare genuine and forged sample documents.")
    parser.add_argument("dataset_name", nargs="?", default="AmAFakePerson123/TrialforGeneratedIDs",
                        help="Hugging Face dataset name (default: AmAFakePerson123/TrialforGeneratedIDs)")
    parser.add_argument("--image-col", help="Column containing images")
    parser.add_argument("--label-col", help="Column containing labels")
    parser.add_argument("--fake-values", help="Comma-separated values in label-col representing forged documents")
    return parser.parse_args()

def inspect_dataset_schema(ds):
    print("=" * 60)
    print("HUGGING FACE DATASET SCHEMA INSPECTION")
    print("=" * 60)
    print(f"Dataset type: {type(ds)}")
    print(f"Features: {ds.features}")
    print(f"Number of rows: {len(ds)}")
    print("-" * 60)
    
    # Get a sample row
    sample = ds[0]
    print("Sample row keys:", list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, Image.Image):
            print(f"  Column '{k}': PIL Image (mode={v.mode}, size={v.size})")
        else:
            # Print truncated value
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  Column '{k}': {type(v).__name__} = {val_str}")
    print("=" * 60)

def detect_columns(ds, requested_image_col=None, requested_label_col=None):
    features = ds.features
    
    # Auto-detect image column
    image_col = requested_image_col
    if not image_col:
        for col, feat in features.items():
            if isinstance(feat, datasets.Image):
                image_col = col
                break
        if not image_col:
            # fallback: look for PIL image in first row
            sample = ds[0]
            for col, val in sample.items():
                if isinstance(val, Image.Image):
                    image_col = col
                    break
    
    # Auto-detect label column
    label_col = requested_label_col
    if not label_col:
        for col, feat in features.items():
            if col != image_col and isinstance(feat, (datasets.ClassLabel, datasets.Value)):
                label_col = col
                break
        if not label_col:
            for col in features.keys():
                if col != image_col:
                    label_col = col
                    break
                    
    return image_col, label_col

def load_annotations(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("_via_img_metadata", {})

def get_region_by_field(regions, field_name):
    for r in regions:
        region_attrs = r.get("region_attributes", {})
        if region_attrs.get("field_name") == field_name:
            return r
    return None

def main():
    args = parse_args()
    
    print(f"Loading Hugging Face dataset '{args.dataset_name}'...")
    try:
        ds = datasets.load_dataset(args.dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to local data directly.")
        ds = None
        
    if ds is not None:
        inspect_dataset_schema(ds)
        image_col, label_col = detect_columns(ds, args.image_col, args.label_col)
        print(f"Auto-detected Columns: Image Column = '{image_col}', Label Column = '{label_col}'")
        
        # Analyze labels
        if label_col in ds.features:
            feat = ds.features[label_col]
            if isinstance(feat, datasets.ClassLabel):
                print(f"Label names: {feat.names}")
            else:
                unique_labels = set(ds[label_col])
                print(f"Unique label values in dataset: {unique_labels}")
        else:
            print(f"Label column '{label_col}' not found in features.")
            
        print("\nNote: The dataset contains 10 document types. None of them represent forged status directly.")
        print("To create a meaningful demo, we will use the local files in the 'dataset' directory")
        print("to copy genuine documents and programmatically generate forged counterparts.")
        print("-" * 60)
        
    # Setup directories
    genuine_dir = "sample_data/genuine"
    forged_dir = "sample_data/forged"
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)
    
    local_dataset_path = "dataset"
    if not os.path.exists(local_dataset_path):
        print(f"Error: Local 'dataset' directory not found at '{local_dataset_path}'.")
        sys.exit(1)
        
    # Document types to use for the demo
    doc_types = ["alb_id", "aze_passport", "esp_id"]
    
    # We will pick 4 alb_id, 4 aze_passport, and 3 esp_id images (total 11)
    selection = {
        "alb_id": ["00.jpg", "01.jpg", "02.jpg", "03.jpg"],
        "aze_passport": ["00.jpg", "01.jpg", "02.jpg", "03.jpg"],
        "esp_id": ["00.jpg", "01.jpg", "02.jpg"]
    }
    
    print("Generating genuine and forged sample document pairs...")
    
    # To perform face swap, we need a pool of source faces. Let's pre-load faces from other documents.
    # We can use the '05.jpg' images of the same document types as face donors!
    face_donors = {}
    for doc_type in doc_types:
        donor_img_path = f"dataset/scan_upright/images/{doc_type}/05.jpg"
        donor_json_path = f"dataset/scan_upright/annotations/{doc_type}.json"
        
        if os.path.exists(donor_img_path) and os.path.exists(donor_json_path):
            donor_meta = load_annotations(donor_json_path)
            # Find metadata for 05.jpg
            for key, val in donor_meta.items():
                if val.get("filename") == "05.jpg":
                    regions = val.get("regions", [])
                    face_region = get_region_by_field(regions, "face")
                    if face_region:
                        shape = face_region.get("shape_attributes", {})
                        if shape.get("name") == "rect":
                            img = cv2.imread(donor_img_path)
                            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
                            face_donors[doc_type] = img[y:y+h, x:x+w]
                            break
                            
    if not face_donors:
        print("Warning: Could not load any face donor images. Using fallback crop.")
        
    # Now generate the genuine and forged documents
    generated_count = 0
    for doc_type in doc_types:
        img_dir = f"dataset/scan_upright/images/{doc_type}"
        json_path = f"dataset/scan_upright/annotations/{doc_type}.json"
        
        if not os.path.exists(img_dir) or not os.path.exists(json_path):
            print(f"Skipping {doc_type} - image directory or annotations missing.")
            continue
            
        metadata = load_annotations(json_path)
        
        for filename in selection[doc_type]:
            src_path = os.path.join(img_dir, filename)
            if not os.path.exists(src_path):
                print(f"File not found: {src_path}")
                continue
                
            # Define output names
            gen_out_name = f"{doc_type}_{filename}"
            gen_out_path = os.path.join(genuine_dir, gen_out_name)
            forged_out_path = os.path.join(forged_dir, gen_out_name)
            
            # 1. Save genuine image
            shutil.copy(src_path, gen_out_path)
            
            # 2. Generate forged image
            img = cv2.imread(src_path)
            forged_img = img.copy()
            
            # Find regions for this image in metadata
            regions = []
            for key, val in metadata.items():
                if val.get("filename") == filename:
                    regions = val.get("regions", [])
                    break
                    
            face_region = get_region_by_field(regions, "face")
            doc_quad_region = get_region_by_field(regions, "doc_quad")
            
            # Perform face swap if face box is available
            if face_region:
                shape = face_region.get("shape_attributes", {})
                if shape.get("name") == "rect":
                    tx, ty, tw, th = shape["x"], shape["y"], shape["width"], shape["height"]
                    
                    # Choose a donor face from a different document type
                    other_types = [t for t in face_donors.keys() if t != doc_type]
                    donor_type = other_types[0] if other_types else None
                    if donor_type and donor_type in face_donors:
                        donor_face = face_donors[donor_type]
                        # Resize donor face to fit target bounding box
                        resized_face = cv2.resize(donor_face, (tw, th))
                        forged_img[ty:ty+th, tx:tx+tw] = resized_face
                        # Write some ELA-triggering pixel alterations on the boundary
                        cv2.rectangle(forged_img, (tx, ty), (tx+tw, ty+th), (0, 0, 255), 1)
                        print(f"  Swapped face on {doc_type}_{filename} with face from {donor_type}")
            
            # Perform MRZ alteration if doc_quad is available
            if doc_quad_region:
                shape = doc_quad_region.get("shape_attributes", {})
                if shape.get("name") == "polygon":
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if xs and ys:
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        card_h = max_y - min_y
                        
                        # MRZ is in the bottom 12% of the card bounding box
                        mrz_y_start = int(max_y - 0.12 * card_h)
                        mrz_y_end = max_y
                        mrz_x_start = min_x
                        mrz_x_end = max_x
                        
                        # Draw a solid white rectangle to erase/cover the MRZ characters
                        cv2.rectangle(forged_img, (mrz_x_start, mrz_y_start), (mrz_x_end, mrz_y_end), (255, 255, 255), -1)
                        print(f"  Erased MRZ on {doc_type}_{filename}")
            else:
                # Fallback: cover bottom 10% of the entire image if doc_quad is missing
                h_img, w_img = img.shape[:2]
                cv2.rectangle(forged_img, (0, int(h_img * 0.9)), (w_img, h_img), (255, 255, 255), -1)
                print(f"  Erased bottom of image on {doc_type}_{filename} (fallback)")
                
            # Save forged image
            cv2.imwrite(forged_out_path, forged_img)
            generated_count += 1
            
    print(f"\nSuccessfully generated {generated_count} genuine and forged document pairs.")
    print(f"Genuine saved in: {genuine_dir}")
    print(f"Forged saved in: {forged_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
