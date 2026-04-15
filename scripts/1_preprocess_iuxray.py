"""
Indiana University X-Ray Dataset Preprocessing (Kaggle CSV Format)
Extracts labels, dual-view image paths, and metadata from CSV files
Adapted for: /kaggle/input/datasets/raddar/chest-xrays-indiana-university/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# CheXpert-style label extraction rules
LABEL_MAP = {
    'Atelectasis': ['atelectasis'],
    'Cardiomegaly': ['cardiomegaly', 'cardiac enlargement', 'enlarged heart'],
    'Consolidation': ['consolidation'],
    'Edema': ['edema', 'pulmonary edema'],
    'Effusion': ['effusion', 'pleural effusion'],
    'Emphysema': ['emphysema'],
    'Pneumonia': ['pneumonia'],
    'Pneumothorax': ['pneumothorax']
}

def extract_labels_from_text(text):
    """Extract binary labels from radiology report text using keyword matching"""
    if pd.isna(text):
        text = ""
    
    text_lower = str(text).lower()
    labels = {}
    
    for label_name, keywords in LABEL_MAP.items():
        # Check if any keyword is present
        found = any(keyword in text_lower for keyword in keywords)
        labels[label_name] = 1 if found else 0
    
    # If all labels are 0, mark as Normal
    labels['Normal'] = 1 if sum(labels.values()) == 0 else 0
    
    return labels

def parse_indiana_dataset(reports_csv, projections_csv, images_dir):
    """
    Parse Indiana dataset from CSV files
    
    Args:
        reports_csv: Path to indiana_reports.csv (contains findings and impressions)
        projections_csv: Path to indiana_projections.csv (contains image filenames and projections)
        images_dir: Path to images/ directory
    
    Returns:
        DataFrame with uid, frontal_file, lateral_file, report_text, labels
    """
    print("Loading CSV files...")
    reports_df = pd.read_csv(reports_csv)
    projections_df = pd.read_csv(projections_csv)
    
    print(f"Reports shape: {reports_df.shape}")
    print(f"Projections shape: {projections_df.shape}")
    print(f"\nReports columns: {list(reports_df.columns)}")
    print(f"Projections columns: {list(projections_df.columns)}")
    
    # Merge on patient/study ID (column name may vary - adapt as needed)
    # Common column names: 'uid', 'patient_id', 'study_id', 'filename'
    # We'll try to auto-detect
    
    # Group projections by patient to get frontal and lateral pairs
    images_path = Path(images_dir)
    data = []
    
    # Get unique patient IDs from reports
    # Assuming there's a UID column - adjust if needed
    if 'uid' in reports_df.columns:
        uid_col = 'uid'
    elif 'UID' in reports_df.columns:
        uid_col = 'UID'
    else:
        # Use first column as UID
        uid_col = reports_df.columns[0]
        print(f"Warning: Using '{uid_col}' as patient ID column")
    
    print(f"\nProcessing {len(reports_df)} patient reports...")
    
    for idx, report_row in reports_df.iterrows():
        uid = str(report_row[uid_col])
        
        # Extract report text
        findings = report_row.get('findings', '') or ''
        impression = report_row.get('impression', '') or ''
        
        # Some datasets use different column names
        if findings == '' and 'FINDINGS' in report_row:
            findings = report_row['FINDINGS'] or ''
        if impression == '' and 'IMPRESSION' in report_row:
            impression = report_row['IMPRESSION'] or ''
        
        full_report = f"Findings: {findings} Impression: {impression}"
        
        # Find corresponding images in projections CSV
        patient_images = projections_df[projections_df[uid_col] == uid]
        
        frontal_file = None
        lateral_file = None
        
        # Look for frontal and lateral projections
        for _, img_row in patient_images.iterrows():
            # Get filename - adapt column name as needed
            if 'filename' in img_row:
                img_filename = img_row['filename']
            elif 'image' in img_row:
                img_filename = img_row['image']
            else:
                img_filename = img_row[projections_df.columns[-1]]  # Last column as fallback
            
            img_path = images_path / img_filename
            
            # Get projection type
            if 'projection' in img_row:
                projection = str(img_row['projection']).upper()
            elif 'view' in img_row:
                projection = str(img_row['view']).upper()
            else:
                # Try to infer from filename
                img_filename_upper = str(img_filename).upper()
                if any(x in img_filename_upper for x in ['PA', 'AP', 'FRONTAL']):
                    projection = 'FRONTAL'
                elif any(x in img_filename_upper for x in ['LAT', 'LATERAL']):
                    projection = 'LATERAL'
                else:
                    projection = 'UNKNOWN'
            
            # Assign to frontal or lateral
            if img_path.exists():
                if any(x in projection for x in ['PA', 'AP', 'FRONTAL']):
                    if frontal_file is None:
                        frontal_file = str(img_path)
                elif 'LAT' in projection or 'LATERAL' in projection:
                    if lateral_file is None:
                        lateral_file = str(img_path)
        
        # Skip if no frontal image (frontal is mandatory)
        if frontal_file is None:
            continue
        
        # Extract labels from report
        labels = extract_labels_from_text(full_report)
        
        # Build row
        row = {
            'uid': uid,
            'frontal_file': frontal_file,
            'lateral_file': lateral_file if lateral_file else '',
            'report_text': full_report,
            **labels
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"\n✓ Extracted {len(df)} valid patient records")
    print(f"  Records with frontal view: {len(df)}")
    print(f"  Records with lateral view: {(df['lateral_file'] != '').sum()}")
    
    return df

def create_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Create stratified train/val/test splits"""
    
    # Use 'Normal' as stratification key
    stratify_col = df['Normal']
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        random_state=random_state,
        stratify=temp_df['Normal']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    return train_df, val_df, test_df

def compute_class_weights(train_df, label_columns):
    """Compute class weights for imbalanced multi-label classification"""
    weights = {}
    total_samples = len(train_df)
    
    for label in label_columns:
        pos_count = train_df[label].sum()
        neg_count = total_samples - pos_count
        
        if pos_count == 0:
            weight = 1.0
        else:
            # Weight = neg_count / pos_count, capped at 10x
            weight = min(neg_count / pos_count, 10.0)
        
        weights[label] = weight
    
    return weights

def main():
    # ===== CONFIGURE PATHS FOR KAGGLE =====
    REPORTS_CSV = "/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv"
    PROJECTIONS_CSV = "/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv"
    IMAGES_DIR = "/kaggle/input/chest-xrays-indiana-university/images"
    OUTPUT_DIR = "/kaggle/working"
    
    print("="*70)
    print("INDIANA UNIVERSITY CHEST X-RAY PREPROCESSING")
    print("="*70)
    
    # ===== PARSE DATASET =====
    df = parse_indiana_dataset(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR)
    
    if len(df) == 0:
        print("\n❌ Error: No valid records extracted!")
        print("Please check:")
        print("  1. CSV file paths are correct")
        print("  2. Column names match expected format")
        print("  3. Image files exist in images/ directory")
        return
    
    # ===== CREATE SPLITS =====
    train_df, val_df, test_df = create_splits(df)
    
    # ===== SAVE SPLITS =====
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    
    print(f"\n✓ Saved splits to {OUTPUT_DIR}/")
    
    # ===== COMPUTE CLASS WEIGHTS =====
    label_columns = list(LABEL_MAP.keys()) + ['Normal']
    weights = compute_class_weights(train_df, label_columns)
    
    # ===== SAVE STATISTICS =====
    with open(f"{OUTPUT_DIR}/label_stats.txt", 'w') as f:
        f.write("LABEL DISTRIBUTION (Training Set)\n")
        f.write("="*50 + "\n\n")
        
        for label in label_columns:
            count = train_df[label].sum()
            pct = (count / len(train_df)) * 100
            weight = weights[label]
            f.write(f"{label:15s}: {count:4d} samples ({pct:5.2f}%) | weight: {weight:.2f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"Total training samples: {len(train_df)}\n")
        f.write(f"Samples with lateral view: {(train_df['lateral_file'] != '').sum()}\n")
        f.write(f"Samples with both views: {((train_df['lateral_file'] != '') & (train_df['frontal_file'] != '')).sum()}\n")
    
    # Print summary to console
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION SUMMARY")
    print("="*70)
    for label in label_columns:
        count = train_df[label].sum()
        pct = (count / len(train_df)) * 100
        print(f"{label:15s}: {count:4d} samples ({pct:5.2f}%)")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - val.csv: {len(val_df)} samples")
    print(f"  - test.csv: {len(test_df)} samples")
    print(f"  - label_stats.txt")
    print(f"\nNext step: Run 2_extract_bert_features.py")

if __name__ == "__main__":
    main()
