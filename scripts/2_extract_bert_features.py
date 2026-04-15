"""
Extract ClinicalBERT text embeddings offline (run once, save forever)
Outputs: train_text_features.npy, val_text_features.npy, test_text_features.npy
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def extract_bert_features(csv_path, tokenizer, model, device, max_length=256):
    """
    Extract [CLS] embeddings from ClinicalBERT for all reports in CSV
    Returns: numpy array of shape (N, 768)
    """
    df = pd.read_csv(csv_path)
    features = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
            text = row['report_text']
            
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Extract [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_embedding[0])
    
    return np.array(features)

def main():
    # ===== CONFIG =====
    CSV_DIR = "/kaggle/working"  # Where train.csv, val.csv, test.csv are
    OUTPUT_DIR = "/kaggle/working"
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== LOAD MODEL =====
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded ✓")
    
    # ===== EXTRACT FEATURES =====
    for split in ['train', 'val', 'test']:
        csv_path = f"{CSV_DIR}/{split}.csv"
        output_path = f"{OUTPUT_DIR}/{split}_text_features.npy"
        
        print(f"\n{'='*60}")
        print(f"Extracting {split} features...")
        print(f"{'='*60}")
        
        features = extract_bert_features(csv_path, tokenizer, model, device, MAX_LENGTH)
        
        # Save
        np.save(output_path, features)
        
        print(f"✓ Saved {features.shape} to {output_path}")
        print(f"  Size: {features.nbytes / 1e6:.2f} MB")
    
    print("\n" + "="*60)
    print("BERT feature extraction complete!")
    print("These .npy files are permanent — never need to regenerate")
    print("="*60)

if __name__ == "__main__":
    main()
