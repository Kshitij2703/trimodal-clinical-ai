"""
Four-Layer Explainability: Grad-CAM + BERT Attention + Metadata + HDC Retrieval
Standalone version - includes model definition inline
Updated for EfficientNet-B0 backbone

FIXES:
  1. bottleneck_dim=256, hdc_dim=4096 — must match training config
  2. target_layer = model.vision_encoder[-1][0] — correct EfficientNet-B0 layer
  3. frontal[0].cpu().numpy() — correct single-channel image extraction
  4. BERT loaded ONCE in main(), passed into functions — no per-case reload
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pickle
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm

# ===== CONFIGURATION =====
CONFIG = {
    'batch_size': 16,
    'num_classes': 8,
    'vision_dim': 1280,              # EfficientNet-B0 output
    'text_dim': 768,
    'metadata_dim': 2,
    'vision_embed_dim': 256,
    'text_embed_dim': 256,
    'metadata_embed_dim': 16,
    'bottleneck_dim': 256,           # FIX 1: was 128, must match training
    'hdc_dim': 4096,                 # FIX 1: was 2048, must match training
    'dropout_vision': 0.5,
    'dropout_text': 0.4,
    'dropout_fusion': 0.5,
}

LABEL_COLUMNS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Pneumonia', 'Pneumothorax'
]

# ===== MODEL DEFINITION =====
class TriModalClinicalAI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        efficientnet        = models.efficientnet_b0(weights=None)
        self.vision_encoder = efficientnet.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)

        self.vision_proj = nn.Sequential(
            nn.Linear(config['vision_dim'], config['vision_embed_dim']),
            nn.BatchNorm1d(config['vision_embed_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_vision'])
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config['text_dim'], config['text_embed_dim']),
            nn.BatchNorm1d(config['text_embed_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_text'])
        )
        self.metadata_proj = nn.Sequential(
            nn.Linear(config['metadata_dim'], config['metadata_embed_dim']),
            nn.ReLU()
        )

        fusion_input_dim = (config['vision_embed_dim'] +
                            config['text_embed_dim'] +
                            config['metadata_embed_dim'])

        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_input_dim, config['bottleneck_dim']),
            nn.BatchNorm1d(config['bottleneck_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout_fusion'])
        )
        self.classifier  = nn.Linear(config['bottleneck_dim'], config['num_classes'])
        self.temperature = nn.Parameter(torch.ones(1))

    def _encode_vision(self, x):
        features = self.vision_encoder(x)
        pooled   = self.vision_pool(features)
        return pooled.flatten(1)

    def forward(self, frontal, lateral, text, metadata,
                return_bottleneck=False, use_temperature=False):
        frontal_feat   = self._encode_vision(frontal)
        lateral_feat   = self._encode_vision(lateral)
        vision_feat    = (frontal_feat + lateral_feat) / 2.0
        vision_embed   = self.vision_proj(vision_feat)
        text_embed     = self.text_proj(text)
        metadata_embed = self.metadata_proj(metadata)

        fused      = torch.cat([vision_embed, text_embed, metadata_embed], dim=1)
        bottleneck = self.fusion_bottleneck(fused)
        logits     = self.classifier(bottleneck)

        if use_temperature:
            logits = logits / self.temperature
        if return_bottleneck:
            return logits, bottleneck
        return logits

# ===== DATASET =====
class IUXRayDataset(Dataset):
    def __init__(self, csv_path, text_features_path, transform=None):
        self.df            = pd.read_csv(csv_path)
        self.text_features = np.load(text_features_path)
        self.transform     = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frontal_img  = Image.open(row['frontal_file']).convert('L')
        lateral_file = str(row['lateral_file']) if pd.notna(row['lateral_file']) else ''
        if lateral_file and lateral_file != 'nan' and os.path.exists(lateral_file):
            lateral_img = Image.open(lateral_file).convert('L')
        else:
            lateral_img = Image.open(row['frontal_file']).convert('L')

        if self.transform:
            frontal_img = self.transform(frontal_img)
            lateral_img = self.transform(lateral_img)

        text = torch.tensor(self.text_features[idx], dtype=torch.float32)

        metadata = torch.tensor([
            1.0 if 'frontal' in str(row.get('view_position', '')).lower() else 0.0,
            1.0 if 'lateral' in str(row.get('view_position', '')).lower() else 0.0
        ], dtype=torch.float32)

        labels = torch.tensor([row[col] for col in LABEL_COLUMNS], dtype=torch.float32)

        return {
            'frontal':  frontal_img,
            'lateral':  lateral_img,
            'text':     text,
            'metadata': metadata,
            'labels':   labels,
            'uid':      row['uid']
        }

# ===== LAYER 1: GRAD-CAM =====
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, gradients, activations):
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam     = torch.sum(weights * activations, dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def generate_dual_gradcam(model, frontal_img, lateral_img, text, metadata,
                           class_idx, device):
    """Generate Grad-CAM for both frontal and lateral views."""

    # FIX 2: correct EfficientNet-B0 target layer (last MBConv block's conv)
    target_layer = model.vision_encoder[-1][0]
    gradcam      = GradCAM(model, target_layer)

    model.eval()

    frontal_tensor  = frontal_img.unsqueeze(0).to(device).requires_grad_(True)
    lateral_tensor  = lateral_img.unsqueeze(0).to(device).requires_grad_(True)
    text_tensor     = text.unsqueeze(0).to(device)
    metadata_tensor = metadata.unsqueeze(0).to(device)

    # ── Frontal Grad-CAM ───────────────────────────────────────────────────
    model.zero_grad()
    with torch.enable_grad():
        logits   = model(frontal_tensor, lateral_tensor, text_tensor, metadata_tensor)
        pred_prob = torch.sigmoid(logits)[0, class_idx].item()
        logits[0, class_idx].backward(retain_graph=True)

    frontal_cam = gradcam.generate_cam(gradcam.gradients, gradcam.activations)
    frontal_cam = cv2.resize(frontal_cam, (224, 224))

    # ── Lateral Grad-CAM ───────────────────────────────────────────────────
    model.zero_grad()
    with torch.enable_grad():
        logits = model(frontal_tensor, lateral_tensor, text_tensor, metadata_tensor)
        logits[0, class_idx].backward()

    lateral_cam = gradcam.generate_cam(gradcam.gradients, gradcam.activations)
    lateral_cam = cv2.resize(lateral_cam, (224, 224))

    return frontal_cam, lateral_cam, pred_prob


# ===== LAYER 2: BERT ATTENTION =====
# FIX 4: tokenizer and bert_model passed in — loaded ONCE in main(), not per case
def extract_bert_attention(text, tokenizer, bert_model, device):
    """Extract token-level attention from ClinicalBERT."""
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=256, truncation=True, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Last layer, mean over heads, CLS token row
    attentions    = outputs.attentions[-1]
    cls_attention = attentions[0, :, 0, :].mean(dim=0)

    tokens           = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_weights = cls_attention.cpu().numpy()

    top_indices = np.argsort(attention_weights)[-15:][::-1]
    top_tokens  = [
        (tokens[i], float(attention_weights[i]))
        for i in top_indices
        if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    return top_tokens[:15]


# ===== LAYER 3: METADATA ATTRIBUTION =====
def format_metadata(metadata_tensor):
    metadata  = metadata_tensor.cpu().numpy()
    view_type = "Frontal" if metadata[0] > metadata[1] else "Lateral"
    return {
        'View Position':  view_type,
        'Metadata Vector': metadata.tolist()
    }


# ===== LAYER 4: HDC RETRIEVAL =====
def retrieve_similar_cases(query_embedding, hdc_index, top_k=3):
    query_hv              = np.sign(query_embedding @ hdc_index['projection_matrix'])
    query_hv[query_hv==0] = 1
    query_hv              = query_hv.astype(np.int8)

    distances    = np.sum(hdc_index['hypervectors'] != query_hv, axis=1)
    top_k_indices = np.argsort(distances)[:top_k]

    similar_cases = []
    for idx in top_k_indices:
        similarity = 1.0 - (distances[idx] / len(query_hv))
        similar_cases.append({
            'uid':              str(hdc_index['train_uids'][idx]),
            'labels':           hdc_index['train_labels'][idx].tolist(),
            'similarity':       round(float(similarity * 100), 2),
            'hamming_distance': int(distances[idx])
        })
    return similar_cases


# ===== GENERATE COMBINED EXPLANATION =====
def generate_explanation(model, dataset, idx, hdc_index, device, output_dir,
                          bert_tokenizer, bert_model_attn):
    """Generate all four layers of explainability for a single case."""

    sample   = dataset[idx]
    frontal  = sample['frontal']
    lateral  = sample['lateral']
    text     = sample['text']
    metadata = sample['metadata']
    labels   = sample['labels']
    uid      = sample['uid']

    # ── Forward pass ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits, bottleneck = model(
            frontal.unsqueeze(0).to(device),
            lateral.unsqueeze(0).to(device),
            text.unsqueeze(0).to(device),
            metadata.unsqueeze(0).to(device),
            return_bottleneck=True
        )
        predictions  = torch.sigmoid(logits)[0].cpu().numpy()
        bottleneck_np = bottleneck[0].cpu().numpy()

    case_dir = f"{output_dir}/{uid}"
    os.makedirs(case_dir, exist_ok=True)

    # ── LAYER 1: Dual-view Grad-CAM ───────────────────────────────────────
    print(f"  Generating Grad-CAM...")
    for class_idx, label_name in enumerate(LABEL_COLUMNS):
        if predictions[class_idx] > 0.3:
            try:
                frontal_cam, lateral_cam, prob = generate_dual_gradcam(
                    model, frontal, lateral, text, metadata, class_idx, device
                )

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # FIX 3: use [0] not squeeze() to get single 2-D channel
                frontal_img_np = frontal[0].cpu().numpy()
                axes[0].imshow(frontal_img_np, cmap='gray')
                axes[0].imshow(frontal_cam, cmap='jet', alpha=0.5)
                axes[0].set_title(f'Frontal — {label_name}: {prob:.2%}')
                axes[0].axis('off')

                lateral_img_np = lateral[0].cpu().numpy()
                axes[1].imshow(lateral_img_np, cmap='gray')
                axes[1].imshow(lateral_cam, cmap='jet', alpha=0.5)
                axes[1].set_title(f'Lateral — {label_name}: {prob:.2%}')
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(f"{case_dir}/gradcam_{label_name}.png",
                            dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"    Warning: Grad-CAM failed for {label_name}: {e}")

    # ── LAYER 2: BERT Attention ───────────────────────────────────────────
    print(f"  Extracting BERT attention...")
    top_tokens = []
    try:
        report_text = dataset.df.iloc[idx]['report_text']
        # FIX 4: pass pre-loaded tokenizer and model
        top_tokens = extract_bert_attention(
            report_text, bert_tokenizer, bert_model_attn, device
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        tokens  = [t[0] for t in top_tokens]
        weights = [t[1] for t in top_tokens]

        ax.barh(range(len(tokens)), weights, color='steelblue')
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Attention Weight')
        ax.set_title('ClinicalBERT Token Attention (Top 15)')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{case_dir}/bert_attention.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Warning: BERT attention failed: {e}")

    # ── LAYER 3: Metadata ─────────────────────────────────────────────────
    print(f"  Formatting metadata...")
    metadata_info = format_metadata(metadata)

    # ── LAYER 4: HDC Retrieval ────────────────────────────────────────────
    print(f"  Retrieving similar cases...")
    similar_cases = retrieve_similar_cases(bottleneck_np, hdc_index, top_k=3)

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {
        'uid':             str(uid),
        'predictions':     {LABEL_COLUMNS[i]: float(predictions[i])
                            for i in range(len(LABEL_COLUMNS))},
        'ground_truth':    {LABEL_COLUMNS[i]: int(labels[i])
                            for i in range(len(LABEL_COLUMNS))},
        'metadata':        metadata_info,
        'top_bert_tokens': top_tokens,
        'similar_cases':   similar_cases
    }

    with open(f"{case_dir}/explanation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Saved to {case_dir}/")
    return summary


# ===== MAIN =====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load tri-modal model ───────────────────────────────────────────────
    print("\nLoading model (EfficientNet-B0)...")
    model = TriModalClinicalAI(CONFIG).to(device)
    model.load_state_dict(
        torch.load('/kaggle/working/efficientnet_fused.pth',
                   map_location=device)
    )
    model.eval()
    print("  ✓ Model loaded")

    # ── Load HDC index ─────────────────────────────────────────────────────
    print("Loading HDC retrieval index...")
    with open('/kaggle/working/retrieval_index.pkl', 'rb') as f:
        hdc_index = pickle.load(f)
    print(f"  ✓ HDC index loaded "
          f"({len(hdc_index['hypervectors'])} vectors × "
          f"{hdc_index['hypervectors'].shape[1]} dims)")

    # ── Load ClinicalBERT ONCE ────────────────────────────────────────────
    # FIX 4: load here, not inside extract_bert_attention()
    print("Loading ClinicalBERT for attention extraction (once)...")
    bert_tokenizer  = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    bert_model_attn = AutoModel.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", output_attentions=True
    ).to(device)
    bert_model_attn.eval()
    print("  ✓ ClinicalBERT loaded")

    # ── Load test dataset ──────────────────────────────────────────────────
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = IUXRayDataset(
        '/kaggle/working/test.csv',
        '/kaggle/working/test_text_features.npy',
        transform=val_transform
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # ── Generate explanations ──────────────────────────────────────────────
    output_dir = '/kaggle/working/explanations'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING FOUR-LAYER EXPLAINABILITY")
    print("="*60)

    num_cases = min(10, len(test_dataset))
    for idx in range(num_cases):
        print(f"\nCase {idx+1}/{num_cases}  (dataset index {idx})")
        summary = generate_explanation(
            model, test_dataset, idx, hdc_index, device, output_dir,
            bert_tokenizer, bert_model_attn          # FIX 4: pass in
        )

        print(f"\n  Predictions (>0.3):")
        for label, prob in summary['predictions'].items():
            gt = summary['ground_truth'][label]
            if prob > 0.3:
                print(f"    {label:15s}: {prob:.2%}  "
                      f"(GT={'✓' if gt else '✗'})")

        if summary['top_bert_tokens']:
            print(f"\n  Top 5 BERT tokens:")
            for token, weight in summary['top_bert_tokens'][:5]:
                print(f"    {token}: {weight:.4f}")

        print(f"\n  Similar cases (HDC):")
        for i, case in enumerate(summary['similar_cases']):
            active = [LABEL_COLUMNS[j]
                      for j, v in enumerate(case['labels']) if v > 0]
            print(f"    #{i+1}: uid={case['uid']}  "
                  f"sim={case['similarity']:.1f}%  "
                  f"labels={active}")

    print("\n" + "="*60)
    print("EXPLAINABILITY GENERATION COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()