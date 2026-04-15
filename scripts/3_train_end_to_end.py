"""
TRI-MODAL NEUROSYMBOLIC CLINICAL AI — Training v4 Final

ARCHITECTURE:
  - EfficientNet-B0 vision backbone (1280-dim, lightweight)
  - ClinicalBERT text features (768-dim, pre-extracted)
  - Metadata encoder (2-dim view position)
  - Fusion bottleneck → 8-class multi-label classifier

FIXES IN v4 Final:
  1. CUDA_VISIBLE_DEVICES=0 → forces GPU 0, not GPU 1
  2. Image cache as uint8 numpy arrays (NOT PIL objects)
     - PIL objects: ~500KB each → OOM crash
     - uint8 arrays: ~50KB each → ~400MB total, safe
     - Resize done ONCE at cache time, not every epoch
  3. TOKENIZERS_PARALLELISM=false → no fork warning bottleneck
  4. num_workers=0 + pin_memory=True → no fork conflicts on Kaggle
  5. batch_size=32 + gradient accumulation=2 → effective batch 64

ANTI-OVERFITTING:
  - Focal Loss (gamma=2.0)
  - Confidence penalty (threshold=0.75)
  - Label smoothing 0.1
  - Composite early stopping (AUC - overconf - underfit penalties)
  - ReduceLROnPlateau scheduler
  - Dominant class weights forced to 1.0
  - Post-training temperature scaling
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # must be before ALL imports
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"      # force GPU 0, not GPU 1

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===== HYPERPARAMETERS =====
CONFIG = {
    'batch_size': 32,
    'accumulate_steps': 2,           # effective batch = 64
    'epochs': 50,
    'lr_fusion': 2e-4,
    'lr_backbone': 5e-5,
    'weight_decay': 5e-3,
    'patience': 10,
    'num_classes': 8,
    'vision_dim': 1280,              # EfficientNet-B0 output dim
    'text_dim': 768,
    'metadata_dim': 2,
    'vision_embed_dim': 256,
    'text_embed_dim': 256,
    'metadata_embed_dim': 16,
    'bottleneck_dim': 256,
    'hdc_dim': 4096,
    'dropout_vision': 0.5,
    'dropout_text': 0.4,
    'dropout_fusion': 0.5,
    'label_smoothing': 0.1,
    'grad_clip': 1.0,
    'overconfidence_threshold': 0.9,
    'overconfidence_penalty': 0.02,
    'focal_gamma': 2.0,
    'conf_penalty_weight': 0.03,
    'conf_penalty_threshold': 0.75,
    'pos_weight_cap': 10.0,
    'freeze_backbone_epochs': 0,     # 0 = train end-to-end
    'lr_plateau_factor': 0.5,
    'lr_plateau_patience': 3,
    'underfit_penalty': 0.5,
    'underfit_threshold': 0.35,
}

LABEL_COLUMNS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Pneumonia', 'Pneumothorax'
]

# ===== FOCAL LOSS =====
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight
        self.reduction  = reduction

    def forward(self, logits, targets):
        bce          = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs        = torch.sigmoid(logits)
        p_t          = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss         = focal_weight * bce

        if self.pos_weight is not None:
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss   = loss * weight

        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ===== DATASET WITH MEMORY-EFFICIENT UINT8 CACHE =====
class IUXRayDataset(Dataset):
    def __init__(self, csv_path, text_features_path, transform=None,
                 label_smoothing=0.0, is_train=False):
        self.df            = pd.read_csv(csv_path)
        self.text_features = np.load(text_features_path)
        self.transform     = transform
        self.label_smoothing = label_smoothing
        self.is_train      = is_train

        raw_labels = self.df[LABEL_COLUMNS].values.astype(np.float32)
        if label_smoothing > 0:
            self.labels = raw_labels * (1 - 2 * label_smoothing) + label_smoothing
        else:
            self.labels = raw_labels

        # ── CACHE AS UINT8 NUMPY ARRAYS (not PIL objects) ──────────────────
        # PIL Image objects have large Python overhead (~500KB each) → OOM
        # uint8 numpy arrays store raw pixels only: 224×224×1 = ~50KB each
        # All 3 splits combined: ~400MB total → safe on Kaggle's 13GB RAM
        # Resize done ONCE here, not repeated every epoch
        n_total = len(self.df)
        print(f"  [{os.path.basename(csv_path)}] "
              f"Caching {n_total} images as uint8 arrays...")

        self.frontal_cache = np.zeros((n_total, 224, 224), dtype=np.uint8)
        self.lateral_cache = np.zeros((n_total, 224, 224), dtype=np.uint8)
        self.has_lateral   = np.zeros(n_total, dtype=bool)

        _resize = transforms.Resize((224, 224))
        n_lateral = 0

        for i, (_, row) in enumerate(tqdm(
                self.df.iterrows(), total=n_total,
                desc=f"  Caching", leave=False)):

            # Frontal — always exists
            img = Image.open(row['frontal_file']).convert('L')
            img = _resize(img)
            self.frontal_cache[i] = np.array(img, dtype=np.uint8)

            # Lateral — fall back to frontal if missing
            lateral_file = str(row['lateral_file']) if pd.notna(row['lateral_file']) else ''
            if lateral_file and lateral_file != 'nan' and os.path.exists(lateral_file):
                img = Image.open(lateral_file).convert('L')
                img = _resize(img)
                self.lateral_cache[i] = np.array(img, dtype=np.uint8)
                self.has_lateral[i]   = True
                n_lateral += 1
            else:
                # Copy frontal pixels into lateral slot
                self.lateral_cache[i] = self.frontal_cache[i]

        mem_mb = (self.frontal_cache.nbytes + self.lateral_cache.nbytes) / 1e6
        print(f"  [{os.path.basename(csv_path)}] "
              f"total={n_total}, with_lateral={n_lateral} "
              f"({n_lateral/n_total*100:.1f}%)")
        print(f"  Cache RAM: {mem_mb:.0f} MB ✓")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert cached uint8 array back to PIL for augmentation transforms
        # Resize already done at cache time — no disk I/O here
        frontal_img = Image.fromarray(self.frontal_cache[idx], mode='L')
        lateral_img = Image.fromarray(self.lateral_cache[idx], mode='L')

        if self.transform:
            frontal_img = self.transform(frontal_img)
            lateral_img = self.transform(lateral_img)

        text_feat = self.text_features[idx]

        view = str(row.get('view_position', '')).lower()
        metadata = np.array([
            1.0 if 'frontal' in view else 0.0,
            1.0 if 'lateral' in view else 0.0
        ], dtype=np.float32)

        return {
            'frontal':  frontal_img,
            'lateral':  lateral_img,
            'text':     torch.FloatTensor(text_feat),
            'metadata': torch.FloatTensor(metadata),
            'labels':   torch.FloatTensor(self.labels[idx]),
            'uid':      row['uid']
        }

# ===== MODEL =====
class TriModalClinicalAI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # EfficientNet-B0 backbone
        efficientnet        = models.efficientnet_b0(weights='IMAGENET1K_V1')
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
        features = self.vision_encoder(x)      # (B, 1280, H, W)
        pooled   = self.vision_pool(features)  # (B, 1280, 1, 1)
        return pooled.flatten(1)               # (B, 1280)

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

    def freeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        print("  🔒 EfficientNet-B0 FROZEN")

    def unfreeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = True
        print("  🔓 EfficientNet-B0 UNFROZEN")

# ===== HDC =====
def encode_to_hypervector(embeddings, projection_matrix):
    projected            = embeddings @ projection_matrix
    binary               = np.sign(projected)
    binary[binary == 0]  = 1
    return binary.astype(np.int8)

def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2, axis=1)

# ===== CONFIDENCE PENALTY =====
def confidence_penalty(logits, threshold=0.75):
    probs  = torch.sigmoid(logits)
    excess = torch.clamp(probs - threshold, min=0.0)
    return (excess ** 2).mean()

# ===== TRAINING =====
def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss  = 0
    total_conf  = 0
    accum_steps = config.get('accumulate_steps', 1)

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        frontal  = batch['frontal'].to(device)
        lateral  = batch['lateral'].to(device)
        text     = batch['text'].to(device)
        metadata = batch['metadata'].to(device)
        labels   = batch['labels'].to(device)

        logits     = model(frontal, lateral, text, metadata)
        focal_loss = criterion(logits, labels)
        conf_pen   = confidence_penalty(logits, config['conf_penalty_threshold'])
        loss       = (focal_loss + config['conf_penalty_weight'] * conf_pen) / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_conf += conf_pen.item()

    n = len(loader)
    return total_loss / n, total_conf / n

def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            frontal  = batch['frontal'].to(device)
            lateral  = batch['lateral'].to(device)
            text     = batch['text'].to(device)
            metadata = batch['metadata'].to(device)
            labels   = batch['labels'].to(device)

            logits = model(frontal, lateral, text, metadata)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
        else:
            auc_scores.append(0.0)
    mean_auc = np.mean(auc_scores)

    binary_preds = (all_preds > 0.5).astype(int)
    f1           = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    pred_mean    = all_preds.mean()
    pred_std     = all_preds.std()
    pred_over90  = (all_preds > 0.9).mean() * 100
    pred_under10 = (all_preds < 0.1).mean() * 100

    print(f"  Val F1  @0.5 : {f1:.4f}")
    print(f"  Pred mean    : {pred_mean:.4f}  (target: 0.10-0.35)")
    print(f"  Pred std     : {pred_std:.4f}  (higher = more discriminative)")
    print(f"  Pred > 0.9   : {pred_over90:.1f}%  (⚠️ if > 5%)")
    print(f"  Pred < 0.1   : {pred_under10:.1f}%")

    if pred_mean > config['underfit_threshold'] and pred_std < 0.15:
        print(f"  ⚠️  UNDERFITTING: pred_mean={pred_mean:.3f} > {config['underfit_threshold']} "
              f"AND pred_std={pred_std:.3f} < 0.15 — model not discriminating yet")

    return total_loss / len(loader), mean_auc, auc_scores, pred_over90, pred_mean

# ===== TEMPERATURE CALIBRATION =====
def calibrate_temperature(model, val_loader, device, n_iter=50, lr=0.01):
    print("\n  Calibrating temperature...")
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            frontal  = batch['frontal'].to(device)
            lateral  = batch['lateral'].to(device)
            text     = batch['text'].to(device)
            metadata = batch['metadata'].to(device)
            labels   = batch['labels'].to(device)
            logits   = model(frontal, lateral, text, metadata)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    temp_opt    = optim.LBFGS([temperature], lr=lr, max_iter=n_iter)

    def _eval():
        temp_opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    temp_opt.step(_eval)
    optimal_temp = temperature.item()

    with torch.no_grad():
        raw_preds    = torch.sigmoid(all_logits).cpu().numpy()
        scaled_preds = torch.sigmoid(all_logits / optimal_temp).cpu().numpy()
        print(f"  Optimal temperature : {optimal_temp:.4f}")
        print(f"  Before: pred>0.9={( raw_preds > 0.9).mean()*100:.1f}%  "
              f"mean={raw_preds.mean():.4f}")
        print(f"  After : pred>0.9={(scaled_preds > 0.9).mean()*100:.1f}%  "
              f"mean={scaled_preds.mean():.4f}")

    return optimal_temp

# ===== MAIN =====
def main():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Device   : {device}")
    print(f"GPUs     : {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i} : {torch.cuda.get_device_name(i)}")

    # Resize removed from transforms — done once at cache time
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ── LOAD & CACHE DATASETS ─────────────────────────────────────────────────
    print("\nLoading and caching datasets (uint8 numpy — ~400MB total)...")
    train_dataset = IUXRayDataset(
        '/kaggle/working/train.csv',
        '/kaggle/working/train_text_features.npy',
        transform=train_transform,
        label_smoothing=CONFIG['label_smoothing'],
        is_train=True
    )
    val_dataset = IUXRayDataset(
        '/kaggle/working/val.csv',
        '/kaggle/working/val_text_features.npy',
        transform=val_transform,
        label_smoothing=0.0
    )
    test_dataset = IUXRayDataset(
        '/kaggle/working/test.csv',
        '/kaggle/working/test_text_features.npy',
        transform=val_transform,
        label_smoothing=0.0
    )

    # num_workers=0 → no fork conflicts; pin_memory=True → fast CPU→GPU
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nDataset sizes: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    # ── CLASS WEIGHTS ─────────────────────────────────────────────────────────
    train_labels = train_dataset.df[LABEL_COLUMNS].values.astype(np.float32)
    pos_counts   = train_labels.sum(axis=0)
    neg_counts   = len(train_labels) - pos_counts
    weights      = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, CONFIG['pos_weight_cap'])

    print(f"\nClass weights:")
    for i, label in enumerate(LABEL_COLUMNS):
        if pos_counts[i] / len(train_labels) > 0.5:
            weights[i] = 1.0
            print(f"  {label:15s}: 1.00  "
                  f"(dominant {pos_counts[i]/len(train_labels)*100:.0f}% → forced)")
        else:
            print(f"  {label:15s}: {weights[i]:.2f}  "
                  f"(pos={int(pos_counts[i])}, "
                  f"{pos_counts[i]/len(train_labels)*100:.1f}%)")

    weights_tensor = torch.FloatTensor(weights).to(device)

    # ── MODEL ─────────────────────────────────────────────────────────────────
    model = TriModalClinicalAI(CONFIG).to(device)

    if CONFIG['freeze_backbone_epochs'] > 0:
        model.freeze_vision()

    accum_steps = CONFIG.get('accumulate_steps', 1)
    eff_batch   = CONFIG['batch_size'] * accum_steps

    backbone_params = list(model.vision_encoder.parameters())
    other_params    = [p for n, p in model.named_parameters()
                       if 'vision_encoder' not in n and 'temperature' not in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['lr_backbone']},
        {'params': other_params,    'lr': CONFIG['lr_fusion']}
    ], weight_decay=CONFIG['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['lr_plateau_factor'],
        patience=CONFIG['lr_plateau_patience']
    )
    criterion = FocalBCEWithLogitsLoss(
        gamma=CONFIG['focal_gamma'], pos_weight=weights_tensor
    )

    # ── PHASE 1: TRAINING ────────────────────────────────────────────────────
    best_composite    = -float('inf')
    best_val_auc      = 0
    patience_counter  = 0
    val_loss_history  = []
    backbone_unfrozen = (CONFIG['freeze_backbone_epochs'] == 0)

    MODEL_SAVE_PATH = '/kaggle/working/efficientnet_fused.pth'

    print("\n" + "="*60)
    print("PHASE 1: TRAINING v4 Final")
    print("="*60)
    print(f"  GPU              : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Backbone         : EfficientNet-B0 (1280-dim)")
    print(f"  Batch size       : {CONFIG['batch_size']}  "
          f"(effective: {eff_batch} with grad accum)")
    print(f"  Image cache      : uint8 numpy (~400MB, no disk I/O during training)")
    print(f"  Focal gamma      : {CONFIG['focal_gamma']}")
    print(f"  Label smoothing  : {CONFIG['label_smoothing']}")
    print(f"  Early stopping   : patience={CONFIG['patience']}")

    for epoch in range(CONFIG['epochs']):

        if (CONFIG['freeze_backbone_epochs'] > 0 and
                epoch == CONFIG['freeze_backbone_epochs'] and
                not backbone_unfrozen):
            model.unfreeze_vision()
            backbone_unfrozen = True
            backbone_params = list(model.vision_encoder.parameters())
            other_params    = [p for n, p in model.named_parameters()
                               if 'vision_encoder' not in n and 'temperature' not in n]
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': CONFIG['lr_backbone']},
                {'params': other_params,    'lr': CONFIG['lr_fusion']}
            ], weight_decay=CONFIG['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=CONFIG['lr_plateau_factor'],
                patience=CONFIG['lr_plateau_patience']
            )
            print(f"  → Optimizer rebuilt with backbone params")

        train_loss, train_conf = train_epoch(
            model, train_loader, criterion, optimizer, device, CONFIG
        )
        val_loss, val_auc, val_auc_per_class, pred_over90, pred_mean = validate(
            model, val_loader, criterion, device, CONFIG
        )
        scheduler.step(val_loss)

        overconf_penalty = pred_over90 * CONFIG['overconfidence_penalty']
        underfit_penalty = (max(0, pred_mean - CONFIG['underfit_threshold'])
                            * CONFIG['underfit_penalty'])
        composite        = val_auc - overconf_penalty - underfit_penalty

        val_loss_history.append(val_loss)
        current_lr = optimizer.param_groups[-1]['lr']
        gap        = val_loss - train_loss
        frozen_tag = '' if backbone_unfrozen else '[FROZEN]'

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} {frozen_tag}")
        print(f"  Train Loss       : {train_loss:.4f}  (conf_pen: {train_conf:.4f})")
        print(f"  Val Loss         : {val_loss:.4f}")
        print(f"  Val AUC          : {val_auc:.4f}")
        print(f"  Overconf Penalty : {overconf_penalty:.4f}")
        print(f"  Underfit Penalty : {underfit_penalty:.4f}")
        print(f"  Composite Score  : {composite:.4f}"
              f"  {'← NEW BEST' if composite > best_composite else ''}")
        print(f"  LR               : {current_lr:.6f}")
        print(f"  Train-Val Gap    : {gap:+.4f}"
              f"  {'⚠️ OVERFITTING' if gap > 0.15 else '✓ healthy'}")

        if len(val_loss_history) >= 3:
            trend = val_loss_history[-1] - val_loss_history[-3]
            if trend > 0.05:
                print(f"  ⚠️  Val loss rising ({trend:+.4f} over 3 epochs)")

        if composite > best_composite:
            best_composite   = composite
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved → {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG['patience']:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    # ── PHASE 2: TEMPERATURE SCALING ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: TEMPERATURE SCALING")
    print("="*60)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    optimal_temp = calibrate_temperature(model, val_loader, device)
    model.temperature.data = torch.tensor([optimal_temp], device=device)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  ✓ Model re-saved with temperature={optimal_temp:.4f}")

    # ── PHASE 3: EXTRACT BOTTLENECK EMBEDDINGS ───────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: EXTRACTING BOTTLENECK EMBEDDINGS")
    print("="*60)

    model.eval()

    def extract_embeddings(loader):
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                frontal  = batch['frontal'].to(device)
                lateral  = batch['lateral'].to(device)
                text     = batch['text'].to(device)
                metadata = batch['metadata'].to(device)
                _, bottleneck = model(frontal, lateral, text, metadata,
                                      return_bottleneck=True)
                embeddings.append(bottleneck.cpu().numpy())
        return np.vstack(embeddings)

    train_embeddings = extract_embeddings(train_loader)
    val_embeddings   = extract_embeddings(val_loader)
    test_embeddings  = extract_embeddings(test_loader)

    np.save('/kaggle/working/train_compressed.npy', train_embeddings)
    np.save('/kaggle/working/val_compressed.npy',   val_embeddings)
    np.save('/kaggle/working/test_compressed.npy',  test_embeddings)

    print(f"✓ Saved: Train={train_embeddings.shape}, "
          f"Val={val_embeddings.shape}, Test={test_embeddings.shape}")

    # ── PHASE 4: BUILD HDC INDEX ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4: BUILDING HDC RETRIEVAL INDEX")
    print("="*60)

    np.random.seed(42)
    projection_matrix  = np.random.randn(
        CONFIG['bottleneck_dim'], CONFIG['hdc_dim']
    ) / np.sqrt(CONFIG['hdc_dim'])
    train_hypervectors = encode_to_hypervector(train_embeddings, projection_matrix)

    hdc_index = {
        'hypervectors':      train_hypervectors,
        'projection_matrix': projection_matrix,
        'train_uids':        train_dataset.df['uid'].values,
        'train_labels':      train_labels
    }
    with open('/kaggle/working/retrieval_index.pkl', 'wb') as f:
        pickle.dump(hdc_index, f)

    kb = os.path.getsize('/kaggle/working/retrieval_index.pkl') / 1024
    print(f"✓ HDC index: {kb:.2f} KB  "
          f"({len(train_hypervectors)} vectors × {CONFIG['hdc_dim']} dims)")

    # ── PHASE 5: FINAL EVALUATION ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 5: FINAL EVALUATION ON TEST SET")
    print("="*60)

    print("\n--- Without Temperature Scaling ---")
    test_loss, test_auc, test_auc_per_class, test_over90, _ = validate(
        model, test_loader, criterion, device, CONFIG
    )

    print("\n--- With Temperature Scaling (T={:.4f}) ---".format(optimal_temp))
    model.eval()
    scaled_preds, scaled_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            frontal  = batch['frontal'].to(device)
            lateral  = batch['lateral'].to(device)
            text     = batch['text'].to(device)
            metadata = batch['metadata'].to(device)
            labels   = batch['labels'].to(device)
            logits   = model(frontal, lateral, text, metadata, use_temperature=True)
            scaled_preds.append(torch.sigmoid(logits).cpu().numpy())
            scaled_labels.append(labels.cpu().numpy())

    scaled_preds  = np.vstack(scaled_preds)
    scaled_labels = np.vstack(scaled_labels)

    scaled_auc = []
    for i in range(scaled_labels.shape[1]):
        if len(np.unique(scaled_labels[:, i])) > 1:
            scaled_auc.append(roc_auc_score(scaled_labels[:, i], scaled_preds[:, i]))
        else:
            scaled_auc.append(0.0)

    print(f"  Mean AUC (scaled)  : {np.mean(scaled_auc):.4f}")
    print(f"  Pred mean (scaled) : {scaled_preds.mean():.4f}")
    print(f"  Pred > 0.9 (scaled): {(scaled_preds > 0.9).mean()*100:.1f}%")

    print(f"\nPer-class AUC:")
    for i, label in enumerate(LABEL_COLUMNS):
        print(f"  {label:15s}: raw={test_auc_per_class[i]:.4f}  "
              f"scaled={scaled_auc[i]:.4f}")

    # ── PHASE 6: HDC RETRIEVAL QUALITY ───────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 6: HDC RETRIEVAL QUALITY")
    print("="*60)

    val_hypervectors = encode_to_hypervector(val_embeddings, projection_matrix)
    top1_overlap, top3_overlap = [], []

    for i in range(len(val_hypervectors)):
        distances    = hamming_distance(train_hypervectors, val_hypervectors[i:i+1])
        top3_idx     = np.argsort(distances)[:3]
        query_labels = val_dataset.df[LABEL_COLUMNS].iloc[i].values.astype(float)

        top1_overlap.append(np.sum(query_labels * train_labels[top3_idx[0]]) > 0)
        top3_overlap.append(np.mean([
            np.sum(query_labels * train_labels[idx]) > 0 for idx in top3_idx
        ]))

    print(f"\n  Top-1 label overlap : {np.mean(top1_overlap)*100:.2f}%")
    print(f"  Top-3 label overlap : {np.mean(top3_overlap)*100:.2f}%")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"  Backbone        : EfficientNet-B0 (1280-dim)")
    print(f"  Effective batch : {eff_batch}")
    print(f"  Best Val AUC    : {best_val_auc:.4f}")
    print(f"  Best Composite  : {best_composite:.4f}")
    print(f"  Temperature     : {optimal_temp:.4f}")
    print(f"\nSaved files:")
    print(f"  {MODEL_SAVE_PATH}")
    print(f"  /kaggle/working/train_compressed.npy")
    print(f"  /kaggle/working/val_compressed.npy")
    print(f"  /kaggle/working/test_compressed.npy")
    print(f"  /kaggle/working/retrieval_index.pkl")

if __name__ == "__main__":
    main()