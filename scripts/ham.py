import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    'train_csv':          '/kaggle/working/train.csv',
    'val_csv':            '/kaggle/working/val.csv',
    'test_csv':           '/kaggle/working/test.csv',
    'train_text_npy':     '/kaggle/working/train_text_features.npy',
    'val_text_npy':       '/kaggle/working/val_text_features.npy',
    'test_text_npy':      '/kaggle/working/test_text_features.npy',
    'model_save_path':    '/kaggle/working/skin_fused.pth',
    'image_size':         224,
    'batch_size':         32,
    'accumulate_steps':   2,
    'epochs':             50,
    'lr_backbone':        5e-5,
    'lr_fusion':          2e-4,
    'weight_decay':       5e-3,
    'patience':           10,
    'grad_clip':          1.0,
    'focal_gamma':        1.0,       # FIX 2: lowered from 2.0 → less aggressive minority focus
    'label_smoothing':    0.1,
    'conf_penalty_weight':0.03,
    'conf_penalty_thresh':0.60,
    'lr_plateau_factor':  0.5,
    'lr_plateau_patience':3,
    # model dims
    'vision_dim':         1280,
    'text_dim':           768,
    'metadata_dim':       17,
    'vision_embed_dim':   256,
    'text_embed_dim':     256,
    'metadata_embed_dim': 32,
    'bottleneck_dim':     128,
    'hdc_dim':            4096,
    'dropout_vision':     0.5,
    'dropout_text':       0.4,
    'dropout_fusion':     0.5,
}

LOCALIZATION_VOCAB = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp",
    "trunk", "unknown", "upper extremity"
]

DX_MALIGNANT = {
    "mel": 1, "bcc": 1, "akiec": 1,
    "nv": 0,  "bkl": 0, "df": 0, "vasc": 0,
}

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        loss = ((1 - p_t) ** self.gamma) * bce
        if self.pos_weight is not None:
            loss = loss * (targets * self.pos_weight + (1 - targets))
        return loss.mean()

# ── Dataset ────────────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, csv_path, text_npy, split, label_smoothing=0.0):
        self.df         = pd.read_csv(csv_path).reset_index(drop=True)
        self.text_feats = np.load(text_npy).astype(np.float32)
        self.transform  = self._get_transform(split)
        self.smooth     = label_smoothing

        n = len(self.df)
        print(f"[{split}] Caching {n} images...")
        self.img_cache = np.zeros((n, 224, 224, 3), dtype=np.uint8)
        _resize = transforms.Resize((224, 224))
        for i, row in tqdm(self.df.iterrows(), total=n, leave=False):
            img = Image.open(row['image_path']).convert('RGB')
            self.img_cache[i] = np.array(_resize(img), dtype=np.uint8)
        print(f"[{split}] Cache: {self.img_cache.nbytes / 1e6:.0f} MB")

        mal = int(self.df['malignant'].sum())
        print(f"[{split}] Loaded {n} samples | Malignant: {mal} | Benign: {n - mal}")

    def _get_transform(self, split):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if split == 'train':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _encode_metadata(self, row):
        age     = float(row['age']) / 85.0 if pd.notna(row['age']) else 0.5
        age     = np.clip(age, 0.0, 1.0)
        sex     = str(row.get('sex', '')).lower()
        sex_enc = 1.0 if sex == 'male' else (0.0 if sex == 'female' else 0.5)
        loc     = str(row.get('localization', 'unknown')).lower()
        loc_vec = np.zeros(15, dtype=np.float32)
        idx     = LOCALIZATION_VOCAB.index(loc) if loc in LOCALIZATION_VOCAB \
                  else LOCALIZATION_VOCAB.index('unknown')
        loc_vec[idx] = 1.0
        return torch.tensor(np.array([age, sex_enc, *loc_vec], dtype=np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = self.transform(Image.fromarray(self.img_cache[idx]))
        label = float(row['malignant'])
        if self.smooth > 0:
            label = label * (1 - 2 * self.smooth) + self.smooth
        return {
            'image':     img,
            'text_feat': torch.tensor(self.text_feats[idx]),
            'metadata':  self._encode_metadata(row),
            'label':     torch.tensor(label, dtype=torch.float32),
        }

# ── Model ──────────────────────────────────────────────────────────────────────
class TriModalSkinAI(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        eff = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.vision_encoder = eff.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(
            nn.Linear(cfg['vision_dim'], cfg['vision_embed_dim']),
            nn.BatchNorm1d(cfg['vision_embed_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_vision']),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(cfg['text_dim'], cfg['text_embed_dim']),
            nn.BatchNorm1d(cfg['text_embed_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_text']),
        )
        self.metadata_proj = nn.Sequential(
            nn.Linear(cfg['metadata_dim'], cfg['metadata_embed_dim']), nn.ReLU()
        )
        fusion_dim = cfg['vision_embed_dim'] + cfg['text_embed_dim'] + cfg['metadata_embed_dim']
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, cfg['bottleneck_dim']),
            nn.BatchNorm1d(cfg['bottleneck_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_fusion']),
        )
        self.classifier  = nn.Linear(cfg['bottleneck_dim'], 1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, image, text_feat, metadata,
                return_bottleneck=False, use_temperature=False):
        v = self.vision_pool(self.vision_encoder(image)).flatten(1)
        v = self.vision_proj(v)
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        bottleneck = self.fusion_bottleneck(torch.cat([v, t, m], dim=1))
        logits     = self.classifier(bottleneck)
        if use_temperature:
            logits = logits / self.temperature
        if return_bottleneck:
            return logits, bottleneck
        return logits

# ── HDC ────────────────────────────────────────────────────────────────────────
def encode_to_hypervector(embeddings, proj):
    binary = np.sign(embeddings @ proj)
    binary[binary == 0] = 1
    return binary.astype(np.int8)

def hamming_distance(hvs, query):
    return np.sum(hvs != query, axis=1)

# ── Confidence penalty ─────────────────────────────────────────────────────────
def conf_penalty(logits, threshold=0.60):
    return torch.clamp(torch.sigmoid(logits) - threshold, min=0.0).pow(2).mean()

# ── Train epoch ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        img  = batch['image'].to(device)
        txt  = batch['text_feat'].to(device)
        meta = batch['metadata'].to(device)
        lbl  = batch['label'].unsqueeze(1).to(device)

        logits = model(img, txt, meta)
        loss   = (criterion(logits, lbl) +
                  cfg['conf_penalty_weight'] * conf_penalty(
                      logits, cfg['conf_penalty_thresh'])
                  ) / cfg['accumulate_steps']
        loss.backward()

        if (step + 1) % cfg['accumulate_steps'] == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg['accumulate_steps']
    return total_loss / len(loader)

# ── Evaluate ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc='Eval', leave=False):
        img  = batch['image'].to(device)
        txt  = batch['text_feat'].to(device)
        meta = batch['metadata'].to(device)
        lbl  = batch['label'].unsqueeze(1).to(device)

        logits = model(img, txt, meta)
        total_loss += criterion(logits, lbl).item()
        preds.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(lbl.cpu().numpy())

    preds       = np.vstack(preds)
    labels      = np.vstack(labels)
    hard_labels = (labels > 0.5).astype(int)   # unsmoothed for metrics
    auc = roc_auc_score(hard_labels, preds)
    f1  = f1_score(hard_labels, (preds > threshold).astype(int), average='binary')
    return total_loss / len(loader), auc, f1, preds.mean(), preds, hard_labels


# FIX 3: threshold sweep uses macro F1 — balances both benign and malignant
def find_best_threshold(preds, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(labels, (preds > t).astype(int),
                      average='macro',          # was 'binary' — now balances both classes
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ── Temperature calibration ────────────────────────────────────────────────────
def calibrate_temperature(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['image'].to(device),
                           batch['text_feat'].to(device),
                           batch['metadata'].to(device))
            all_logits.append(logits)
            all_labels.append(batch['label'].unsqueeze(1).to(device))
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    T   = nn.Parameter(torch.ones(1, device=device) * 1.5)
    opt = optim.LBFGS([T], lr=0.01, max_iter=50)

    def _eval():
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(all_logits / T, all_labels).backward()
        return F.binary_cross_entropy_with_logits(all_logits / T, all_labels)

    opt.step(_eval)
    print(f"  Optimal temperature: {T.item():.4f}")
    return T.item()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = SkinDataset(CONFIG['train_csv'], CONFIG['train_text_npy'],
                           'train', CONFIG['label_smoothing'])
    val_ds   = SkinDataset(CONFIG['val_csv'],   CONFIG['val_text_npy'],   'val')
    test_ds  = SkinDataset(CONFIG['test_csv'],  CONFIG['test_text_npy'],  'test')

    # WeightedRandomSampler on train — use raw integer labels (not smoothed)
    raw_labels = train_ds.df['malignant'].values.astype(int)
    counts     = np.bincount(raw_labels)
    weights    = torch.tensor([1.0 / counts[l] for l in raw_labels], dtype=torch.float32)
    sampler    = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    # FIX 1: pos_weight removed — sampler already balances classes
    # Using pos_weight ON TOP of WeightedRandomSampler double-penalises benign errors
    criterion = FocalBCELoss(gamma=CONFIG['focal_gamma'], pos_weight=None)

    model = TriModalSkinAI(CONFIG).to(device)

    backbone_params = list(model.vision_encoder.parameters())
    other_params    = [p for n, p in model.named_parameters()
                       if 'vision_encoder' not in n and 'temperature' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['lr_backbone']},
        {'params': other_params,    'lr': CONFIG['lr_fusion']},
    ], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['lr_plateau_factor'],
        patience=CONFIG['lr_plateau_patience']
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_auc, best_macro_f1, patience_counter = 0.0, 0.0, 0
    print(f"\n{'='*60}\nTRAINING\n{'='*60}")

    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, device, CONFIG)
        val_loss, auc, _, pmean, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)

        best_t, macro_f1 = find_best_threshold(val_preds, val_labels)
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"AUC={auc:.4f} | macroF1={macro_f1:.4f} (t={best_t:.2f}) | "
              f"pred_mean={pmean:.3f}")

        # FIX 4: save on AUC improvement AND require macro F1 > 0.60
        # prevents saving the "predict everything malignant" collapse
        if auc > best_auc and macro_f1 > 0.60:
            best_auc        = auc
            best_macro_f1   = macro_f1
            patience_counter = 0
            torch.save({'state_dict': model.state_dict(), 'threshold': best_t},
                       CONFIG['model_save_path'])
            print(f"  ✓ Best model saved (AUC={auc:.4f}, macroF1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if auc > best_auc and macro_f1 <= 0.60:
                print(f"  ✗ AUC improved but macroF1={macro_f1:.4f} < 0.60 — not saving")
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Load best + temperature scaling ───────────────────────────────────────
    ckpt      = torch.load(CONFIG['model_save_path'], weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    threshold = ckpt['threshold']
    opt_temp  = calibrate_temperature(model, val_loader, device)
    model.temperature.data = torch.tensor([opt_temp], device=device)
    torch.save({'state_dict': model.state_dict(), 'threshold': threshold},
               CONFIG['model_save_path'])

    # ── Extract bottleneck embeddings ──────────────────────────────────────────
    print(f"\n{'='*60}\nEXTRACTING EMBEDDINGS\n{'='*60}")
    model.eval()

    def extract(loader):
        embs = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                _, b = model(batch['image'].to(device),
                             batch['text_feat'].to(device),
                             batch['metadata'].to(device),
                             return_bottleneck=True)
                embs.append(b.cpu().numpy())
        return np.vstack(embs)

    train_embs = extract(train_loader)
    val_embs   = extract(val_loader)
    test_embs  = extract(test_loader)
    np.save('/kaggle/working/train_embeddings.npy', train_embs)
    np.save('/kaggle/working/val_embeddings.npy',   val_embs)
    np.save('/kaggle/working/test_embeddings.npy',  test_embs)

    # ── Build HDC index ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nBUILDING HDC INDEX\n{'='*60}")
    np.random.seed(42)
    proj      = np.random.randn(CONFIG['bottleneck_dim'],
                                CONFIG['hdc_dim']) / np.sqrt(CONFIG['hdc_dim'])
    train_hvs = encode_to_hypervector(train_embs, proj)

    hdc_index = {
        'hypervectors':      train_hvs,
        'projection_matrix': proj,
        'image_ids':         train_ds.df['image_id'].values,
        'labels':            train_ds.df['malignant'].values,
        'dx':                train_ds.df['dx'].values,
    }
    with open('/kaggle/working/retrieval_index.pkl', 'wb') as f:
        pickle.dump(hdc_index, f)
    print(f"  HDC index: {os.path.getsize('/kaggle/working/retrieval_index.pkl')/1024:.1f} KB")

    # ── HDC retrieval quality ──────────────────────────────────────────────────
    val_hvs      = encode_to_hypervector(val_embs, proj)
    train_labels = train_ds.df['malignant'].values
    val_labels_r = val_ds.df['malignant'].values
    top1_match   = [
        train_labels[np.argmin(hamming_distance(train_hvs, val_hvs[i:i+1]))] == val_labels_r[i]
        for i in range(len(val_hvs))
    ]
    print(f"  HDC top-1 label match: {np.mean(top1_match)*100:.2f}%")

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nFINAL TEST EVALUATION\n{'='*60}")
    _, test_auc, _, _, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, threshold=threshold)

    print(f"AUC: {test_auc:.4f}  |  Threshold: {threshold:.2f}")
    print(classification_report(
        test_labels,
        (test_preds > threshold).astype(int),
        target_names=['benign', 'malignant']
    ))

    # ── Per-class breakdown ────────────────────────────────────────────────────
    print(f"\n{'Class':6s}  {'N':>5}  {'Mal':3}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}")
    print('-' * 45)
    for dx, is_mal in DX_MALIGNANT.items():
        mask = test_ds.df['dx'].values == dx
        if mask.sum() == 0:
            continue
        y_true = test_labels[mask]
        y_pred = (test_preds[mask] > threshold).astype(int)
        f1_c = f1_score(y_true, y_pred, zero_division=0)
        pr   = precision_score(y_true, y_pred, zero_division=0)
        rc   = recall_score(y_true, y_pred, zero_division=0)
        print(f"{dx:6s}  {mask.sum():>5}  {'✓' if is_mal else ' ':3}  "
              f"{f1_c:>7.4f}  {pr:>7.4f}  {rc:>7.4f}")

    print(f"\nSaved: {CONFIG['model_save_path']}")
    print(f"Saved: /kaggle/working/retrieval_index.pkl")


if __name__ == '__main__':
    main()