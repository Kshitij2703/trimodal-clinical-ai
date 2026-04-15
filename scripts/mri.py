import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from PIL import Image
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    'train_csv':       '/kaggle/working/mri_train.csv',
    'val_csv':         '/kaggle/working/mri_val.csv',
    'test_csv':        '/kaggle/working/mri_test.csv',
    'train_text_npy':  '/kaggle/working/mri_train_text_features.npy',
    'val_text_npy':    '/kaggle/working/mri_val_text_features.npy',
    'test_text_npy':   '/kaggle/working/mri_test_text_features.npy',
    'model_save_path': '/kaggle/working/mri_fused.pth',
    'batch_size':      32,
    'accumulate_steps':2,
    'epochs':          50,
    'lr_backbone':     5e-5,
    'lr_fusion':       2e-4,
    'weight_decay':    5e-3,
    'patience':        10,
    'grad_clip':       1.0,
    'label_smoothing': 0.1,
    'lr_plateau_factor':  0.5,
    'lr_plateau_patience':3,
    'vision_dim':      1280,
    'text_dim':        768,
    'metadata_dim':    2,
    'vision_embed_dim':256,
    'text_embed_dim':  256,
    'metadata_embed_dim':32,
    'bottleneck_dim':  128,
    'hdc_dim':         4096,
    'dropout_vision':  0.5,
    'dropout_text':    0.4,
    'dropout_fusion':  0.5,
    'n_classes':       4,
}

CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── Dataset ────────────────────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, csv_path, text_npy, split):
        self.df         = pd.read_csv(csv_path).reset_index(drop=True)
        self.text_feats = np.load(text_npy).astype(np.float32)
        mean, std       = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        n = len(self.df)
        print(f"[{split}] Caching {n} images...")
        self.img_cache = np.zeros((n, 224, 224, 3), dtype=np.uint8)
        _resize = transforms.Resize((224, 224))
        for i, row in tqdm(self.df.iterrows(), total=n, leave=False):
            img = Image.open(row['image_path']).convert('RGB')
            self.img_cache[i] = np.array(_resize(img), dtype=np.uint8)
        print(f"[{split}] {n} samples | " +
              " | ".join(f"{c}:{(self.df['label']==c).sum()}" for c in CLASSES))

    def _encode_metadata(self, row):
        age = np.clip(float(row['age']) / 85.0, 0, 1)
        sex = 1.0 if str(row['sex']).upper() in ('M', 'MALE') else 0.0
        return torch.tensor([age, sex], dtype=torch.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'image':     self.transform(Image.fromarray(self.img_cache[idx])),
            'text_feat': torch.tensor(self.text_feats[idx]),
            'metadata':  self._encode_metadata(row),
            'label':     torch.tensor(CLASS_TO_IDX[row['label']], dtype=torch.long),
        }

# ── Model ──────────────────────────────────────────────────────────────────────
class TriModalMRI(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        eff = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.vision_encoder = eff.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(
            nn.Linear(cfg['vision_dim'], cfg['vision_embed_dim']),
            nn.BatchNorm1d(cfg['vision_embed_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_vision']))
        self.text_proj = nn.Sequential(
            nn.Linear(cfg['text_dim'], cfg['text_embed_dim']),
            nn.BatchNorm1d(cfg['text_embed_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_text']))
        self.metadata_proj = nn.Sequential(
            nn.Linear(cfg['metadata_dim'], cfg['metadata_embed_dim']), nn.ReLU())
        fusion_dim = cfg['vision_embed_dim'] + cfg['text_embed_dim'] + cfg['metadata_embed_dim']
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, cfg['bottleneck_dim']),
            nn.BatchNorm1d(cfg['bottleneck_dim']), nn.ReLU(),
            nn.Dropout(cfg['dropout_fusion']))
        self.classifier = nn.Linear(cfg['bottleneck_dim'], cfg['n_classes'])

    def forward(self, image, text_feat, metadata, return_bottleneck=False):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(image)).flatten(1))
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        b = self.fusion_bottleneck(torch.cat([v, t, m], dim=1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

# ── HDC ────────────────────────────────────────────────────────────────────────
def encode_to_hypervector(emb, proj):
    b = np.sign(emb @ proj); b[b==0] = 1; return b.astype(np.int8)
def hamming_distance(hvs, query): return np.sum(hvs != query, axis=1)

# ── Train / Eval ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        img  = batch['image'].to(device)
        txt  = batch['text_feat'].to(device)
        meta = batch['metadata'].to(device)
        lbl  = batch['label'].to(device)
        loss = criterion(model(img, txt, meta), lbl) / cfg['accumulate_steps']
        loss.backward()
        if (step+1) % cfg['accumulate_steps'] == 0 or (step+1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step(); optimizer.zero_grad()
        total_loss += loss.item() * cfg['accumulate_steps']
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    for batch in tqdm(loader, desc='Eval', leave=False):
        img  = batch['image'].to(device)
        txt  = batch['text_feat'].to(device)
        meta = batch['metadata'].to(device)
        lbl  = batch['label'].to(device)
        logits = model(img, txt, meta)
        total_loss += criterion(logits, lbl).item()
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    y_bin  = label_binarize(labels, classes=list(range(CONFIG['n_classes'])))
    auc    = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    acc    = (probs.argmax(1) == labels).mean()
    return total_loss / len(loader), auc, acc, probs, labels

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = MRIDataset(CONFIG['train_csv'], CONFIG['train_text_npy'], 'train')
    val_ds   = MRIDataset(CONFIG['val_csv'],   CONFIG['val_text_npy'],   'val')
    test_ds  = MRIDataset(CONFIG['test_csv'],  CONFIG['test_text_npy'],  'test')

    labels  = train_ds.df['label'].map(CLASS_TO_IDX).values
    counts  = np.bincount(labels)
    weights = torch.tensor([1.0/counts[l] for l in labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    model     = TriModalMRI(CONFIG).to(device)
    optimizer = optim.AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': list(model.vision_proj.parameters()) +
                   list(model.text_proj.parameters()) +
                   list(model.metadata_proj.parameters()) +
                   list(model.fusion_bottleneck.parameters()) +
                   list(model.classifier.parameters()),
         'lr': CONFIG['lr_fusion']},
    ], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=CONFIG['lr_plateau_factor'],
        patience=CONFIG['lr_plateau_patience'])

    best_auc, patience_counter = 0, 0
    print(f"\n{'='*60}\nTRAINING\n{'='*60}")

    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, CONFIG)
        val_loss, auc, acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(auc)
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"AUC={auc:.4f} | Acc={acc:.4f}")
        if auc > best_auc:
            best_auc, patience_counter = auc, 0
            torch.save({'state_dict': model.state_dict()}, CONFIG['model_save_path'])
            print(f"  ✓ Best model saved (AUC={auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}"); break

    # ── Test evaluation ────────────────────────────────────────────────────────
    ckpt = torch.load(CONFIG['model_save_path'], weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    _, test_auc, test_acc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n{'='*60}\nFINAL TEST EVALUATION\n{'='*60}")
    print(f"AUC: {test_auc:.4f}  |  Accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_probs.argmax(1), target_names=CLASSES))
    cm = confusion_matrix(test_labels, test_probs.argmax(1))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # Per-class AUC
    y_bin = label_binarize(test_labels, classes=list(range(CONFIG['n_classes'])))
    print(f"\n{'Class':<15} {'AUC':>7}")
    print('-'*24)
    for i, c in enumerate(CLASSES):
        print(f"{c:<15} {roc_auc_score(y_bin[:,i], test_probs[:,i]):>7.4f}")

    # ── Extract embeddings + HDC ───────────────────────────────────────────────
    print(f"\n{'='*60}\nBUILDING HDC INDEX\n{'='*60}")
    model.eval()
    def extract(loader):
        embs = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                _, b = model(batch['image'].to(device), batch['text_feat'].to(device),
                             batch['metadata'].to(device), return_bottleneck=True)
                embs.append(b.cpu().numpy())
        return np.vstack(embs)

    train_loader_ordered = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                                      shuffle=False, num_workers=2, pin_memory=True)
    train_embs = extract(train_loader_ordered)
    np.random.seed(42)
    proj      = np.random.randn(CONFIG['bottleneck_dim'], CONFIG['hdc_dim']) / np.sqrt(CONFIG['hdc_dim'])
    train_hvs = encode_to_hypervector(train_embs, proj)
    hdc_index = {
        'hypervectors':      train_hvs,
        'projection_matrix': proj,
        'image_ids':         train_ds.df['filename'].values,
        'labels':            train_ds.df['label'].values,
    }
    with open('/kaggle/working/mri_retrieval_index.pkl', 'wb') as f:
        pickle.dump(hdc_index, f)

    val_embs  = extract(val_loader)
    val_hvs   = encode_to_hypervector(val_embs, proj)
    top1_match = [hdc_index['labels'][np.argmin(hamming_distance(train_hvs, val_hvs[i:i+1]))] ==
                  val_ds.df['label'].values[i] for i in range(len(val_hvs))]
    print(f"HDC top-1 label match: {np.mean(top1_match)*100:.2f}%")
    print(f"Saved: {CONFIG['model_save_path']}")
    print(f"Saved: /kaggle/working/mri_retrieval_index.pkl")

if __name__ == '__main__':
    main()
