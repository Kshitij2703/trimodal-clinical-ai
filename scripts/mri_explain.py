import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import cv2
import warnings
warnings.filterwarnings('ignore')

CLASSES      = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CONFIG = {
    'vision_dim':1280,'text_dim':768,'metadata_dim':2,
    'vision_embed_dim':256,'text_embed_dim':256,'metadata_embed_dim':32,
    'bottleneck_dim':128,'dropout_vision':0.5,'dropout_text':0.4,'dropout_fusion':0.5,
    'n_classes':4, 'bert_model':'emilyalsentzer/Bio_ClinicalBERT',
}

# ── Model (must match mri.py exactly) ─────────────────────────────────────────
class TriModalMRI(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.vision_encoder = eff.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(
            nn.Linear(cfg['vision_dim'], cfg['vision_embed_dim']),
            nn.BatchNorm1d(cfg['vision_embed_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_vision']))
        self.text_proj = nn.Sequential(
            nn.Linear(cfg['text_dim'], cfg['text_embed_dim']),
            nn.BatchNorm1d(cfg['text_embed_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_text']))
        self.metadata_proj = nn.Sequential(
            nn.Linear(cfg['metadata_dim'], cfg['metadata_embed_dim']), nn.ReLU())
        fusion_dim = cfg['vision_embed_dim'] + cfg['text_embed_dim'] + cfg['metadata_embed_dim']
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, cfg['bottleneck_dim']),
            nn.BatchNorm1d(cfg['bottleneck_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_fusion']))
        self.classifier = nn.Linear(cfg['bottleneck_dim'], cfg['n_classes'])

    def forward(self, image, text_feat, metadata, return_bottleneck=False):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(image)).flatten(1))
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        b = self.fusion_bottleneck(torch.cat([v, t, m], dim=1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_metadata(row):
    age = np.clip(float(row['age']) / 85.0, 0, 1)
    sex = 1.0 if str(row['sex']).upper() in ('M', 'MALE') else 0.0
    return torch.tensor([age, sex], dtype=torch.float32).unsqueeze(0)

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t(img).unsqueeze(0), img

def encode_text(report_text, tokenizer, bert_model, device):
    enc = tokenizer(report_text, return_tensors='pt', max_length=128,
                    padding='max_length', truncation=True).to(device)
    with torch.no_grad():
        feat = bert_model(**enc).last_hidden_state[:, 0]
    return feat, enc['input_ids'], enc['attention_mask']

def encode_to_hypervector(emb, proj):
    b = np.sign(emb @ proj); b[b==0] = 1; return b.astype(np.int8)
def hamming_distance(hvs, query): return np.sum(hvs != query, axis=1)

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def grad_cam(model, img_tensor, text_feat, metadata, pred_idx, device):
    model.eval()
    features, grads = [], []
    hook_f = model.vision_encoder[8].register_forward_hook(lambda m,i,o: features.append(o))
    hook_b = model.vision_encoder[8].register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    model.zero_grad()
    logits = model(img_tensor.to(device), text_feat.to(device), metadata.to(device))
    logits[0, pred_idx].backward()
    hook_f.remove(); hook_b.remove()
    weights = grads[0].mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * features[0]).sum(dim=1).squeeze()).cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, (224, 224))

# ── BERT Attention ─────────────────────────────────────────────────────────────
def bert_attention(report_text, tokenizer, bert_model, device):
    enc = tokenizer(report_text, return_tensors='pt', max_length=128,
                    padding='max_length', truncation=True).to(device)
    with torch.no_grad():
        out = bert_model(**enc, output_attentions=True)
    attn     = torch.stack(out.attentions).mean(dim=(0,2)).squeeze()
    cls_attn = attn[0, 1:]
    tokens   = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])[1:]
    pairs    = [(tok, score.item()) for tok, score in zip(tokens, cls_attn)
                if tok not in ['[PAD]','[SEP]','[CLS]']]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:10]

# ── Metadata Attribution ───────────────────────────────────────────────────────
def metadata_attribution(model, img_tensor, text_feat, metadata, device):
    model.eval()
    metadata = metadata.to(device).requires_grad_(True)
    logits   = model(img_tensor.to(device), text_feat.to(device), metadata)
    logits[0, logits.argmax(1).item()].backward()
    attr = (metadata.grad * metadata).squeeze().cpu().detach().numpy()
    return list(zip(['age', 'sex'], attr))

# ── HDC Retrieval ──────────────────────────────────────────────────────────────
def hdc_retrieve(model, img_tensor, text_feat, metadata, hdc_index, device, k=3):
    model.eval()
    with torch.no_grad():
        _, b = model(img_tensor.to(device), text_feat.to(device),
                     metadata.to(device), return_bottleneck=True)
    query_hv  = encode_to_hypervector(b.cpu().numpy(), hdc_index['projection_matrix'])
    distances = hamming_distance(hdc_index['hypervectors'], query_hv)
    return [{'image_id': hdc_index['image_ids'][i], 'label': hdc_index['labels'][i],
             'distance': int(distances[i])} for i in np.argsort(distances)[:k]]

# ── Explain ────────────────────────────────────────────────────────────────────
def explain(row, model, tokenizer, bert_model, hdc_index, device,
            out_dir='/kaggle/working/mri_explanations', threshold=0.5):
    os.makedirs(f"{out_dir}/{row['filename']}", exist_ok=True)

    img_tensor, pil_img = preprocess_image(row['image_path'])
    text_feat, _, _     = encode_text(str(row['report_text']), tokenizer, bert_model, device)
    metadata            = encode_metadata(row)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device), text_feat, metadata.to(device))
    probs      = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = probs.argmax()
    pred_label = CLASSES[pred_idx]
    true_label = row['label']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{row['filename']} | True: {true_label} | "
                 f"Pred: {pred_label} ({probs[pred_idx]:.2f})",
                 fontsize=12, fontweight='bold')

    # Panel 1: Grad-CAM
    cam     = grad_cam(model, img_tensor.clone(), text_feat.detach().clone(),
                       metadata.clone(), pred_idx, device)
    overlay = np.clip(0.5*np.array(pil_img.resize((224,224)))/255.0 + 0.5*cm.jet(cam)[:,:,:3], 0, 1)
    axes[0].imshow(overlay); axes[0].set_title('Grad-CAM (Vision)'); axes[0].axis('off')

    # Panel 2: BERT Attention
    attn_pairs = bert_attention(str(row['report_text']), tokenizer, bert_model, device)
    tokens, scores = zip(*attn_pairs) if attn_pairs else ([], [])
    axes[1].barh(range(len(tokens)), scores, color='steelblue')
    axes[1].set_yticks(range(len(tokens))); axes[1].set_yticklabels(tokens, fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_title('BERT Attention (Text)')

    # Panel 3: Class probabilities
    axes[2].barh(CLASSES, probs, color=['tomato' if c == pred_label else 'steelblue' for c in CLASSES])
    axes[2].set_xlim(0, 1); axes[2].set_title('Class Probabilities')
    for i, p in enumerate(probs):
        axes[2].text(p + 0.01, i, f'{p:.3f}', va='center', fontsize=9)

    # Panel 4: HDC Retrieval
    retrieved = hdc_retrieve(model, img_tensor.clone(), text_feat.detach().clone(),
                              metadata.clone(), hdc_index, device)
    axes[3].axis('off')
    text = "Top-3 Similar Cases (HDC)\n" + "─"*30 + "\n"
    for i, r in enumerate(retrieved):
        text += f"\n#{i+1} {r['image_id']}\n  {r['label']}\n  dist={r['distance']}\n"
    axes[3].text(0.05, 0.95, text, transform=axes[3].transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3].set_title('HDC Retrieval (Analogical)')

    plt.tight_layout()
    save_path = f"{out_dir}/{row['filename']}/explanation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {save_path}")
    return pred_label, true_label

# ── Main ───────────────────────────────────────────────────────────────────────
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer  = AutoTokenizer.from_pretrained(CONFIG['bert_model'])
bert_model = AutoModel.from_pretrained(CONFIG['bert_model'],
                                        attn_implementation='eager').to(device).eval()

model = TriModalMRI(CONFIG).to(device)
ckpt  = torch.load('/kaggle/working/mri_fused.pth', weights_only=False, map_location=device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Model loaded")

with open('/kaggle/working/mri_retrieval_index.pkl', 'rb') as f:
    hdc_index = pickle.load(f)

test_df = pd.read_csv('/kaggle/working/mri_test.csv')
samples = test_df.groupby('label').apply(
    lambda g: g.sample(min(2, len(g)), random_state=42)).reset_index(drop=True)

print("Generating explanations...")
for _, row in samples.iterrows():
    print(f"\n── {row['filename']} ({row['label']}) ──")
    explain(row, model, tokenizer, bert_model, hdc_index, device)

print(f"\n✓ Done. Saved to /kaggle/working/mri_explanations/")
