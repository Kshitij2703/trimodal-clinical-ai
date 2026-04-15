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

LOCALIZATION_VOCAB = [
    "abdomen","acral","back","chest","ear","face","foot",
    "genital","hand","lower extremity","neck","scalp",
    "trunk","unknown","upper extremity"
]
DX_FULL = {
    "mel":"Melanoma","nv":"Melanocytic Nevi","bcc":"Basal Cell Carcinoma",
    "akiec":"Actinic Keratosis","bkl":"Benign Keratosis","df":"Dermatofibroma","vasc":"Vascular Lesion"
}

CONFIG = {
    'vision_dim':1280,'text_dim':768,'metadata_dim':17,
    'vision_embed_dim':256,'text_embed_dim':256,'metadata_embed_dim':32,
    'bottleneck_dim':128,'dropout_vision':0.5,'dropout_text':0.4,'dropout_fusion':0.5,
    'bert_model':'emilyalsentzer/Bio_ClinicalBERT',
}

# ── Model (matches training — no BERT inside, uses pre-extracted text features) ─
class TriModalSkinAI(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.vision_encoder    = eff.features
        self.vision_pool       = nn.AdaptiveAvgPool2d(1)
        self.vision_proj       = nn.Sequential(
            nn.Linear(cfg['vision_dim'], cfg['vision_embed_dim']),
            nn.BatchNorm1d(cfg['vision_embed_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_vision']))
        self.text_proj         = nn.Sequential(
            nn.Linear(cfg['text_dim'], cfg['text_embed_dim']),
            nn.BatchNorm1d(cfg['text_embed_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_text']))
        self.metadata_proj     = nn.Sequential(nn.Linear(cfg['metadata_dim'], 32), nn.ReLU())
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(cfg['vision_embed_dim']+cfg['text_embed_dim']+32, cfg['bottleneck_dim']),
            nn.BatchNorm1d(cfg['bottleneck_dim']), nn.ReLU(), nn.Dropout(cfg['dropout_fusion']))
        self.classifier  = nn.Linear(cfg['bottleneck_dim'], 1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, image, text_feat, metadata, return_bottleneck=False):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(image)).flatten(1))
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        b = self.fusion_bottleneck(torch.cat([v, t, m], dim=1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_metadata(row):
    age = np.clip(float(row['age'])/85.0 if pd.notna(row.get('age')) else 0.5, 0, 1)
    sex = str(row.get('sex', '')).lower()
    sex_enc = 1.0 if sex == 'male' else (0.0 if sex == 'female' else 0.5)
    loc = str(row.get('localization', 'unknown')).lower()
    loc_vec = np.zeros(15, dtype=np.float32)
    loc_vec[LOCALIZATION_VOCAB.index(loc) if loc in LOCALIZATION_VOCAB else LOCALIZATION_VOCAB.index('unknown')] = 1.0
    return torch.tensor(np.array([age, sex_enc, *loc_vec], dtype=np.float32)).unsqueeze(0)

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

def hamming_distance(hvs, query): return np.sum(hvs != query, axis=1)
def encode_to_hypervector(emb, proj):
    b = np.sign(emb @ proj); b[b==0] = 1; return b.astype(np.int8)

# ── Layer 1: Grad-CAM ──────────────────────────────────────────────────────────
def grad_cam(model, img_tensor, text_feat, metadata, device):
    model.eval()
    img_tensor = img_tensor.to(device)
    features, grads = [], []
    hook_f = model.vision_encoder[8].register_forward_hook(lambda m,i,o: features.append(o))
    hook_b = model.vision_encoder[8].register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    model.zero_grad()
    logits = model(img_tensor, text_feat.to(device), metadata.to(device))
    logits.backward()
    hook_f.remove(); hook_b.remove()
    weights = grads[0].mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * features[0]).sum(dim=1).squeeze()).cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, (224, 224))

# ── Layer 2: BERT Attention ────────────────────────────────────────────────────
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

# ── Layer 3: Metadata Attribution ─────────────────────────────────────────────
def metadata_attribution(model, img_tensor, text_feat, metadata, device):
    model.eval()
    metadata = metadata.to(device).requires_grad_(True)
    logits = model(img_tensor.to(device), text_feat.to(device), metadata)
    logits.backward()
    attr = (metadata.grad * metadata).squeeze().cpu().detach().numpy()
    meta_names = ['age', 'sex'] + [f'loc:{v}' for v in LOCALIZATION_VOCAB]
    return list(zip(meta_names, attr))

# ── Layer 4: HDC Retrieval ─────────────────────────────────────────────────────
def hdc_retrieve(model, img_tensor, text_feat, metadata, hdc_index, device, k=3):
    model.eval()
    with torch.no_grad():
        _, bottleneck = model(img_tensor.to(device), text_feat.to(device),
                              metadata.to(device), return_bottleneck=True)
    query_hv  = encode_to_hypervector(bottleneck.cpu().numpy(), hdc_index['projection_matrix'])
    distances = hamming_distance(hdc_index['hypervectors'], query_hv)
    return [{'image_id': hdc_index['image_ids'][i], 'dx': hdc_index['dx'][i],
             'dx_name': DX_FULL.get(hdc_index['dx'][i], '?'),
             'malignant': int(hdc_index['labels'][i]), 'distance': int(distances[i])}
            for i in np.argsort(distances)[:k]]

# ── Explain ────────────────────────────────────────────────────────────────────
def explain(row, model, tokenizer, bert_model, hdc_index, device,
            out_dir='/kaggle/working/explanations', threshold=0.67):
    os.makedirs(f"{out_dir}/{row['image_id']}", exist_ok=True)

    img_tensor, pil_img = preprocess_image(row['image_path'])
    text_feat, _, _     = encode_text(str(row['report_text']), tokenizer, bert_model, device)
    metadata            = encode_metadata(row)

    model.eval()
    with torch.no_grad():
        logit = model(img_tensor.to(device), text_feat, metadata.to(device))
    prob       = torch.sigmoid(logit).item()
    pred_label = 'malignant' if prob >= threshold else 'benign'
    true_label = 'malignant' if row['malignant'] == 1 else 'benign'

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{row['image_id']} | {DX_FULL.get(row['dx'], row['dx'])} | "
                 f"Pred: {pred_label} ({prob:.2f}) | True: {true_label}",
                 fontsize=12, fontweight='bold')

    # Panel 1: Grad-CAM
    cam     = grad_cam(model, img_tensor.clone(), text_feat.detach().clone(), metadata.clone(), device)
    overlay = np.clip(0.5*np.array(pil_img.resize((224,224)))/255.0 + 0.5*cm.jet(cam)[:,:,:3], 0, 1)
    axes[0].imshow(overlay); axes[0].set_title('Grad-CAM (Vision)'); axes[0].axis('off')

    # Panel 2: BERT Attention
    attn_pairs = bert_attention(str(row['report_text']), tokenizer, bert_model, device)
    tokens, scores = zip(*attn_pairs) if attn_pairs else ([], [])
    axes[1].barh(range(len(tokens)), scores, color='steelblue')
    axes[1].set_yticks(range(len(tokens))); axes[1].set_yticklabels(tokens, fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_title('BERT Attention (Text)'); axes[1].set_xlabel('Attention weight')

    # Panel 3: Metadata Attribution
    meta_attr = sorted([(n,v) for n,v in metadata_attribution(model, img_tensor.clone(),
                         text_feat.detach().clone(), metadata.clone(), device) if abs(v)>1e-6],
                        key=lambda x: abs(x[1]), reverse=True)[:8]
    if meta_attr:
        names, vals = zip(*meta_attr)
        axes[2].barh(range(len(names)), vals, color=['tomato' if v>0 else 'steelblue' for v in vals])
        axes[2].set_yticks(range(len(names))); axes[2].set_yticklabels(names, fontsize=9)
        axes[2].invert_yaxis(); axes[2].axvline(0, color='black', linewidth=0.8)
    axes[2].set_title('Metadata Attribution')

    # Panel 4: HDC Retrieval
    retrieved = hdc_retrieve(model, img_tensor.clone(), text_feat.detach().clone(),
                              metadata.clone(), hdc_index, device)
    axes[3].axis('off')
    text = "Top-3 Similar Cases (HDC)\n" + "─"*30 + "\n"
    for i, r in enumerate(retrieved):
        text += f"\n#{i+1} {r['image_id']}\n  {r['dx_name']}\n  {'🔴 malignant' if r['malignant'] else '🟢 benign'}\n  dist={r['distance']}\n"
    axes[3].text(0.05, 0.95, text, transform=axes[3].transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3].set_title('HDC Retrieval (Analogical)')

    plt.tight_layout()
    save_path = f"{out_dir}/{row['image_id']}/explanation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {save_path}")
    return prob, pred_label, true_label

# ── Main ───────────────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(CONFIG['bert_model'])
bert_model = AutoModel.from_pretrained(CONFIG['bert_model'], attn_implementation='eager').to(device).eval()

model = TriModalSkinAI(CONFIG).to(device)
ckpt  = torch.load('/kaggle/working/skin_fused.pth', weights_only=False, map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)
threshold = ckpt.get('threshold', 0.67)
model.eval()
print(f"Model loaded | threshold={threshold:.2f}")

with open('/kaggle/working/retrieval_index.pkl', 'rb') as f:
    hdc_index = pickle.load(f)

test_df = pd.read_csv('/kaggle/working/test.csv')
samples = test_df.groupby('dx').apply(lambda g: g.sample(min(2, len(g)), random_state=42)).reset_index(drop=True)

print("Generating explanations...")
for _, row in samples.iterrows():
    print(f"\n── {row['image_id']} ({row['dx']}) ──")
    explain(row, model, tokenizer, bert_model, hdc_index, device, threshold=threshold)

print(f"\n✓ Done. Explanations saved to /kaggle/working/explanations/")
