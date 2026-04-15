"""
demo.py — Tri-Modal Clinical AI Gradio Demo
Run: python demo.py
"""

import json, pickle, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from agent import ClinicalAgent

# ── Load agent + PyTorch models for Grad-CAM ──────────────────────────────────
agent = ClinicalAgent()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

SKIN_LOCS = ["abdomen","acral","back","chest","ear","face","foot",
             "genital","hand","lower extremity","neck","scalp",
             "trunk","unknown","upper extremity"]

MRI_CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
SKIN_CLASSES = ["benign", "malignant"]
XRAY_CLASSES = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
                "Effusion","Emphysema","Pneumonia","Pneumothorax"]

# ── PyTorch model definitions (for Grad-CAM) ───────────────────────────────────
class _FusedBase(nn.Module):
    def __init__(self, meta_dim, bottleneck, n_classes, fusion_dim_extra=0):
        super().__init__()
        b = models.efficientnet_b0(weights=None)
        self.vision_encoder    = b.features
        self.vision_pool       = nn.AdaptiveAvgPool2d(1)
        self.vision_proj       = nn.Sequential(nn.Linear(1280,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5))
        self.text_proj         = nn.Sequential(nn.Linear(768,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.metadata_proj     = nn.Sequential(nn.Linear(meta_dim, 32), nn.ReLU())
        self.fusion_bottleneck = nn.Sequential(nn.Linear(256+256+32+fusion_dim_extra, bottleneck),
                                               nn.BatchNorm1d(bottleneck), nn.ReLU(), nn.Dropout(0.5))
        self.classifier        = nn.Linear(bottleneck, n_classes)

class MRIModel(_FusedBase):
    def __init__(self): super().__init__(2, 128, 4)
    def forward(self, img, txt, meta, return_bottleneck=False):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(img)).flatten(1))
        t = self.text_proj(txt); m = self.metadata_proj(meta)
        b = self.fusion_bottleneck(torch.cat([v,t,m],1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

class SkinModel(_FusedBase):
    def __init__(self): super().__init__(17, 128, 1)
    def forward(self, img, txt, meta, return_bottleneck=False):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(img)).flatten(1))
        t = self.text_proj(txt); m = self.metadata_proj(meta)
        b = self.fusion_bottleneck(torch.cat([v,t,m],1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

class XRayModel(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b0(weights=None)
        self.vision_encoder = b.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(nn.Linear(1280,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5))
        self.text_proj      = nn.Sequential(nn.Linear(768,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.metadata_proj  = nn.Sequential(nn.Linear(2,16),     nn.ReLU())
        self.fusion         = nn.Sequential(nn.Linear(528,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5))
        self.classifier     = nn.Linear(256, 8)
    def forward(self, frontal, lateral, txt, meta, return_bottleneck=False):
        f = self.vision_pool(self.vision_encoder(frontal)).flatten(1)
        l = self.vision_pool(self.vision_encoder(lateral)).flatten(1)
        v = self.vision_proj((f+l)/2.0)
        t = self.text_proj(txt); m = self.metadata_proj(meta)
        b = self.fusion(torch.cat([v,t,m],1))
        logits = self.classifier(b)
        return (logits, b) if return_bottleneck else logits

def _load(cls, path):
    m = cls(); sd = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(sd.get('state_dict', sd), strict=False)
    return m.eval()

_pt_models = {
    'mri':  _load(MRIModel,  'models/mri_fused.pth'),
    'skin': _load(SkinModel, 'models/skin_fused.pth'),
    'xray': _load(XRayModel, 'models/efficientnet_fused.pth'),
}

_hdc = {
    'mri':  pickle.load(open('retrieval/mri_retrieval_index.pkl',  'rb')),
    'skin': pickle.load(open('retrieval/skin_retrieval_index.pkl', 'rb')),
    'xray': pickle.load(open('retrieval/retrieval_index.pkl',      'rb')),
}

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def _grad_cam(model, hook_layer, forward_fn, pred_idx):
    feats, grads = [], []
    fh = hook_layer.register_forward_hook(lambda m,i,o: feats.append(o))
    bh = hook_layer.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    model.zero_grad()
    logits = forward_fn()
    logits[0, pred_idx].backward()
    fh.remove(); bh.remove()
    w = grads[0].mean(dim=(2,3), keepdim=True)
    cam = F.relu((w * feats[0]).sum(1).squeeze()).detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, (224, 224))

def make_gradcam_overlay(image_path, modality, txt_feat, meta_tensor,
                         pred_idx, lateral_path=None):
    model = _pt_models[modality]
    img_t = TRANSFORM(Image.open(image_path).convert('RGB')).unsqueeze(0)
    pil   = Image.open(image_path).convert('RGB').resize((224,224))

    if modality == 'xray':
        lat_t = TRANSFORM(Image.open(lateral_path).convert('RGB')).unsqueeze(0) if lateral_path else img_t
        cam = _grad_cam(model, model.vision_encoder[8],
                        lambda: model(img_t, lat_t, txt_feat, meta_tensor), pred_idx)
    else:
        cam = _grad_cam(model, model.vision_encoder[8],
                        lambda: model(img_t, txt_feat, meta_tensor), pred_idx)

    overlay = np.clip(
        0.5 * np.array(pil) / 255.0 + 0.5 * cm.jet(cam)[:,:,:3], 0, 1)
    return (overlay * 255).astype(np.uint8)

# ── HDC retrieval ──────────────────────────────────────────────────────────────
def _encode_hv(emb, proj):
    b = np.sign(emb @ proj); b[b==0] = 1; return b.astype(np.int8)

def hdc_retrieve(modality, image_path, txt_feat, meta_tensor,
                 lateral_path=None, k=3):
    model = _pt_models[modality]
    img_t = TRANSFORM(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        if modality == 'xray':
            lat_t = TRANSFORM(Image.open(lateral_path).convert('RGB')).unsqueeze(0) if lateral_path else img_t
            _, b = model(img_t, lat_t, txt_feat, meta_tensor, return_bottleneck=True)
        else:
            _, b = model(img_t, txt_feat, meta_tensor, return_bottleneck=True)

    idx  = _hdc[modality]
    proj = idx['projection_matrix']
    hvs  = idx['hypervectors']
    qhv  = _encode_hv(b.numpy(), proj)
    dist = np.sum(hvs != qhv, axis=1)
    top  = np.argsort(dist)[:k]

    ids_key    = 'image_ids' if 'image_ids' in idx else ('train_uids' if 'train_uids' in idx else None)
    labels_key = 'labels' if 'labels' in idx else 'train_labels'
    dx_key     = 'dx' if 'dx' in idx else None

    results = []
    for i in top:
        r = {'distance': int(dist[i]), 'label': str(idx[labels_key][i])}
        if ids_key:   r['id']  = str(idx[ids_key][i])
        if dx_key:    r['dx']  = str(idx[dx_key][i])
        results.append(r)
    return results

# ── Build text feat tensor from ONNX BERT output ──────────────────────────────
def _bert_feat(text):
    from transformers import AutoTokenizer
    import onnxruntime as ort
    # reuse agent's session
    enc = agent._tokenizer(text, return_tensors='np', max_length=128,
                           padding='max_length', truncation=True)
    out = agent._bert.run(None, {
        'input_ids':      enc['input_ids'].astype(np.int64),
        'attention_mask': enc['attention_mask'].astype(np.int64),
    })
    return torch.tensor(out[0], dtype=torch.float32)  # (1,768)

# ── Metadata tensors ───────────────────────────────────────────────────────────
def _meta_tensor(modality, age, sex, localization, view):
    if modality == 'mri':
        return torch.tensor([[age/80., 1. if sex=='M' else 0.]], dtype=torch.float32)
    elif modality == 'skin':
        sex_b = 1. if sex=='M' else 0.
        loc_oh = np.zeros(15, dtype=np.float32)
        loc = localization.lower()
        if loc in SKIN_LOCS[:15]: loc_oh[SKIN_LOCS.index(loc)] = 1.
        return torch.tensor([[age/80., sex_b, *loc_oh]], dtype=torch.float32)
    else:
        return torch.tensor([[1.,0.] if view=='PA' else [0.,1.]], dtype=torch.float32)

# ── Main predict ───────────────────────────────────────────────────────────────
# ── Modality mismatch detection ────────────────────────────────────────────────
# Modality mismatch + OOD detection is now handled inside agent.py
# Warnings are returned in result['warnings']

# Default reports are now generated inside agent.py

# ── Session history ─────────────────────────────────────────────────────────────
_history = []  # list of dicts

def predict(modality, image, lateral_image, report_text, age, sex, localization, view):
    if image is None:
        return "⚠️ Please upload an image.", None, None, None, _render_history()

    meta = {}
    if modality == 'skin':  meta = {'age': age, 'sex': sex, 'localization': localization}
    elif modality == 'mri': meta = {'age': age, 'sex': sex}
    else:                   meta = {'view': view}

    # ── Effective report text ──
    effective_report = report_text.strip() if report_text and report_text.strip() else None

    result = agent.predict(modality=modality, image_path=image,
                           lateral_path=lateral_image if modality=='xray' else None,
                           report_text=effective_report, metadata=meta)

    # ── Result text ──
    pred         = result['prediction']
    used_report  = result['report_text']
    report_src   = result['report_source']
    lines = []
    for w in result.get('warnings', []):
        lines.append(f"{w}\n")

    lines.append(f"**Modality:** {modality.upper()}")
    src_label = "user-provided" if report_src == "user" else ("auto-generated" if report_src == "auto" else "default")
    lines.append(f"**Report ({src_label}):** *{used_report}*\n")

    if result.get('abstained'):
        lines.append("🚫 **Model abstained — prediction unreliable.**")
    elif isinstance(pred, list):
        lines.append(f"**Findings:** {', '.join(pred) if pred != ['Normal'] else 'Normal'}")
    else:
        conf = result.get('confidence')
        if conf is not None:
            warn = "  ⚠️ *Low confidence — recommend clinical review*" if conf < 0.6 else ""
            lines.append(f"**Prediction:** {pred}{warn}")
            lines.append(f"**Confidence:** {conf*100:.1f}%")
        else:
            lines.append(f"**Prediction:** {pred}")
    lines.append(f"**Latency:** {result['latency_ms']} ms\n")
    lines.append("**Probabilities:**")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        lines.append(f"  `{cls:<20}` {prob:.4f}  {bar}")

    # ── Grad-CAM ──
    txt_feat = _bert_feat(used_report)
    meta_t   = _meta_tensor(modality, age, sex, localization, view)
    labels   = {'mri': MRI_CLASSES, 'skin': SKIN_CLASSES, 'xray': XRAY_CLASSES}[modality]

    cam_img  = None
    hdc_text = "Skipped — model abstained or OOD detected."

    if not result.get('abstained'):
        if isinstance(pred, list):
            probs_arr = np.array([result['probabilities'][l] for l in XRAY_CLASSES])
            pred_idx  = int(probs_arr.argmax())
        else:
            pred_idx = labels.index(pred) if pred in labels else 0

        try:
            cam_img = make_gradcam_overlay(image, modality, txt_feat, meta_t, pred_idx,
                                           lateral_path=lateral_image if modality=='xray' else None)
        except Exception:
            cam_img = None

        # ── HDC retrieval ──
        try:
            hits = hdc_retrieve(modality, image, txt_feat, meta_t,
                                lateral_path=lateral_image if modality=='xray' else None)
            hdc_lines = ["**Top-3 Similar Training Cases (HDC):**\n"]
            for i, h in enumerate(hits, 1):
                label = h.get('dx', h['label'])
                id_   = h.get('id', '—')
                match = "✅" if label == (pred[0] if isinstance(pred, list) else pred) else "❌"
                hdc_lines.append(f"  {i}. {match} `{id_}` → **{label}** (distance: {h['distance']})")
            hdc_text = "\n".join(hdc_lines)
        except Exception as e:
            hdc_text = f"HDC retrieval unavailable: {e}"

    # ── Update history ──
    import os
    conf_val = result.get('confidence')
    _history.insert(0, {
        'modality':   modality.upper(),
        'file':       os.path.basename(image),
        'prediction': ', '.join(pred) if isinstance(pred, list) else pred,
        'confidence': f"{conf_val*100:.1f}%" if conf_val is not None else '—',
        'latency':    f"{result['latency_ms']} ms",
    })

    return "\n".join(lines), json.dumps(result, indent=2), cam_img, hdc_text, _render_history()


def _render_history():
    if not _history:
        return "No inferences yet."
    rows = ["| # | Modality | File | Prediction | Confidence | Latency |",
            "|---|----------|------|------------|------------|---------|"]
    for i, h in enumerate(_history, 1):
        rows.append(f"| {i} | {h['modality']} | `{h['file']}` | {h['prediction']} | {h['confidence']} | {h['latency']} |")
    return "\n".join(rows)


def update_visibility(modality):
    return (
        gr.update(visible=modality == 'xray'),
        gr.update(visible=modality == 'skin'),
        gr.update(visible=modality == 'xray'),
        gr.update(visible=modality in ('skin','mri')),
        gr.update(visible=modality in ('skin','mri')),
    )

# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Clinical AI Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Tri-Modal Clinical AI Agent\nCPU-only · ONNX · Grad-CAM · HDC Retrieval")

    with gr.Row():
        modality = gr.Radio(["xray","skin","mri"], value="mri", label="Modality")

    with gr.Row():
        with gr.Column(scale=1):
            image       = gr.Image(type="filepath", label="Image")
            lateral     = gr.Image(type="filepath", label="Lateral View (X-Ray only)", visible=False)
            report_text = gr.Textbox(label="Clinical Report (optional — auto-generated if empty)", lines=2)
            with gr.Row():
                age = gr.Slider(1, 100, value=45, step=1, label="Age",  visible=True)
                sex = gr.Radio(["M","F"], value="M",                     label="Sex", visible=True)
            localization = gr.Dropdown(SKIN_LOCS, value="back", label="Localization (Skin)", visible=False)
            view         = gr.Radio(["PA","AP"], value="PA", label="View Position (X-Ray)", visible=False)
            btn          = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Result")
            hdc_out     = gr.Markdown(label="HDC Retrieval")
            cam_image   = gr.Image(label="Grad-CAM", type="numpy")
            output_json = gr.Code(label="Raw JSON", language="json")

    with gr.Accordion("📋 Session History", open=False):
        history_md = gr.Markdown(value="No inferences yet.")

    modality.change(update_visibility, modality, [lateral, localization, view, age, sex])
    btn.click(predict,
              inputs=[modality, image, lateral, report_text, age, sex, localization, view],
              outputs=[output_text, output_json, cam_image, hdc_out, history_md])

if __name__ == "__main__":
    demo.launch()
