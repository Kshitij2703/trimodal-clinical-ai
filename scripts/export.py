"""
export.py
Exports ClinicalBERT + all three fusion models to ONNX with INT8 quantization.
Run on Kaggle where all .pth files are saved.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import onnxruntime as ort
import time

OUT_DIR    = '/kaggle/working/onnx_models'
BERT_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared fusion block (all three models use this structure) ──────────────────
class FusionModel(nn.Module):
    def __init__(self, vision_dim, text_dim, meta_dim,
                 v_emb, t_emb, m_emb, bottleneck, n_classes,
                 dv, dt, df):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.vision_encoder = eff.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(nn.Linear(vision_dim, v_emb), nn.BatchNorm1d(v_emb), nn.ReLU(), nn.Dropout(dv))
        self.text_proj      = nn.Sequential(nn.Linear(text_dim, t_emb),   nn.BatchNorm1d(t_emb), nn.ReLU(), nn.Dropout(dt))
        self.metadata_proj  = nn.Sequential(nn.Linear(meta_dim, m_emb), nn.ReLU())
        self.fusion         = nn.Sequential(nn.Linear(v_emb+t_emb+m_emb, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(), nn.Dropout(df))
        self.classifier     = nn.Linear(bottleneck, n_classes)
        if hasattr(self, 'temperature'):
            pass
        self.temperature    = nn.Parameter(torch.ones(1))

    def forward(self, image, text_feat, metadata):
        v = self.vision_proj(self.vision_pool(self.vision_encoder(image)).flatten(1))
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        b = self.fusion(torch.cat([v, t, m], dim=1))
        return self.classifier(b)


# ── Model configs ──────────────────────────────────────────────────────────────
BASE = '/kaggle/input/datasets/kshitijnavale/modals12'
MODELS = {
    'skin': {
        'ckpt':      f'{BASE}/skin_fused (1).pth',
        'meta_dim':  17,
        'n_classes': 1,
        'binary':    True,
    },
    'mri': {
        'ckpt':      f'{BASE}/mri_fused.pth',
        'meta_dim':  2,
        'n_classes': 4,
        'binary':    False,
    },
}

COMMON = dict(vision_dim=1280, text_dim=768, v_emb=256, t_emb=256,
              m_emb=32, bottleneck=128, dv=0.5, dt=0.4, df=0.5)


def load_fusion(cfg):
    m = FusionModel(
        vision_dim=COMMON['vision_dim'], text_dim=COMMON['text_dim'],
        meta_dim=cfg['meta_dim'], v_emb=COMMON['v_emb'], t_emb=COMMON['t_emb'],
        m_emb=COMMON['m_emb'], bottleneck=COMMON['bottleneck'],
        n_classes=cfg['n_classes'], dv=COMMON['dv'], dt=COMMON['dt'], df=COMMON['df']
    )
    ckpt = torch.load(cfg['ckpt'], weights_only=False, map_location='cpu')
    m.load_state_dict(ckpt['state_dict'], strict=False)
    m.eval()
    return m


def export_and_quantize(model, name, meta_dim, binary):
    img   = torch.randn(1, 3, 224, 224)
    txt   = torch.randn(1, 768)
    meta  = torch.randn(1, meta_dim)

    fp32_path = f"{OUT_DIR}/{name}_fp32.onnx"
    int8_path = f"{OUT_DIR}/{name}_int8.onnx"

    torch.onnx.export(
        model, (img, txt, meta), fp32_path,
        input_names=['image', 'text_feat', 'metadata'],
        output_names=['logits'],
        dynamic_axes={'image':{0:'B'}, 'text_feat':{0:'B'}, 'metadata':{0:'B'}, 'logits':{0:'B'}},
        opset_version=14, do_constant_folding=True
    )
    onnx.checker.check_model(onnx.load(fp32_path))
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

    fp32_mb = os.path.getsize(fp32_path) / 1e6
    int8_mb = os.path.getsize(int8_path) / 1e6
    print(f"  {name}: FP32={fp32_mb:.1f}MB → INT8={int8_mb:.1f}MB ({100*(1-int8_mb/fp32_mb):.0f}% reduction)")

    # Latency benchmark
    sess = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
    feeds = {'image': img.numpy(), 'text_feat': txt.numpy(), 'metadata': meta.numpy()}
    for _ in range(5): sess.run(None, feeds)  # warmup
    t0 = time.perf_counter()
    for _ in range(50): sess.run(None, feeds)
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  {name} INT8 latency: {ms:.1f}ms")
    return int8_path


# ── Export BERT ────────────────────────────────────────────────────────────────
print("Exporting ClinicalBERT...")
tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL)
bert       = AutoModel.from_pretrained(BERT_MODEL).eval()

class BERTWrapper(nn.Module):
    def __init__(self, bert): super().__init__(); self.bert = bert
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

bert_wrapper = BERTWrapper(bert)
dummy_enc    = tokenizer("sample text", return_tensors='pt', max_length=128,
                          padding='max_length', truncation=True)
bert_fp32    = f"{OUT_DIR}/bert_fp32.onnx"
bert_int8    = f"{OUT_DIR}/bert_int8.onnx"

torch.onnx.export(
    bert_wrapper,
    (dummy_enc['input_ids'], dummy_enc['attention_mask']),
    bert_fp32,
    input_names=['input_ids', 'attention_mask'],
    output_names=['cls_embedding'],
    dynamic_axes={'input_ids':{0:'B'}, 'attention_mask':{0:'B'}, 'cls_embedding':{0:'B'}},
    opset_version=14
)
onnx.checker.check_model(onnx.load(bert_fp32))
quantize_dynamic(bert_fp32, bert_int8, weight_type=QuantType.QInt8)
print(f"  BERT: FP32={os.path.getsize(bert_fp32)/1e6:.1f}MB → INT8={os.path.getsize(bert_int8)/1e6:.1f}MB")

# Benchmark BERT INT8
sess  = ort.InferenceSession(bert_int8, providers=['CPUExecutionProvider'])
feeds = {'input_ids': dummy_enc['input_ids'].numpy(),
         'attention_mask': dummy_enc['attention_mask'].numpy()}
for _ in range(3): sess.run(None, feeds)
t0 = time.perf_counter()
for _ in range(20): sess.run(None, feeds)
print(f"  BERT INT8 latency: {(time.perf_counter()-t0)/20*1000:.1f}ms")

# ── Export fusion models ───────────────────────────────────────────────────────
print("\nExporting fusion models...")
for name, cfg in MODELS.items():
    print(f"\n[{name}]")
    model = load_fusion(cfg)
    export_and_quantize(model, name, cfg['meta_dim'], cfg['binary'])

# ── Export X-ray model (dual-view: frontal + lateral) ─────────────────────────
print("\n[xray]")

class XRayFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.vision_encoder = eff.features
        self.vision_pool    = nn.AdaptiveAvgPool2d(1)
        self.vision_proj    = nn.Sequential(nn.Linear(1280,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5))
        self.text_proj      = nn.Sequential(nn.Linear(768,256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.metadata_proj  = nn.Sequential(nn.Linear(2,16), nn.ReLU())
        self.fusion         = nn.Sequential(nn.Linear(528,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5))
        self.classifier     = nn.Linear(256, 8)
        self.temperature    = nn.Parameter(torch.ones(1))

    def forward(self, frontal, lateral, text_feat, metadata):
        f = self.vision_pool(self.vision_encoder(frontal)).flatten(1)
        l = self.vision_pool(self.vision_encoder(lateral)).flatten(1)
        v = self.vision_proj((f + l) / 2.0)
        t = self.text_proj(text_feat)
        m = self.metadata_proj(metadata)
        return self.classifier(self.fusion(torch.cat([v, t, m], dim=1)))

xray_model = XRayFusionModel()
xray_ckpt  = torch.load(f'{BASE}/efficientnet_fused.pth', weights_only=False, map_location='cpu')
xray_model.load_state_dict(xray_ckpt['state_dict'] if 'state_dict' in xray_ckpt else xray_ckpt, strict=False)
xray_model.eval()

frontal = torch.randn(1, 3, 224, 224)
lateral = torch.randn(1, 3, 224, 224)
txt     = torch.randn(1, 768)
meta    = torch.randn(1, 2)

xray_fp32 = f"{OUT_DIR}/xray_fp32.onnx"
xray_int8 = f"{OUT_DIR}/xray_int8.onnx"
torch.onnx.export(
    xray_model, (frontal, lateral, txt, meta), xray_fp32,
    input_names=['frontal', 'lateral', 'text_feat', 'metadata'],
    output_names=['logits'],
    dynamic_axes={'frontal':{0:'B'},'lateral':{0:'B'},'text_feat':{0:'B'},'metadata':{0:'B'},'logits':{0:'B'}},
    opset_version=14, do_constant_folding=True
)
onnx.checker.check_model(onnx.load(xray_fp32))
quantize_dynamic(xray_fp32, xray_int8, weight_type=QuantType.QInt8)
print(f"  xray: FP32={os.path.getsize(xray_fp32)/1e6:.1f}MB → INT8={os.path.getsize(xray_int8)/1e6:.1f}MB")

sess  = ort.InferenceSession(xray_int8, providers=['CPUExecutionProvider'])
feeds = {'frontal': frontal.numpy(), 'lateral': lateral.numpy(),
         'text_feat': txt.numpy(), 'metadata': meta.numpy()}
for _ in range(5): sess.run(None, feeds)
t0 = time.perf_counter()
for _ in range(50): sess.run(None, feeds)
print(f"  xray INT8 latency: {(time.perf_counter()-t0)/50*1000:.1f}ms")

print(f"\n✓ All models exported to {OUT_DIR}")
print("Files:", os.listdir(OUT_DIR))
