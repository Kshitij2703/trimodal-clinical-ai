# Tri-Modal Neurosymbolic Clinical AI

A lightweight, CPU-only medical AI agent that performs clinical image classification across three modalities — chest X-Ray, skin dermoscopy, and brain MRI — using a fused vision + language + metadata architecture, with explainability and case-based retrieval built in.

---

## Architecture

All three models share the same backbone design:

```
EfficientNet-B0 (1280 → 256)  ─┐
Bio_ClinicalBERT CLS (768 → 256) ├─→ Fusion bottleneck → Classifier
Metadata MLP (N → 16/32)      ─┘
                                ↓
                    HDC 4096-dim binary hypervectors (retrieval)
```

| Modality | Task | Classes | Metadata |
|---|---|---|---|
| X-Ray | Multi-label pathology | 8 | View position (PA/AP) |
| Skin | Binary malignancy | 2 | Age + sex + localization |
| Brain MRI | Tumor type | 4 | Age + sex |

---

## Results

| Model | Dataset | AUC | Accuracy | F1 Macro |
|---|---|---|---|---|
| X-Ray | IU X-Ray (554 test) | 0.9416 | 90.1% | 0.6964 |
| Skin | HAM10000 | 0.9106 | 84.56% | 0.8037 |
| Brain MRI | Kaggle MRI (720 test) | 0.9998 | 98.75% | 0.99 |

---

## Quickstart

```bash
cd /Users/kshitijnavale/Desktop/Xray

# Run Gradio demo
.venv/bin/python demo.py
# Open: http://localhost:7860

# Run agent directly
.venv/bin/python agent.py mri samples/Te-no_1.jpg
.venv/bin/python agent.py skin samples/ISIC_0024364.jpg
```

---

## Project Structure

```
Xray/
├── agent.py                  # Tri-modal inference agent (main entry point)
├── demo.py                   # Gradio web demo
│
├── bert_int8.onnx            # Shared ClinicalBERT (INT8, used by all 3 models)
├── xray_int8.onnx
├── skin_int8.onnx
├── mri_int8.onnx
│
├── models/                   # PyTorch .pth files (used for Grad-CAM)
│   ├── efficientnet_fused.pth
│   ├── skin_fused.pth
│   └── mri_fused.pth
│
├── retrieval/                # HDC hypervector indices
│   ├── retrieval_index.pkl
│   ├── skin_retrieval_index.pkl
│   └── mri_retrieval_index.pkl
│
├── data/                     # Training CSVs with synthetic reports
│   ├── train.csv / val.csv / test.csv   (X-Ray)
│   ├── ham10000_with_reports 3.csv      (Skin)
│   ├── mri_metadata.csv                 (MRI synthetic demographics)
│   └── mri_with_reports.csv             (MRI with generated reports)
│
├── scripts/                  # Training, export, explainability scripts
│   ├── 1_preprocess_iuxray.py
│   ├── 2_extract_bert_features.py
│   ├── 3_train_end_to_end.py
│   ├── 4_explainability_standalone.py
│   ├── ham.py                # Skin cancer training
│   ├── mri.py                # Brain MRI training
│   ├── report.py             # Skin report generator
│   ├── mri_report.py         # MRI report generator
│   ├── mri_metadata.py       # MRI synthetic metadata generator
│   ├── explain.py            # Skin explainability
│   ├── mri_explain.py        # MRI explainability
│   ├── export.py             # ONNX INT8 export
│   └── deploy_local.py       # Local X-Ray demo (legacy)
│
├── explanations/             # Grad-CAM + HDC explanation outputs
│   ├── xray/
│   ├── skin/
│   └── mri/
│
├── samples/                  # Test images
│   ├── Te-no_1.jpg           # Brain MRI — no tumor
│   ├── Tr-pi_1018.jpg        # Brain MRI — pituitary
│   ├── ISIC_0024306.jpg      # Skin dermoscopy
│   └── ISIC_0024364.jpg      # Skin dermoscopy
│
├── requirements_local.txt
├── README.md
└── REPORT.md
```

---

## Agent API

```python
from agent import ClinicalAgent

agent = ClinicalAgent()

# MRI — report auto-generated from age/sex if not provided
result = agent.predict(
    modality="mri",
    image_path="samples/Te-no_1.jpg",
    metadata={"age": 35, "sex": "M"}
)

# Skin
result = agent.predict(
    modality="skin",
    image_path="samples/ISIC_0024364.jpg",
    metadata={"age": 45, "sex": "M", "localization": "back"}
)

# X-Ray (dual-view)
result = agent.predict(
    modality="xray",
    image_path="frontal.jpg",
    lateral_path="lateral.jpg",
    metadata={"view": "PA"}
)

print(result)
# {
#   "modality": "mri",
#   "prediction": "notumor",
#   "confidence": 0.9336,
#   "probabilities": {...},
#   "latency_ms": 57.5,
#   "report_text": "Brain MRI performed on a 35-year-old male...",
#   "report_source": "auto"
# }
```

### Result fields

| Field | Description |
|---|---|
| `prediction` | Predicted class (or `"UNCERTAIN"` if abstained) |
| `confidence` | Max class probability (None for X-Ray multi-label) |
| `probabilities` | Per-class scores |
| `report_text` | Clinical report fed to BERT |
| `report_source` | `"user"` / `"auto"` / `"default"` |
| `latency_ms` | End-to-end inference time |
| `warnings` | List of modality mismatch / OOD / low-confidence warnings |
| `abstained` | `True` if model refused to predict |

---

## Safety Features

- **OOD detection** — normalized entropy check; near-uniform output → model abstains
- **Confidence floor** — predictions below 55% confidence are flagged as `UNCERTAIN`
- **Modality mismatch** — grayscale vs color check flags wrong image type before inference
- **Per-class X-Ray thresholds** — rare classes (Emphysema, Pneumonia) use lower thresholds (0.35) to improve recall

---

## Inference Performance (CPU, ONNX INT8)

| Component | Latency |
|---|---|
| BERT encoding | ~50ms |
| Fusion model | ~30ms |
| Preprocessing | ~50ms |
| **Total** | **~130ms** |

---

## Known Limitations

- MRI metadata (age/sex) is synthetic — no real patient demographics in the Kaggle dataset
- Skin HDC top-1 retrieval (69.88%) is weaker than MRI (98.19%) due to class imbalance
- X-Ray single-view inference is untested (trained on dual frontal+lateral)
- BERT attention visualization shows structural tokens — expected for frozen ClinicalBERT
