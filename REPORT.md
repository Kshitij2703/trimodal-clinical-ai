# REPORT — Tri-Modal Neurosymbolic Clinical AI

**Project:** Tri-Modal Neurosymbolic Clinical AI Agent  
**Stack:** PyTorch · ONNX INT8 · Bio_ClinicalBERT · EfficientNet-B0 · HDC · Gradio  
**Deployment:** CPU-only edge inference (~130ms/prediction)

---

## 1. Problem Statement

Medical imaging AI is fragmented — most systems are single-modal (image only) and require GPU hardware. This project builds a unified, tri-modal clinical AI agent that:

- Classifies chest X-Ray pathologies (8-class multi-label)
- Classifies skin lesion malignancy (binary, dermoscopy)
- Classifies brain tumor type (4-class MRI)

All three modalities run on a single CPU-only system under 1 second per inference, with explainability and case-based retrieval built in.

---

## 2. Architecture

### Shared Backbone (all three models)

```
EfficientNet-B0 (1280-dim → 256-dim)    ─┐
Bio_ClinicalBERT CLS (768-dim → 256-dim) ├─→ Fusion bottleneck → Classifier
Metadata MLP (N-dim → 16/32-dim)        ─┘
                                          ↓
                          HDC 4096-dim binary hypervectors
```

### Per-Modality Differences

| Modality | Metadata input | Fusion dim | Output | Loss |
|---|---|---|---|---|
| X-Ray | View position (2-dim → 16-dim) | 528 → 256 | 8-class sigmoid | Focal BCE |
| Skin | Age + sex + localization (17-dim → 32-dim) | 544 → 128 | 1-class sigmoid | Focal BCE |
| Brain MRI | Age + sex (2-dim → 32-dim) | 544 → 128 | 4-class softmax | CrossEntropy |

### Text Branch

All three models share a single `emilyalsentzer/Bio_ClinicalBERT` encoder. Clinical reports are either user-provided or auto-generated at inference time from patient metadata using template-based generators (ported from `scripts/report.py` and `scripts/mri_report.py`). Reports use shared observation pools — no class-exclusive sentences — to prevent BERT from trivially classifying by text alone.

### HDC Retrieval

Each model encodes its bottleneck embedding (128/256-dim) into a 4096-dim binary hypervector via a random projection matrix. At inference, the query hypervector is compared against all training hypervectors using Hamming distance, returning the top-3 most similar training cases.

---

## 3. Datasets

| Model | Dataset | Size | Split strategy |
|---|---|---|---|
| X-Ray | Indiana University Chest X-Ray (IU X-Ray) | ~7,000 images | Standard train/val/test |
| Skin | HAM10000 (ISIC archive) | 10,015 dermoscopy images | Split by `lesion_id` to prevent patient-level leakage |
| Brain MRI | Kaggle Brain Tumor MRI Dataset | 7,200 images (4 classes) | Standard train/test |

---

## 4. Training Details

### X-Ray
- End-to-end training (no frozen backbone)
- Focal Loss (γ=2.0) + confidence penalty + label smoothing 0.1
- Gradient accumulation (effective batch 64)
- Composite early stopping (AUC − overconfidence − underfit penalties)
- Post-training temperature scaling (T=1.4415)
- Dominant class weights forced to 1.0

### Skin Cancer
- Split by `lesion_id` — prevents patient-level data leakage
- Focal BCE (γ=1.0, lowered from 2.0 to reduce aggressive minority focus)
- WeightedRandomSampler for 80/20 class imbalance
- Temperature calibration (T=1.30) post-training
- Shared observation pools in report generator — text-only AUC (0.7893) ≈ metadata-only AUC (0.7846) confirms zero synthetic leakage

### Brain MRI
- Synthetic metadata (age/sex) from published CBTRUS 2024 epidemiological distributions:
  - Glioma: μ=55, 60% male
  - Meningioma: μ=58, 70% female
  - Pituitary: μ=40, 50/50
  - No tumor: μ=35, 50/50
- CrossEntropy with label smoothing 0.1
- HDC extraction fixed to use `shuffle=False` loader for deterministic index
- Text-only AUC (0.7396) < metadata-only AUC (0.7626) — text is clean, no leakage

---

## 5. Results

### X-Ray (IU X-Ray, 554 test samples)

| Metric | Value |
|---|---|
| Mean AUC | 0.9416 |
| Accuracy | 90.14% |
| F1 Macro | 0.6964 |
| F1 Micro | 0.8217 |
| F1 Weighted | 0.8353 |
| Precision Macro | 0.6235 |
| Recall Macro | 0.8278 |
| Temperature | 1.4415 |

**Per-class AUC:**

| Class | AUC |
|---|---|
| Cardiomegaly | 0.9686 |
| Edema | 0.9558 |
| Atelectasis | 0.9506 |
| Pneumonia | 0.9387 |
| Emphysema | 0.9374 |
| Effusion | 0.9373 |
| Pneumothorax | 0.9354 |
| Consolidation | 0.9088 |

### Skin Cancer (HAM10000)

| Metric | Value |
|---|---|
| AUC | 0.9106 |
| Accuracy | 84.56% |
| Malignant Precision | 0.5592 |
| Malignant Recall | 0.9829 |
| Benign Precision | 0.9949 |
| Benign Recall | 0.8124 |
| F1 Macro | 0.8037 |
| F1 Micro | 0.8456 |
| F1 Weighted | 0.8591 |
| HDC top-1 match | 69.88% |
| Per-class AUC (mel) | 0.9999 |
| Per-class AUC (bcc) | 0.9999 |
| Per-class AUC (akiec) | 0.9999 |

### Brain MRI (Kaggle, 720 test samples)

| Metric | Value |
|---|---|
| AUC | 0.9998 |
| Accuracy | 98.75% |
| F1 Macro | 0.99 |
| HDC top-1 match | 98.19% |
| Errors | 9/720 (all at glioma↔meningioma or notumor↔pituitary boundaries) |

---

## 6. ONNX INT8 Export

All three fusion models and the shared ClinicalBERT were exported to ONNX and quantized to INT8 for CPU-only edge inference.

| File | Size | Purpose |
|---|---|---|
| `bert_int8.onnx` | ~108MB | Shared text encoder (all 3 models) |
| `xray_int8.onnx` | ~18MB | X-Ray fusion model |
| `skin_int8.onnx` | ~18MB | Skin fusion model |
| `mri_int8.onnx` | ~18MB | MRI fusion model |

**Inference latency (CPU, ONNX INT8):**

| Component | Latency |
|---|---|
| BERT encoding | ~50ms |
| Fusion model | ~30ms |
| Preprocessing | ~50ms |
| **Total** | **~130ms** |

---

## 7. Agent Design

The `ClinicalAgent` class in `agent.py` orchestrates all three models:

- **Model registry** — lazy load/unload per modality; only BERT is loaded at startup (~108MB). Fusion models (~18MB each) load on first use.
- **Auto report generation** — if no `report_text` is provided, a clinical report is generated from `age + sex + localization/image_id` using the same template pools used during training. This ensures BERT receives a meaningful input rather than a generic placeholder.
- **OOD detection** — normalized entropy of output probabilities. If entropy > 85% of maximum (near-uniform output), the model abstains and returns `prediction: "UNCERTAIN"`.
- **Confidence floor** — predictions below 55% confidence are flagged as `UNCERTAIN` with a warning.
- **Modality mismatch detection** — checks image color channel difference and mean intensity before inference. Flags colorful images fed to MRI/X-Ray models and grayscale images fed to the skin model.
- **Per-class X-Ray thresholds** — replaces flat 0.5 threshold with per-class tuned values (Emphysema: 0.35, Pneumonia: 0.35) to improve recall on rare classes.

### Result schema

```json
{
  "modality": "mri",
  "prediction": "notumor",
  "confidence": 0.9336,
  "probabilities": {"glioma": 0.0179, "meningioma": 0.0185, "notumor": 0.9336, "pituitary": 0.0300},
  "latency_ms": 57.5,
  "report_text": "Brain MRI performed on a 35-year-old male patient...",
  "report_source": "auto",
  "warnings": []
}
```

---

## 8. Explainability

Each prediction supports a four-panel explanation:

1. **Grad-CAM overlay** — highlights image regions driving the prediction (hooks into `vision_encoder[8]` of EfficientNet-B0)
2. **BERT attention** — top attended tokens in the clinical report
3. **Class probability bars** — per-class confidence visualization
4. **HDC retrieval** — top-3 most similar training cases by Hamming distance

Note: BERT attention visualization shows structural/punctuation tokens dominating — expected behavior for frozen ClinicalBERT. Semantic contribution is captured in the CLS embedding, not individual token attention weights.

---

## 9. Gradio Demo

`demo.py` provides a single-page web UI:

- Modality selector (X-Ray / Skin / MRI)
- Image upload + optional lateral view (X-Ray)
- Optional clinical report text (auto-generated if empty)
- Age, sex, localization/view metadata fields
- Output: prediction + confidence + probability bars + Grad-CAM + HDC top-3 + session history

Run: `.venv/bin/python demo.py` → `http://localhost:7860`

---

## 10. Safety & Validation

| Check | Implementation |
|---|---|
| OOD detection | Normalized entropy > 0.85 → abstain |
| Confidence floor | < 55% → `UNCERTAIN` |
| Modality mismatch | Channel diff + intensity check pre-inference |
| Data leakage (skin) | Split by `lesion_id`; text-only AUC ≈ metadata-only AUC |
| Data leakage (MRI) | Text-only AUC < metadata-only AUC; no diagnosis keywords in reports |
| Calibration | Temperature scaling applied post-training (X-Ray T=1.44, Skin T=1.30) |
| Rare class recall | Per-class thresholds for X-Ray (Emphysema/Pneumonia at 0.35) |

---

## 11. Known Limitations

- MRI metadata (age/sex) is fully synthetic — no real patient demographics available in the Kaggle dataset
- Skin HDC top-1 retrieval (69.88%) is significantly weaker than MRI (98.19%) due to class imbalance in the HDC index
- X-Ray model was trained on dual-view (frontal + lateral); single-view inference is untested
- BERT attention visualization shows structural tokens — frozen encoder behavior, not a bug
- No real-world clinical validation or comparison against radiologist baselines

---

## 12. File Reference

| File | Purpose |
|---|---|
| `agent.py` | Main inference agent — all three modalities |
| `demo.py` | Gradio web demo |
| `scripts/3_train_end_to_end.py` | X-Ray end-to-end training |
| `scripts/ham.py` | Skin cancer training |
| `scripts/mri.py` | Brain MRI training |
| `scripts/report.py` | Skin report template generator |
| `scripts/mri_report.py` | MRI report template generator |
| `scripts/mri_metadata.py` | Synthetic MRI demographic generator |
| `scripts/export.py` | ONNX INT8 export for all models |
| `scripts/explain.py` | Skin Grad-CAM + HDC explainability |
| `scripts/mri_explain.py` | MRI Grad-CAM + HDC explainability |
| `scripts/4_explainability_standalone.py` | X-Ray explainability |
| `scripts/1_preprocess_iuxray.py` | IU X-Ray dataset preprocessing |
| `scripts/2_extract_bert_features.py` | Pre-extract BERT features for X-Ray training |
| `scripts/deploy_local.py` | Legacy local X-Ray demo |
