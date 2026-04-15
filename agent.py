"""
agent.py — Tri-Modal Clinical AI Agent
Runs all three ONNX INT8 models on CPU. No PyTorch required at inference time.

Usage:
    from agent import ClinicalAgent
    agent = ClinicalAgent()

    # X-Ray (dual-view)
    result = agent.predict(
        modality="xray",
        image_path="frontal.jpg",
        lateral_path="lateral.jpg",   # xray only
        report_text="Cardiomegaly noted.",
        metadata={"view": "PA"}       # view: PA or AP
    )

    # Skin
    result = agent.predict(
        modality="skin",
        image_path="lesion.jpg",
        metadata={"age": 45, "sex": "M", "localization": "back"}
    )

    # MRI
    result = agent.predict(
        modality="mri",
        image_path="brain.jpg",
        metadata={"age": 55, "sex": "M"}
    )

    print(result)
    # {
    #   "modality": "mri",
    #   "prediction": "glioma",
    #   "confidence": 0.97,
    #   "probabilities": {"glioma": 0.97, "meningioma": 0.02, ...},
    #   "latency_ms": 280.4
    # }
"""

import os
import time
import random
import hashlib
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BERT_INT8  = os.path.join(BASE_DIR, "bert_int8.onnx")
SKIN_INT8  = os.path.join(BASE_DIR, "skin_int8.onnx")
MRI_INT8   = os.path.join(BASE_DIR, "mri_int8.onnx")
XRAY_INT8  = os.path.join(BASE_DIR, "xray_int8.onnx")
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# ── Label maps ─────────────────────────────────────────────────────────────────
XRAY_LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
                "Effusion","Emphysema","Pneumonia","Pneumothorax"]
SKIN_LABELS = ["benign", "malignant"]
MRI_LABELS  = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Per-class thresholds for X-Ray (tuned for rare classes) ───────────────────
XRAY_THRESHOLDS = {
    "Atelectasis": 0.45, "Cardiomegaly": 0.50, "Consolidation": 0.40,
    "Edema": 0.45, "Effusion": 0.45, "Emphysema": 0.35,
    "Pneumonia": 0.35, "Pneumothorax": 0.45,
}

# ── OOD / confidence constants ─────────────────────────────────────────────────
_OOD_ENTROPY_THRESHOLD  = 0.85   # normalized entropy above this → OOD
_CONFIDENCE_FLOOR       = 0.55   # below this → model abstains

# Skin localization one-hot (17-dim, same order as training)
SKIN_LOCS = ["abdomen","acral","back","chest","ear","face","foot",
             "genital","hand","lower extremity","neck","scalp",
             "trunk","unknown","upper extremity",""]
# 17th dim = "other/missing"

# ── Image preprocessing (ImageNet stats, 224×224) ──────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocess(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)[None]   # (1,3,224,224)


# ── Metadata encoders ──────────────────────────────────────────────────────────
def _skin_meta(age, sex, localization) -> np.ndarray:
    """Returns (1, 17): [age_norm, sex_bin, loc_onehot×15]"""
    age_n = float(age) / 80.0
    sex_b = 1.0 if str(sex).upper().startswith("M") else 0.0
    loc   = str(localization).lower().strip()
    loc_oh = np.zeros(15, dtype=np.float32)
    if loc in SKIN_LOCS[:15]:
        loc_oh[SKIN_LOCS.index(loc)] = 1.0
    return np.array([[age_n, sex_b, *loc_oh]], dtype=np.float32)

def _mri_meta(age, sex) -> np.ndarray:
    """Returns (1, 2): [age_norm, sex_bin]"""
    age_n = float(age) / 80.0
    sex_b = 1.0 if str(sex).upper().startswith("M") else 0.0
    return np.array([[age_n, sex_b]], dtype=np.float32)

def _xray_meta(view: str) -> np.ndarray:
    """Returns (1, 2): one-hot [PA, AP]"""
    v = str(view).upper()
    return np.array([[1.0, 0.0] if v == "PA" else [0.0, 1.0]], dtype=np.float32)


# ── Report generators (ported from scripts, no CSV dependency) ─────────────────

_SKIN_SHAPE    = ["The lesion is asymmetric along one or both axes.","A symmetric round to oval lesion is identified.","The lesion outline is irregular with a multicomponent pattern.","Symmetry is maintained along both axes of the lesion.","Mild asymmetry is noted in the color distribution.","Structural irregularity is noted on dermoscopic evaluation.","The lesion is well-circumscribed with a regular outline.","A raised lesion with irregular borders is present.","The lesion has a nodular morphology.","The lesion is dome-shaped with a smooth surface."]
_SKIN_BORDER   = ["Borders are poorly defined with notching at the periphery.","Borders are smooth and well-circumscribed.","Peripheral border shows abrupt cutoff in multiple segments.","Well-defined borders are noted without notching.","The lesion demonstrates poorly circumscribed margins.","A sharply demarcated border is present throughout.","Irregular margins are present with peripheral irregularity.","The lesion fades gradually into the surrounding skin.","Borders are irregular with focal thickening at the margin.","Sharp peripheral demarcation is present."]
_SKIN_COLOR    = ["Variegated pigmentation is present with brown and gray hues.","Homogeneous tan-brown pigmentation is present throughout.","Heterogeneous color distribution is identified.","Uniform brown coloration without focal color variation.","Multiple color components including tan and dark brown are present.","Polychromatic coloration is noted across the lesion surface.","Pink-red background with surface scale is present.","Focal areas of hypopigmentation are noted.","Brown to dark brown homogeneous coloration is present.","Erythematous base with overlying surface changes is noted."]
_SKIN_STRUCT   = ["A pigment network is appreciated across the lesion surface.","Dotted vessels are distributed within the lesion.","Globules are distributed within the lesion.","Regression structures are present centrally.","White structureless areas are present within the lesion.","Surface scale is noted overlying the lesion.","Milia-like cysts are scattered throughout the lesion.","Follicular openings are identified within the lesion.","Lacunae of varying sizes are distributed throughout.","Crystalline structures are visible under polarized dermoscopy.","Irregular globules are distributed within the lesion.","A reticular pattern is appreciated across the lesion."]
_SKIN_NEG      = ["No ulceration is identified.","No atypical vascular structures are identified.","No regression structures are present.","No blue-white veil is identified.","No pseudopods or radial streaming at the periphery.","No irregular border segments are noted.","No multicomponent pattern is identified.","No surface scale is identified.","No focal hypopigmentation is noted.","No irregular pigment network is appreciated."]
_SKIN_CLOSING  = ["Clinical correlation with dermoscopic findings is recommended.","Further evaluation is advised based on clinical context.","Dermatologic assessment is recommended for definitive management.","Short-term follow-up imaging is suggested to assess interval change.","Specialist review is recommended for definitive clinical management.","Correlation with patient history and clinical examination is advised.","Periodic skin surveillance is recommended based on overall risk profile."]
_SKIN_LOCS_PH  = {"face":["on the face","involving the facial skin"],"scalp":["on the scalp","involving the scalp"],"back":["on the back","on the posterior trunk"],"chest":["on the chest","on the anterior chest wall"],"acral":["on an acral surface","involving the acral skin"],"ear":["on the ear","involving the auricular region"],"lower extremity":["on the lower leg","on the lower extremity"],"upper extremity":["on the upper arm","on the upper extremity"],"neck":["on the neck","involving the neck"],"foot":["on the foot","involving the plantar surface"],"hand":["on the hand","involving the dorsal hand"],"abdomen":["on the abdomen","involving the abdominal skin"],"trunk":["on the trunk","involving the truncal skin"],"unknown":["at an unspecified site","at an undocumented anatomical site"]}

def _generate_skin_report(age, sex, localization, image_id="default") -> str:
    seed = int(hashlib.md5(str(image_id).encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    sex_str = "male" if str(sex).upper().startswith("M") else "female"
    loc     = str(localization).lower().strip()
    loc_ph  = rng.choice(_SKIN_LOCS_PH.get(loc, [f"on the {loc}"]))
    demo    = f"A {age}-year-old {sex_str} presenting with a lesion {loc_ph}."
    body    = f"{rng.choice(_SKIN_SHAPE)} {rng.choice(_SKIN_BORDER)} {rng.choice(_SKIN_COLOR)} {rng.choice(_SKIN_STRUCT)} {rng.choice(_SKIN_NEG)}"
    return f"{demo} {body} {rng.choice(_SKIN_CLOSING)}"


_MRI_SIGNAL    = ["A focal area of heterogeneous signal intensity is identified.","A region of mixed signal is noted on multisequence imaging.","Focal signal alteration is present on T2-weighted sequences.","An area of intermediate signal intensity is identified.","No focal signal abnormality is identified on any sequence.","Brain parenchyma demonstrates normal signal intensity throughout.","Normal gray-white matter differentiation is maintained."]
_MRI_LOCATION  = ["The finding is located in the frontal lobe region.","The abnormality is centered in the temporal lobe.","The finding is situated in the posterior fossa.","The lesion appears to arise from the convexity region.","The abnormality is identified in the parietal lobe.","The lesion is identified in the sellar and parasellar region.","No focal intracranial lesion is identified.","The brain parenchyma appears unremarkable in signal and morphology."]
_MRI_MORPH     = ["The lesion demonstrates well-defined margins with a rounded morphology.","The mass exhibits irregular borders and heterogeneous internal architecture.","The lesion appears ovoid with relatively smooth margins.","The lesion is characterized by a solid component.","No discrete mass lesion or focal morphological abnormality is identified.","Cortical and subcortical structures appear morphologically intact."]
_MRI_ENHANCE   = ["Following contrast administration, the lesion demonstrates heterogeneous enhancement.","Post-contrast imaging reveals peripheral rim enhancement.","No abnormal enhancement is identified following contrast administration.","Post-contrast imaging demonstrates no pathological enhancement.","Avid homogeneous enhancement is identified on post-contrast sequences."]
_MRI_MASS      = ["Mild local mass effect with sulcal effacement is noted.","No significant mass effect or midline shift is demonstrated.","Moderate perilesional edema is present contributing to local mass effect.","No evidence of herniation or ventricular obstruction is identified.","Ventricular system appears normal in size and configuration."]
_MRI_CLOSING   = ["Clinical correlation with neuroimaging findings is recommended.","Neurosurgical consultation is recommended for definitive management.","Short-interval follow-up MRI is suggested to assess interval change.","Histopathological confirmation is recommended for definitive diagnosis.","Multidisciplinary team review is recommended for treatment planning.","The treating clinician should correlate findings with the patient's history."]
_MRI_SIGNAL_N  = [4, 5, 6]
_MRI_LOC_N     = [6, 7]
_MRI_MORPH_N   = [4, 5]
_MRI_ENHANCE_N = [2, 3]
_MRI_MASS_N    = [1, 3, 4]

def _mri_weighted(pool, normal_idx, rng, p_normal):
    if rng.random() < p_normal:
        return pool[rng.choice(normal_idx)]
    abnormal = [i for i in range(len(pool)) if i not in normal_idx]
    return pool[rng.choice(abnormal)]

def _generate_mri_report(age, sex, image_id="default") -> str:
    seed = int(hashlib.md5(str(image_id).encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    sex_str = "male" if str(sex).upper().startswith("M") else "female"
    p_normal = 0.4   # neutral — we don't know the class at inference time
    demo    = f"Brain MRI performed on a {age}-year-old {sex_str} patient."
    signal  = _mri_weighted(_MRI_SIGNAL,   _MRI_SIGNAL_N,   rng, p_normal)
    loc     = _mri_weighted(_MRI_LOCATION, _MRI_LOC_N,      rng, p_normal)
    morph   = _mri_weighted(_MRI_MORPH,    _MRI_MORPH_N,    rng, p_normal)
    enhance = _mri_weighted(_MRI_ENHANCE,  _MRI_ENHANCE_N,  rng, p_normal)
    mass    = _mri_weighted(_MRI_MASS,     _MRI_MASS_N,     rng, p_normal)
    return f"{demo} {signal} {loc} {morph} {enhance} {mass} {rng.choice(_MRI_CLOSING)}"


def _default_report(modality: str) -> str:
    """Fallback only used for X-Ray (no structured report generator for IU X-Ray)."""
    return "Chest radiograph obtained for clinical evaluation."


# ── Model registry (lazy load / unload) ────────────────────────────────────────
class _ModelRegistry:
    """Loads one fusion model at a time to minimise RAM on edge devices."""
    def __init__(self, opts, providers):
        self._opts      = opts
        self._providers = providers
        self._cache     = {}   # modality → session

    def get(self, modality: str) -> ort.InferenceSession:
        if modality not in self._cache:
            paths = {"skin": SKIN_INT8, "mri": MRI_INT8, "xray": XRAY_INT8}
            self._cache[modality] = ort.InferenceSession(
                paths[modality], self._opts, providers=self._providers)
        return self._cache[modality]

    def unload(self, modality: str):
        self._cache.pop(modality, None)


# ── OOD detection ──────────────────────────────────────────────────────────────
def _is_ood(probs: dict) -> bool:
    """Normalized entropy > threshold means model is uniformly uncertain → OOD."""
    p = np.array(list(probs.values()), dtype=np.float32)
    entropy = -np.sum(p * np.log(p + 1e-8))
    max_entropy = np.log(len(p))
    return (entropy / max_entropy) > _OOD_ENTROPY_THRESHOLD


# ── Modality mismatch (image-level heuristic) ──────────────────────────────────
def _check_image_modality(image_path: str, modality: str) -> list:
    """Returns list of warning strings (empty = no issues detected)."""
    img = np.array(Image.open(image_path).convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    channel_diff = np.mean([np.abs(r-g).mean(), np.abs(g-b).mean(), np.abs(r-b).mean()])
    is_grayscale = channel_diff < 12   # MRI/Xray: near-zero diff; skin: 20-50+

    mean_intensity = img.mean()
    # MRI/Xray have lots of black background → mean can be as low as 10-15
    # Skin dermoscopy is brighter overall but vignette pulls it down → typically 40-180
    # Flag only extremes that no real medical image would have
    intensity_ok = mean_intensity > 5   # catches pure-black/corrupt images only

    warns = []
    if modality == "skin" and is_grayscale:
        warns.append("⚠️ Image looks grayscale — skin dermoscopy images are typically colorful.")
    elif modality in ("mri", "xray") and not is_grayscale:
        warns.append(f"⚠️ Image looks colorful — {modality.upper()} scans are typically grayscale.")
    if not intensity_ok:
        warns.append("⚠️ Image appears nearly black — may be corrupt or invalid.")
    return warns


# ── Agent ──────────────────────────────────────────────────────────────────────
class ClinicalAgent:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        print("Loading BERT tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

        print("Loading BERT ONNX session...")
        self._bert     = ort.InferenceSession(BERT_INT8, opts, providers=providers)
        self._registry = _ModelRegistry(opts, providers)
        print("Agent ready.")

    # ── BERT encoding ──────────────────────────────────────────────────────────
    def _encode_text(self, text: str) -> np.ndarray:
        enc = self._tokenizer(text, return_tensors="np", max_length=128,
                              padding="max_length", truncation=True)
        out = self._bert.run(None, {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        return out[0].astype(np.float32)   # (1, 768)

    # ── Main predict ───────────────────────────────────────────────────────────
    def predict(self, modality: str, image_path: str,
                lateral_path: str = None,
                report_text: str  = None,
                metadata: dict    = None) -> dict:

        modality = modality.lower()
        assert modality in ("xray", "skin", "mri"), f"Unknown modality: {modality}"
        metadata = metadata or {}

        # ── Modality mismatch check (before inference) ─────────────────────────
        image_warnings = _check_image_modality(image_path, modality)

        t0 = time.perf_counter()

        # ── Auto-generate report if not provided ───────────────────────────────
        if report_text and report_text.strip():
            text = report_text.strip()
            report_source = "user"
        elif modality == "skin":
            text = _generate_skin_report(
                age=metadata.get("age", 45),
                sex=metadata.get("sex", "U"),
                localization=metadata.get("localization", "unknown"),
                image_id=os.path.basename(image_path),
            )
            report_source = "auto"
        elif modality == "mri":
            text = _generate_mri_report(
                age=metadata.get("age", 45),
                sex=metadata.get("sex", "U"),
                image_id=os.path.basename(image_path),
            )
            report_source = "auto"
        else:
            text = _default_report(modality)
            report_source = "default"

        txt_ft = self._encode_text(text)

        # Image
        img    = _preprocess(image_path)
        model  = self._registry.get(modality)

        if modality == "skin":
            meta   = _skin_meta(metadata.get("age", 45),
                                metadata.get("sex", "U"),
                                metadata.get("localization", "unknown"))
            logits = model.run(None, {"image": img, "text_feat": txt_ft, "metadata": meta})[0]
            prob   = float(1 / (1 + np.exp(-logits[0, 0])))
            pred   = "malignant" if prob >= 0.5 else "benign"
            probs  = {"benign": round(1 - prob, 4), "malignant": round(prob, 4)}

        elif modality == "mri":
            meta      = _mri_meta(metadata.get("age", 45), metadata.get("sex", "U"))
            logits    = model.run(None, {"image": img, "text_feat": txt_ft, "metadata": meta})[0]
            probs_arr = _softmax(logits[0])
            pred      = MRI_LABELS[int(np.argmax(probs_arr))]
            probs     = {k: round(float(v), 4) for k, v in zip(MRI_LABELS, probs_arr)}

        else:  # xray
            lat    = _preprocess(lateral_path) if lateral_path else img
            meta   = _xray_meta(metadata.get("view", "PA"))
            logits = model.run(None, {"frontal": img, "lateral": lat,
                                      "text_feat": txt_ft, "metadata": meta})[0]
            sigs   = _sigmoid(logits[0])
            probs  = {k: round(float(v), 4) for k, v in zip(XRAY_LABELS, sigs)}
            positives = sorted([(k, v) for k, v in probs.items() if v >= XRAY_THRESHOLDS[k]],
                               key=lambda x: -x[1])
            pred = [k for k, _ in positives] if positives else ["Normal"]

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── OOD check ──────────────────────────────────────────────────────────
        if modality != "xray" and _is_ood(probs):
            return {
                "modality":   modality,
                "prediction": "UNCERTAIN",
                "confidence": None,
                "probabilities": probs,
                "latency_ms": latency_ms,
                "warnings":   image_warnings + ["⚠️ OOD detected: model output is near-uniform. Image may not match selected modality."],
                "abstained":  True,
            }

        # ── Confidence floor (non-xray only) ───────────────────────────────────
        confidence = max(probs.values()) if modality != "xray" else None
        abstained  = False
        if modality != "xray" and confidence < _CONFIDENCE_FLOOR:
            pred      = "UNCERTAIN"
            abstained = True
            image_warnings.append(
                f"⚠️ Confidence {confidence*100:.1f}% is below threshold — model abstains. Please review manually."
            )

        result = {
            "modality":      modality,
            "prediction":    pred,
            "probabilities": probs,
            "latency_ms":    latency_ms,
            "report_text":   text,
            "report_source": report_source,   # "user" | "auto" | "default"
        }
        if modality != "xray":
            result["confidence"] = round(confidence, 4)
        if abstained:
            result["abstained"] = True
        if image_warnings:
            result["warnings"] = image_warnings

        return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    agent = ClinicalAgent()

    if len(sys.argv) < 3:
        print("Usage: python agent.py <modality> <image_path> [lateral_path]")
        sys.exit(1)

    modality   = sys.argv[1]
    image_path = sys.argv[2]
    lateral    = sys.argv[3] if len(sys.argv) > 3 else None

    result = agent.predict(modality=modality, image_path=image_path,
                           lateral_path=lateral)
    import json
    print(json.dumps(result, indent=2))
