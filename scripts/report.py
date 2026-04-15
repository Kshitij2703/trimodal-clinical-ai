import pandas as pd
import numpy as np
import os
import random
import hashlib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
META_PATH   = f"{BASE}/HAM10000_metadata.csv"
IMAGE_DIR_1 = f"{BASE}/HAM10000_images_part_1"
IMAGE_DIR_2 = f"{BASE}/HAM10000_images_part_2"
OUTPUT_CSV  = "/kaggle/working/ham10000_with_reports.csv"

DX_FULL = {
    "mel"   : "Melanoma",
    "nv"    : "Melanocytic Nevi",
    "bcc"   : "Basal Cell Carcinoma",
    "akiec" : "Actinic Keratosis / Intraepithelial Carcinoma",
    "bkl"   : "Benign Keratosis-like Lesion",
    "df"    : "Dermatofibroma",
    "vasc"  : "Vascular Lesion",
}

MALIGNANCY = {
    "mel"   : 1,
    "nv"    : 0,
    "bcc"   : 1,
    "akiec" : 1,
    "bkl"   : 0,
    "df"    : 0,
    "vasc"  : 0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED OBSERVATION POOLS — no sentence is class-exclusive.
# Every sentence is plausible for multiple dx classes.
# The model cannot classify from text alone; it must use image + metadata.
# ═══════════════════════════════════════════════════════════════════════════════

SHAPE_POOL = [
    "The lesion is asymmetric along one or both axes.",
    "A symmetric round to oval lesion is identified.",
    "The lesion outline is irregular with a multicomponent pattern.",
    "Symmetry is maintained along both axes of the lesion.",
    "Mild asymmetry is noted in the color distribution.",
    "The lesion demonstrates a symmetric global pattern.",
    "Structural irregularity is noted on dermoscopic evaluation.",
    "The lesion is well-circumscribed with a regular outline.",
    "Asymmetry is present along the horizontal axis.",
    "A slightly raised lesion with irregular contour is identified.",
    "The lesion is flat with a regular symmetric outline.",
    "A raised lesion with irregular borders is present.",
    "The lesion has a nodular morphology.",
    "A flat to slightly raised lesion is identified.",
    "The lesion is dome-shaped with a smooth surface.",
]

BORDER_POOL = [
    "Borders are poorly defined with notching at the periphery.",
    "Borders are smooth and well-circumscribed.",
    "Peripheral border shows abrupt cutoff in multiple segments.",
    "Well-defined borders are noted without notching.",
    "The lesion demonstrates poorly circumscribed margins.",
    "A sharply demarcated border is present throughout.",
    "Irregular margins are present with peripheral irregularity.",
    "The lesion fades gradually into the surrounding skin.",
    "Peripheral borders appear indurated.",
    "The lesion margin is raised at the periphery.",
    "Borders are irregular with focal thickening at the margin.",
    "Sharp peripheral demarcation is present.",
    "The lesion is sharply demarcated with abrupt border transition.",
    "Indistinct borders with surrounding skin changes are noted.",
    "Well-demarcated borders with surface changes are present.",
]

COLOR_POOL = [
    "Variegated pigmentation is present with brown and gray hues.",
    "Homogeneous tan-brown pigmentation is present throughout.",
    "Heterogeneous color distribution is identified.",
    "Uniform brown coloration without focal color variation.",
    "Multiple color components including tan and dark brown are present.",
    "A single shade of brown pigmentation is distributed evenly.",
    "Polychromatic coloration is noted across the lesion surface.",
    "Diffuse regular pigmentation without multicolor components.",
    "Pink-red background with surface scale is present.",
    "Focal areas of hypopigmentation are noted.",
    "Brown to dark brown homogeneous coloration is present.",
    "Erythematous base with overlying surface changes is noted.",
    "Central hypopigmentation is surrounded by peripheral pigmentation.",
    "Homogeneous red coloration is present.",
    "Tan to dark brown pigmentation with focal lighter areas is noted.",
]

STRUCTURE_POOL = [
    "A pigment network is appreciated across the lesion surface.",
    "Dotted vessels are distributed within the lesion.",
    "Globules are distributed within the lesion.",
    "Regression structures are present centrally.",
    "Vascular structures are identified at the periphery.",
    "White structureless areas are present within the lesion.",
    "Surface scale is noted overlying the lesion.",
    "Milia-like cysts are scattered throughout the lesion.",
    "Follicular openings are identified within the lesion.",
    "Fibrous structures are present centrally.",
    "Lacunae of varying sizes are distributed throughout.",
    "Crystalline structures are visible under polarized dermoscopy.",
    "Hairpin vessels with white halos are noted.",
    "Comedo-like openings are present.",
    "Shiny white structures are seen under polarized light.",
    "Peripheral streaks are noted at the lesion margin.",
    "Irregular globules are distributed within the lesion.",
    "A reticular pattern is appreciated across the lesion.",
    "Rosette structures are visible under polarized dermoscopy.",
    "Fissures and ridges are present across the surface.",
]

NEGATIVE_POOL = [
    "No ulceration is identified.",
    "No atypical vascular structures are identified.",
    "No regression structures are present.",
    "No blue-white veil is identified.",
    "No pseudopods or radial streaming at the periphery.",
    "No irregular border segments are noted.",
    "No multicomponent pattern is identified.",
    "No atypical globules at the periphery.",
    "No surface scale is identified.",
    "No nodular morphology is present.",
    "No focal hypopigmentation is noted.",
    "No irregular pigment network is appreciated.",
    "No vascular atypia is identified.",
    "No asymmetry in color or structure.",
    "No focal color variation is identified.",
]

NEUTRAL_CLOSINGS = [
    "Clinical correlation with dermoscopic findings is recommended.",
    "Further evaluation is advised based on clinical context.",
    "Dermatologic assessment is recommended for definitive management.",
    "Short-term follow-up imaging is suggested to assess interval change.",
    "Specialist review is recommended for definitive clinical management.",
    "Correlation with patient history and clinical examination is advised.",
    "Management decisions should be guided by clinical and histopathological findings.",
    "Periodic skin surveillance is recommended based on overall risk profile.",
    "Clinical judgment in conjunction with dermoscopic findings is advised.",
    "The treating clinician should correlate findings with the patient's history.",
]

LOC_PHRASES = {
    "face"           : ["on the face", "involving the facial skin",
                        "on the cheek or nasal region", "on the face with chronic UV exposure"],
    "scalp"          : ["on the scalp", "involving the scalp",
                        "on the scalp where self-examination is limited"],
    "back"           : ["on the back", "on the posterior trunk",
                        "involving the upper back", "on the back, a high-incidence site"],
    "chest"          : ["on the chest", "on the anterior chest wall", "involving the chest"],
    "acral"          : ["on an acral surface", "involving the acral skin", "on an acral site"],
    "ear"            : ["on the ear", "involving the auricular region",
                        "on the ear with chronic sun exposure"],
    "genital"        : ["on the genital region", "involving the anogenital skin"],
    "lower extremity": ["on the lower leg", "on the lower extremity",
                        "involving the calf or thigh"],
    "upper extremity": ["on the upper arm", "on the upper extremity", "involving the forearm"],
    "neck"           : ["on the neck", "involving the neck", "on the posterior neck"],
    "foot"           : ["on the foot", "involving the plantar surface", "on the sole of the foot"],
    "hand"           : ["on the hand", "involving the dorsal hand", "on the dorsal hand surface"],
    "abdomen"        : ["on the abdomen", "involving the abdominal skin"],
    "trunk"          : ["on the trunk", "involving the truncal skin", "on the truncal region"],
    "unknown"        : ["at an unspecified site", "at an undocumented anatomical site",
                        "at a site not documented in records"],
}

DX_TYPE_PHRASES = {
    "histo"    : ["via histopathological biopsy", "by excisional biopsy",
                  "via punch biopsy with histological review",
                  "by tissue biopsy and pathological analysis"],
    "follow_up": ["via serial dermoscopic follow-up", "through longitudinal digital dermoscopy",
                  "via follow-up imaging", "through interval dermoscopic monitoring"],
    "consensus": ["via expert dermoscopic consensus", "by multi-observer consensus",
                  "through consensus assessment by experienced dermatologists",
                  "by specialist panel consensus review"],
    "confocal" : ["via reflectance confocal microscopy", "using in-vivo confocal imaging",
                  "via confocal microscopy with dermoscopic correlation",
                  "by confocal microscopy providing cellular-level resolution"],
}

OPENING_HEADERS = [
    "Findings: {demo}",
    "Clinical history: {demo}",
    "Dermoscopy report: {demo}",
    "History and findings: {demo}",
    "{demo}",
    "Dermatology consultation: {demo}",
    "Skin examination: {demo}",
]

UNCERTAINTY_PHRASES = [
    "Interval change is noted, though definitive diagnosis requires histopathological correlation.",
    "Lesion evolution is documented on serial imaging. Clinical correlation is recommended.",
    "Serial imaging shows change. Biopsy may be warranted to exclude significant pathology.",
    "Documented interval change is noted. Further workup may be required.",
    "Follow-up imaging demonstrates lesion evolution. Excision may be considered.",
]


def demographic_sentence(age_val, sex_str, loc_str, dt_str, rng):
    if age_val == "unknown":
        templates = [
            f"A {sex_str} patient presents with a lesion {loc_str}.",
            f"A lesion {loc_str} is noted in a {sex_str} patient. Evaluation performed {dt_str}.",
            f"Dermoscopic evaluation {dt_str} was performed for a lesion {loc_str}.",
            f"A {sex_str} patient was referred for evaluation of a lesion {loc_str}.",
        ]
    else:
        templates = [
            f"The patient is a {age_val}-year-old {sex_str} presenting with a lesion {loc_str}.",
            f"A lesion {loc_str} is noted in a {age_val}-year-old {sex_str}.",
            f"Dermoscopic evaluation {dt_str} was performed on a {age_val}-year-old {sex_str} with a lesion {loc_str}.",
            f"A {age_val}-year-old {sex_str} was referred for evaluation of a lesion {loc_str}.",
            f"{age_val}-year-old {sex_str}, lesion {loc_str}. Evaluation {dt_str}.",
            f"Patient is a {age_val}-year-old {sex_str}. A lesion {loc_str} is under evaluation.",
        ]
    return rng.choice(templates)


def generate_report(row: pd.Series) -> str:
    dx_type  = str(row.get("dx_type", "consensus")).lower().strip()
    age      = row.get("age", None)
    sex      = str(row.get("sex", "unknown")).lower().strip()
    loc      = str(row.get("localization", "unknown")).lower().strip()
    image_id = str(row.get("image_id", ""))

    seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = random.Random(seed)

    age_val = int(age) if pd.notna(age) else "unknown"
    sex_str = "male" if sex == "male" else ("female" if sex == "female" else "patient")
    loc_str = rng.choice(LOC_PHRASES.get(loc, [f"on the {loc.replace('_', ' ')}"]))
    dt_str  = rng.choice(DX_TYPE_PHRASES.get(dx_type, ["via dermoscopy"]))

    demo       = demographic_sentence(age_val, sex_str, loc_str, dt_str, rng)
    shape      = rng.choice(SHAPE_POOL)
    border     = rng.choice(BORDER_POOL)
    color      = rng.choice(COLOR_POOL)
    structs    = rng.sample(STRUCTURE_POOL, rng.choice([1, 2]))
    struct_txt = " ".join(structs)
    negative   = rng.choice(NEGATIVE_POOL)
    closing    = rng.choice(NEUTRAL_CLOSINGS)

    size_txt = ""
    if rng.random() < 0.4:
        size_txt = rng.choice([
            f"The lesion measures approximately {rng.randint(3, 15)}mm in greatest diameter.",
            f"Lesion diameter is estimated at {rng.randint(3, 15)}mm.",
            f"A sub-centimeter lesion is identified.",
            f"The lesion is {rng.randint(3, 15)}mm in size.",
        ]) + " "

    uncertainty_txt = ""
    if dx_type == "follow_up" and rng.random() < 0.6:
        uncertainty_txt = rng.choice(UNCERTAINTY_PHRASES) + " "

    order = rng.choice(["standard", "color_first", "structure_first", "brief"])
    if order == "standard":
        body = f"{shape} {border} {color} {struct_txt} {size_txt}{uncertainty_txt}{negative}"
    elif order == "color_first":
        body = f"{color} {shape} {border} {struct_txt} {size_txt}{uncertainty_txt}{negative}"
    elif order == "structure_first":
        body = f"{struct_txt} {shape} {color} {size_txt}{uncertainty_txt}{negative}"
    else:
        body = f"{shape} {color} {size_txt}{uncertainty_txt}{negative}"

    opening = rng.choice(OPENING_HEADERS).format(demo=demo)
    return f"{opening} {body} {closing}"


def find_image_path(image_id: str) -> str:
    for folder in [IMAGE_DIR_1, IMAGE_DIR_2]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return ""


# ── Build CSV ──────────────────────────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
print(f"Loaded {len(meta)} rows\n")

meta["report_text"] = meta.apply(generate_report, axis=1)
meta["malignant"]   = meta["dx"].str.lower().map(MALIGNANCY)
meta["image_path"]  = meta["image_id"].apply(find_image_path)

missing = meta["image_path"] == ""
if missing.sum() > 0:
    print(f"⚠️  Dropping {missing.sum()} rows with missing images")
    meta = meta[~missing].reset_index(drop=True)

for dx_code in DX_FULL.keys():
    meta[dx_code] = (meta["dx"].str.lower() == dx_code).astype(int)

keep = ["image_id", "image_path", "report_text",
        "age", "sex", "localization", "dx", "dx_type", "malignant",
        "mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
df = meta[[c for c in keep if c in meta.columns]]
df.to_csv(OUTPUT_CSV, index=False)

# ── Validation ─────────────────────────────────────────────────────────────────
total   = len(df)
unique  = df["report_text"].nunique()
lengths = df["report_text"].str.len()

print(f"✓ Saved           : {total} rows → {OUTPUT_CSV}")
print(f"✓ Unique reports  : {unique}/{total} ({100*unique/total:.1f}%)")
print(f"✓ Report length   : min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")
print(f"✓ Malignant       : {df['malignant'].sum()} | Benign: {(df['malignant']==0).sum()}")

# ── Leakage check ──────────────────────────────────────────────────────────────
DIAGNOSIS_KEYWORDS = [
    "melanoma", "nevus", "nevi", "basal cell", "carcinoma",
    "keratosis", "dermatofibroma", "vascular lesion", "angiokeratoma",
    "hemangioma", "bowen", "intraepithelial", "malignant", "benign",
    "squamous", "fibroma", "angioma",
]

print(f"\n── Leakage check ──────────────────────────────────────────────")
leakage_found = 0
for keyword in DIAGNOSIS_KEYWORDS:
    matches = df["report_text"].str.lower().str.contains(keyword, regex=False)
    count = matches.sum()
    if count > 0:
        print(f"  ⚠️  '{keyword}' found in {count} reports")
        leakage_found += count
if leakage_found == 0:
    print("  ✓ Zero leakage — no diagnosis keywords in any report")

print(f"\n── Sample reports ─────────────────────────────────────────────")
for dx_code in ["mel", "nv", "bcc", "akiec"]:
    sample = df[df["dx"] == dx_code].iloc[0]
    print(f"\n{dx_code.upper()}:")
    print(sample["report_text"])
