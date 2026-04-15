import hashlib
import random
import pandas as pd

INPUT_CSV  = "/kaggle/working/mri_metadata.csv"
OUTPUT_CSV = "/kaggle/working/mri_with_reports.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED POOLS — every sentence is plausible across multiple classes.
# notumor uses the same vocabulary but draws from NEGATIVE_FINDING_POOL
# more often, mirroring real radiology practice.
# ═══════════════════════════════════════════════════════════════════════════════

SIGNAL_POOL = [
    "A focal area of heterogeneous signal intensity is identified.",
    "A region of mixed signal is noted on multisequence imaging.",
    "Focal signal alteration is present on T2-weighted sequences.",
    "An area of intermediate signal intensity is identified.",
    "Signal change is noted within the parenchyma on FLAIR sequences.",
    "No focal signal abnormality is identified on any sequence.",
    "Brain parenchyma demonstrates normal signal intensity throughout.",
    "No abnormal signal is identified on T1 or T2-weighted sequences.",
    "Normal gray-white matter differentiation is maintained.",
    "Diffuse signal change is noted in the periventricular region.",
]

LOCATION_POOL = [
    "The finding is located in the frontal lobe region.",
    "The abnormality is centered in the temporal lobe.",
    "The finding is situated in the posterior fossa.",
    "The lesion appears to arise from the convexity region.",
    "The abnormality is identified in the parietal lobe.",
    "The finding is noted within the midline supratentorial compartment.",
    "The lesion is located adjacent to the ventricular system.",
    "The abnormality occupies a superficial cortical location.",
    "The finding is identified in the deep white matter region.",
    "The lesion is identified in the sellar and parasellar region.",
    "No focal intracranial lesion is identified.",
    "The brain parenchyma appears unremarkable in signal and morphology.",
    "No space-occupying lesion is identified within the intracranial compartment.",
]

MORPHOLOGY_POOL = [
    "The lesion demonstrates well-defined margins with a rounded morphology.",
    "The mass exhibits irregular borders and heterogeneous internal architecture.",
    "The lesion appears ovoid with relatively smooth margins.",
    "The abnormality shows lobulated contours.",
    "The lesion is characterized by a solid component.",
    "The mass shows a predominantly solid appearance.",
    "The finding appears as a discrete lesion with a well-delineated capsule.",
    "No discrete mass lesion or focal morphological abnormality is identified.",
    "Cortical and subcortical structures appear morphologically intact.",
    "No abnormal lesion morphology is identified on any sequence.",
]

ENHANCEMENT_POOL = [
    "Following contrast administration, the lesion demonstrates heterogeneous enhancement.",
    "Post-contrast imaging reveals peripheral rim enhancement.",
    "The lesion demonstrates minimal contrast enhancement.",
    "Post-contrast sequences show moderate patchy enhancement.",
    "Following gadolinium administration, nodular enhancement is identified.",
    "No abnormal enhancement is identified following contrast administration.",
    "Post-contrast imaging demonstrates no pathological enhancement.",
    "No abnormal intracranial enhancement is identified.",
    "Post-contrast sequences are unremarkable without focal enhancement.",
    "Avid homogeneous enhancement is identified on post-contrast sequences.",
]

MASS_EFFECT_POOL = [
    "Mild local mass effect with sulcal effacement is noted.",
    "No significant mass effect or midline shift is demonstrated.",
    "Moderate perilesional edema is present contributing to local mass effect.",
    "Mild compression of adjacent structures is noted.",
    "No evidence of herniation or ventricular obstruction is identified.",
    "Mild effacement of the adjacent sulci is noted.",
    "No midline shift is identified; ventricles appear within normal limits.",
    "Ventricular system appears normal in size and configuration.",
    "No hydrocephalus or ventricular dilatation is identified.",
    "Sulci and gyri appear normal for patient age.",
]

NEUTRAL_CLOSINGS = [
    "Clinical correlation with neuroimaging findings is recommended.",
    "Further evaluation with advanced MRI sequences may be considered.",
    "Neurosurgical consultation is recommended for definitive management.",
    "Short-interval follow-up MRI is suggested to assess interval change.",
    "Correlation with clinical presentation and laboratory findings is advised.",
    "Multidisciplinary team review is recommended for treatment planning.",
    "Histopathological confirmation is recommended for definitive diagnosis.",
    "Management decisions should be guided by clinical and imaging findings.",
    "Specialist neuroradiology review is recommended.",
    "The treating clinician should correlate findings with the patient's history.",
]

OPENING_HEADERS = [
    "MRI Brain Report: {demo}",
    "Neuroradiology findings: {demo}",
    "Brain MRI: {demo}",
    "Clinical history: {demo}",
    "{demo}",
    "Neuroimaging report: {demo}",
    "MRI examination: {demo}",
]

SEQUENCE_PHRASES = [
    "on multisequence MRI",
    "on T1 and T2-weighted sequences",
    "on contrast-enhanced MRI",
    "on FLAIR and T1-weighted sequences",
    "on MRI with and without contrast",
    "on routine brain MRI protocol",
]

# Indices of "negative/normal" sentences in each pool
# notumor draws from these with higher probability
SIGNAL_NORMAL_IDX    = [5, 6, 7, 8]
LOCATION_NORMAL_IDX  = [10, 11, 12]
MORPHOLOGY_NORMAL_IDX= [7, 8, 9]
ENHANCEMENT_NORMAL_IDX=[5, 6, 7, 8]
MASS_NORMAL_IDX      = [1, 4, 6, 7, 8, 9]


def _weighted_choice(pool, normal_idx, rng, p_normal):
    """Pick from pool; with probability p_normal pick from normal_idx subset."""
    if rng.random() < p_normal:
        return pool[rng.choice(normal_idx)]
    abnormal_idx = [i for i in range(len(pool)) if i not in normal_idx]
    return pool[rng.choice(abnormal_idx)]


def generate_report(row: pd.Series) -> str:
    image_id = str(row["filename"])
    age      = int(row["age"])
    sex      = str(row["sex"]).lower()
    dx       = str(row["label"]).lower()

    seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)

    # All classes draw from both normal and abnormal sentences.
    # notumor slightly favors normal, tumor classes slightly favor abnormal.
    # Overlap is intentional — prevents BERT from trivially classifying.
    p_normal = 0.55 if dx == "notumor" else 0.35

    sex_str = "male" if sex in ("m", "male") else "female"
    seq     = rng.choice(SEQUENCE_PHRASES)
    demo    = f"a {age}-year-old {sex_str} patient"

    signal   = _weighted_choice(SIGNAL_POOL,      SIGNAL_NORMAL_IDX,     rng, p_normal)
    location = _weighted_choice(LOCATION_POOL,    LOCATION_NORMAL_IDX,   rng, p_normal)
    morph    = _weighted_choice(MORPHOLOGY_POOL,  MORPHOLOGY_NORMAL_IDX, rng, p_normal)
    enhance  = _weighted_choice(ENHANCEMENT_POOL, ENHANCEMENT_NORMAL_IDX,rng, p_normal)
    mass     = _weighted_choice(MASS_EFFECT_POOL, MASS_NORMAL_IDX,       rng, p_normal)
    closing  = rng.choice(NEUTRAL_CLOSINGS)

    order = rng.choice(["standard", "location_first", "signal_first", "brief"])
    if order == "standard":
        body = f"{signal} {location} {morph} {enhance} {mass}"
    elif order == "location_first":
        body = f"{location} {signal} {morph} {enhance} {mass}"
    elif order == "signal_first":
        body = f"{signal} {location} {enhance} {mass}"
    else:
        body = f"{location} {signal} {mass}"

    opening = rng.choice(OPENING_HEADERS).format(
        demo=f"Brain MRI {seq} performed on {demo}."
    )
    return f"{opening} {body} {closing}"


# ── Build CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")

df["report_text"] = df.apply(generate_report, axis=1)
df.to_csv(OUTPUT_CSV, index=False)

total  = len(df)
unique = df["report_text"].nunique()
lengths = df["report_text"].str.len()
print(f"✓ Saved          : {total} rows → {OUTPUT_CSV}")
print(f"✓ Unique reports : {unique}/{total} ({100*unique/total:.1f}%)")
print(f"✓ Report length  : min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")

# ── Leakage check ──────────────────────────────────────────────────────────────
DIAGNOSIS_KEYWORDS = [
    "glioma", "glioblastoma", "meningioma", "pituitary", "adenoma",
    "astrocytoma", "oligodendroglioma", "tumor", "tumour", "malignant",
    "benign", "carcinoma", "metastasis", "lymphoma",
]
print(f"\n── Leakage check ──────────────────────────────────────────────")
leakage_found = 0
for kw in DIAGNOSIS_KEYWORDS:
    n = df["report_text"].str.lower().str.contains(kw, regex=False).sum()
    if n > 0:
        print(f"  ⚠️  '{kw}' found in {n} reports")
        leakage_found += n
if leakage_found == 0:
    print("  ✓ Zero leakage")

print(f"\n── Sample reports ─────────────────────────────────────────────")
for dx in ["glioma", "meningioma", "notumor", "pituitary"]:
    s = df[df["label"] == dx].iloc[0]
    print(f"\n{dx.upper()} (age={s['age']}, sex={s['sex']}):")
    print(s["report_text"])
