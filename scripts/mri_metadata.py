import os
import hashlib
import numpy as np
import pandas as pd

BASE      = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
TRAIN_DIR = f"{BASE}/Training"
TEST_DIR  = f"{BASE}/Testing"
OUTPUT    = "/kaggle/working/mri_metadata.csv"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Epidemiologically grounded distributions
# Sources: CBTRUS 2024, WHO CNS Tumor Classification 2021
AGE_DIST = {
    "glioma"     : (55, 12),   # peaks 45-65, male-dominant
    "meningioma" : (58, 13),   # peaks 50-70, female-dominant
    "pituitary"  : (40, 14),   # peaks 30-50, roughly equal
    "notumor"    : (40, 15),   # broad working-age range
}
SEX_DIST = {
    "glioma"     : 0.60,   # P(male)
    "meningioma" : 0.30,
    "pituitary"  : 0.50,
    "notumor"    : 0.50,
}

def synthetic_metadata(image_id: str, dx: str) -> dict:
    seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.default_rng(seed)

    mu, sigma = AGE_DIST[dx]
    age = int(np.clip(rng.normal(mu, sigma), 18, 85))
    sex = "male" if rng.random() < SEX_DIST[dx] else "female"
    return {"age": age, "sex": sex}


rows = []
for split_dir, split in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
    for dx in CLASSES:
        folder = os.path.join(split_dir, dx)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id   = os.path.splitext(fname)[0]
            image_path = os.path.join(folder, fname)
            meta       = synthetic_metadata(image_id, dx)
            rows.append({
                "image_id"  : image_id,
                "image_path": image_path,
                "dx"        : dx,
                "split"     : split,
                "age"       : meta["age"],
                "sex"       : meta["sex"],
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print(f"✓ Total images : {len(df)}")
print(f"✓ Saved        : {OUTPUT}\n")
print(df.groupby(["dx", "split"]).size().unstack(fill_value=0))
print(f"\nAge stats per class:")
print(df.groupby("dx")["age"].agg(["mean", "std", "min", "max"]).round(1))
print(f"\nSex distribution per class:")
print(df.groupby(["dx", "sex"]).size().unstack(fill_value=0))
