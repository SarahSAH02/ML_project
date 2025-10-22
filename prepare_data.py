import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === Konfigurasjon ===
base_dir = Path("/Users/sarahsamiaahsan/DAT109-PROSJEKT/ML_project/data/archive (3)/my_real_vs_ai_dataset/my_real_vs_ai_dataset")

real_dir = base_dir / "real"
ai_dir = base_dir / "ai_images"

output_base = Path("/Users/sarahsamiaahsan/DAT109-PROSJEKT/ML_project/data/processed")

split_ratios = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# === Funksjon for å splitte og kopiere filer ===
def split_and_copy(class_dir, label_name):
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios["train"])
    val_end = int(total * (split_ratios["train"] + split_ratios["val"]))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_files in splits.items():
        target_dir = output_base / split_name / label_name
        os.makedirs(target_dir, exist_ok=True)

        for img_name in tqdm(split_files, desc=f"{label_name} → {split_name}", ncols=80):
            src = class_dir / img_name
            dst = target_dir / img_name
            if os.path.exists(src):
                shutil.copy(src, dst)

# === Kjør ===
print("🚀 Starter datasett-splitting ...")
split_and_copy(real_dir, "real")
split_and_copy(ai_dir, "ai")
print("✅ Ferdig! Data er delt og kopiert til:", output_base)
