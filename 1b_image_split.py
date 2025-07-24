import random, shutil
from pathlib import Path

raw_dir = Path("all_images/")
manual_dir = Path("Datasets/gtFinePanopticParts_trainval/manual_labeling")
auto_dir = Path("Datasets/gtFinePanopticParts_trainval/auto_labeling")
manual_dir.mkdir(parents=True, exist_ok=True)
auto_dir.mkdir(parents=True, exist_ok=True)

images = list(raw_dir.glob("*.jpg"))
random.shuffle(images)
n_manual = int(0.1 * len(images))  # 10% for manual labeling

for img in images[:n_manual]:
    shutil.copy(img, manual_dir / img.name)
for img in images[n_manual:]:
    shutil.copy(img, auto_dir / img.name)
