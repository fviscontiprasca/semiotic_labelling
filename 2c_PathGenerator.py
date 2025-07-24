import json
from pathlib import Path

image_folder = Path("Datasets/gtFinePanopticParts_trainval/manual_labeling")
images = sorted(image_folder.glob("*.tif"))

data = [{"image": str(img.resolve())} for img in images]

with open("import_images.json", "w") as f:
    json.dump(data, f, indent=2)