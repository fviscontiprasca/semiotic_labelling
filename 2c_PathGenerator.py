import json
from pathlib import Path

image_folder = Path("/home/you/projects/my-labeling/images")
images = sorted(image_folder.glob("*.jpg"))

data = [{"image": str(img.resolve())} for img in images]

with open("import_images.json", "w") as f:
    json.dump(data, f, indent=2)