import json
import os
from pathlib import Path
from PIL import Image

# === Config ===
label_studio_export = "label_studio_export.json"
output_dir = Path("dataset")
output_img_dir = output_dir / "images" / "train"
output_lbl_dir = output_dir / "labels" / "train"
hierarchy_map_file = "label_hierarchy.json"

output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

# === Load hierarchy for name mapping ===
with open(hierarchy_map_file) as f:
    hierarchy = json.load(f)

# Create flat label → class_id map
flat_labels = []
for parent, children in hierarchy.items():
    flat_labels.append(parent)
    flat_labels.extend(children)
class_map = {name: idx for idx, name in enumerate(flat_labels)}

# === Convert ===
with open(label_studio_export) as f:
    export_data = json.load(f)

for item in export_data:
    img_path = item["data"]["image"]
    img_filename = Path(img_path).name
    img_full_path = Path(img_path)

    # Copy image to training folder
    dest_img_path = output_img_dir / img_filename
    if not dest_img_path.exists():
        Image.open(img_full_path).save(dest_img_path)

    # Write YOLO label file
    shapes = item["annotations"][0]["result"]
    label_lines = []
    for shape in shapes:
        label = shape["value"]["labels"][0]
        normalized_label = label.replace(" / ", "_").replace(" ", "_")
        class_id = class_map.get(normalized_label)
        if class_id is None:
            print(f"⚠️ Skipping unknown label: {label}")
            continue

        points = shape["value"]["points"]
        width = shape["original_width"]
        height = shape["original_height"]
        normalized_points = []
        for pt in points:
            x = pt[0] / width
            y = pt[1] / height
            normalized_points.extend([x, y])

        # Dummy box center/width/height for YOLOv8 segmentation
        x_coords = normalized_points[::2]
        y_coords = normalized_points[1::2]
        x_center = sum(x_coords) / len(x_coords)
        y_center = sum(y_coords) / len(y_coords)
        box_w = (max(x_coords) - min(x_coords))
        box_h = (max(y_coords) - min(y_coords))

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} " + " ".join(f"{p:.6f}" for p in normalized_points)
        label_lines.append(line)

    label_path = output_lbl_dir / f"{img_filename.rsplit('.', 1)[0]}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
