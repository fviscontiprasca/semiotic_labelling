from ultralytics import YOLO
from pathlib import Path
import cv2
import os

# Setup
model = YOLO("yolov8n-seg.pt")
image_dir = Path("dataset/images/train")
label_dir = Path("dataset/labels/train")
label_dir.mkdir(parents=True, exist_ok=True)

# Define your label mapping manually
label_map = {
    0: "Car",
    1: "Truck",
    2: "Dragon"
}

# Inference and save YOLO-format masks
for img_path in image_dir.glob("*.jpg"):
    result = model(img_path)[0]
    lines = []
    for i in range(len(result.masks)):
        cls = int(result.boxes.cls[i])
        if cls not in label_map:
            continue
        box = result.boxes.xywhn[i]
        seg = result.masks.xy[i].flatten().tolist()
        seg_norm = [
            round(seg[j] / (result.orig_shape[1 if j % 2 == 0 else 0]), 6)
            for j in range(len(seg))
        ]
        line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} " + " ".join(map(str, seg_norm))
        lines.append(line)
    with open(label_dir / f"{img_path.stem}.txt", "w") as f:
        f.write("\n".join(lines))
