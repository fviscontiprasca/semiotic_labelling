# scripts/labels_to_masks.py
from PIL import Image, ImageDraw
from pathlib import Path

IMG_DIR = Path("data/fiftyone_export/images")
LBL_DIR = Path("data/fiftyone_export/labels")
OUT_MASK = Path("data/fiftyone_export/masks")
OUT_MASK.mkdir(parents=True, exist_ok=True)

for img_path in IMG_DIR.glob("*.jpg"):
    w, h = Image.open(img_path).size
    mask = Image.new("L", (w, h), 0)
    lbl_file = LBL_DIR / (img_path.stem + ".txt")

    if not lbl_file.exists():
        mask.save(OUT_MASK / (img_path.stem + ".png"))
        continue

    draw = ImageDraw.Draw(mask)
    for line in open(lbl_file):
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        seg = [float(x) for x in parts[5:]]
        pts = [(seg[i] * w, seg[i + 1] * h) for i in range(0, len(seg), 2)]
        draw.polygon(pts, outline=255, fill=255)

    mask.save(OUT_MASK / (img_path.stem + ".png"))

print("✅ Converted YOLO labels to masks")
