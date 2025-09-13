# scripts/export_for_training.py
import fiftyone as fo
from pathlib import Path
import csv

ds = fo.load_dataset("semiocity_urban")

OUT_IMG = Path("data/fiftyone_export/images")
OUT_MASK = Path("data/fiftyone_export/masks")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

rows = []

for sample in ds:
    img_path = Path(sample.filepath)
    img_dst = OUT_IMG / img_path.name
    if not img_dst.exists():
        img_dst.write_bytes(img_path.read_bytes())

    mask_field = None
    for f in ("ground_truth", "detector_preds", "sam_predictions"):
        if f in sample and sample[f] is not None:
            mask_field = f
            break

    mask_path = ""
    if mask_field:
        seg = sample[mask_field]
        if hasattr(seg, "mask") and seg.mask:
            mask_path = OUT_MASK / (img_path.stem + ".png")
            Path(mask_path).write_bytes(Path(seg.mask).read_bytes())

    caption = sample.get("caption_auto") or ""
    rows.append([str(img_dst.resolve()), str(mask_path), caption])

with open("data/fiftyone_export/captions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "mask", "caption"])
    writer.writerows(rows)

print("✅ Exported dataset for training")
