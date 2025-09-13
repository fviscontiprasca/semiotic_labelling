# scripts/auto_segment_sam.py
from segment_anything import sam_model_registry, SamPredictor
import cv2, numpy as np
from pathlib import Path
import fiftyone as fo
from PIL import Image

MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # download from SAM repo

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

ds = fo.load_dataset("semiocity_urban")
out_field = "sam_predictions"

if out_field in ds.get_field_schema():
    ds.delete_sample_field(out_field)
ds.add_sample_field(out_field, fo.Segmentation)

for sample in ds.iter_samples():
    img = cv2.imread(sample.filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=None, box=None, multimask_output=True
    )

    if len(masks) > 0:
        mask = masks[0].astype(np.uint8) * 255
        out_path = Path("data/fiftyone_export/masks") / f"{Path(sample.filepath).stem}_sam.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(out_path)

        sample[out_field] = fo.Segmentation(mask=str(out_path))
        sample.save()

print("✅ SAM predictions added to FiftyOne dataset")
