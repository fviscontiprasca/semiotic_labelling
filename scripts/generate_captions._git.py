# scripts/generate_captions_git.py
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import fiftyone as fo
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("microsoft/git-large")
model = AutoModelForVision2Seq.from_pretrained("microsoft/git-large").to(device)

ds = fo.load_dataset("semiocity_urban")

for sample in ds:
    img = Image.open(sample.filepath).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    sample["caption_auto"] = caption
    sample.save()

print("✅ Captions generated and stored in FiftyOne dataset")
