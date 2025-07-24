import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil

def run_inference(model_path, source_dir, output_dir):
    model = YOLO(model_path)
    source_dir = Path(source_dir)
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    results = model.predict(source=source_dir, save=False, save_txt=True, save_conf=False, stream=True, imgsz=640)

    for r in results:
        img_path = r.path
        label_path = Path(r.save_dir) / "labels" / (Path(img_path).stem + ".txt")
        
        # Copy image
        shutil.copy(img_path, out_images / Path(img_path).name)
        # Copy label
        if label_path.exists():
            shutil.copy(label_path, out_labels / label_path.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using YOLOv8")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model checkpoint (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Directory of images to annotate")
    parser.add_argument("--output", type=str, required=True, help="Where to save labeled data")
    args = parser.parse_args()
    
    run_inference(args.model, args.source, args.output)
