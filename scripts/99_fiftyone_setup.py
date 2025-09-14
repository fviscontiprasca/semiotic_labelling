# scripts/fiftyone_setup.py
import fiftyone as fo
from fiftyone import Dataset
from fiftyone.types import ImageDirectory

dataset_name = "semiocity_urban"

if Dataset.exists(dataset_name):
    ds = Dataset.load_dataset(dataset_name)
else:
    ds = Dataset(dataset_name)

ds.add_dir("data/merged", dataset_type=ImageDirectory)
print(f"✅ Dataset '{dataset_name}' created with {len(ds)} samples")

session = fo.launch_app(ds)
session.wait()
