import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["segmentations"],
    classes=["Building", "Cart", "Castle", "Convenience store", "Fountain", "House", "Lighthouse", "Office building", 
             "Skyscraper", "Tower"],
    max_samples=7,
    shuffle=True,
    dataset_name="train_dataset"
)

session = fo.launch_app(dataset.view(), port=5151)