from fastai.vision.all import *
from pathlib import Path

def load_data(path: str = "data/01_raw/") -> DataLoaders:
    return ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        bs=32
    )

def train_model(dls: DataLoaders, model_output_path: str) -> None:
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    learn = vision_learner(dls, resnet18, metrics=accuracy, path='.')
    learn.fine_tune(3)
    torch.save(learn.model.state_dict(), model_path.with_suffix('.pth'))