from fastai.vision.all import *
import pandas as pd

def train_model(csv_path: str) -> Learner:
    df = pd.read_csv(csv_path)

    dls = ImageDataLoaders.from_df(
        df,
        fn_col="image",
        label_col="label",
        folder=".",
        item_tfms=Resize(224),
        bs=32
    )

    learn = cnn_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(3)

    return learn
