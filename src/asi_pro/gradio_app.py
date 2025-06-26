import gradio as gr
from fastai.vision.all import *
from pathlib import Path
import sys, os

# Dodaj src do PATH, jeśli będzie potrzeba
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

# Dummy DataLoaders (by móc odbudować Learner)
dls = ImageDataLoaders.from_folder(
    "data/01_raw",
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    bs=32
)

learn = vision_learner(dls, resnet18, metrics=accuracy)
model_path = Path(__file__).parent.parent.parent / "data/06_models/model"
learn.load(model_path)



def predict(img):
    pred, idx, probs = learn.predict(PILImage.create(img))
    return f"Prediction: {pred}, Probability: {probs[idx]:.2f}"

iface = gr.Interface(fn=predict, inputs="image", outputs="text")
iface.launch(server_name="0.0.0.0", server_port=80)