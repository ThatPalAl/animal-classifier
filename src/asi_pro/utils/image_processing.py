from PIL import Image
import numpy as np

def load_image(path: str, target_size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    return np.array(img)

def preprocess_for_model(img: np.ndarray):
    return img / 255.0
