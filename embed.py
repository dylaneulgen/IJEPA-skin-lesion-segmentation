import os
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "facebook/ijepa_vith14_1k"
IMAGES_DIR = "data/isic2018_task1/ISIC2018_Task1-2_Training_Input"
OUT_PATH = "embeddings/isic2018_task1.npz"
BATCH_SIZE = 8

paths = sorted(pathlib.Path(IMAGES_DIR).glob("*.jpg"))
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

embeddings = []
image_ids = []
for i in range(0, len(paths), BATCH_SIZE):
    chunk = paths[i : i + BATCH_SIZE]
    images = [Image.open(path).convert("RGB") for path in chunk]
    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)

    embeddings.append(out.last_hidden_state.numpy().astype(np.float32))
    image_ids.extend(path.stem for path in chunk)

embeddings = np.concatenate(embeddings, axis=0)

pathlib.Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_PATH, embeddings=embeddings, image_ids=np.array(image_ids))
