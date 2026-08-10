import pathlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split

EMBEDDINGS_PATH = "embeddings/isic2018_task1.npz"
IMAGES_DIR = pathlib.Path("data/isic2018_task1/ISIC2018_Task1-2_Training_Input")
MASKS_DIR = pathlib.Path("data/isic2018_task1/ISIC2018_Task1_Training_GroundTruth")
WEIGHTS_PATH = "weights/head.pt"
GRID_SIZE = 16
VAL_FRACTION = 0.2
SEED = 0  # must match train.py so val_idx points at the same held-out images
NUM_HEATMAPS = 25
OVERLAY_ALPHA = 0.75
OUTLINE_THICKNESS = 5  # odd number, higher = thicker green outline

OUT_DIR = pathlib.Path("predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(EMBEDDINGS_PATH)
embeddings = data["embeddings"]
image_ids = data["image_ids"]
IMAGE_COUNT = len(image_ids)

train_idx, val_idx = train_test_split(np.arange(IMAGE_COUNT), test_size=VAL_FRACTION, random_state=SEED)

head = nn.Linear(embeddings.shape[-1], 1)
head.load_state_dict(torch.load(WEIGHTS_PATH))
head.eval()

for idx in val_idx[:NUM_HEATMAPS]:
    image_id = image_ids[idx]
    embedding = torch.from_numpy(embeddings[idx])
    with torch.no_grad():
        out = head(embedding).squeeze(-1)
        probs = torch.sigmoid(out)

    grid = probs.numpy().reshape(GRID_SIZE, GRID_SIZE)

    image = Image.open(IMAGES_DIR / f"{image_id}.jpg").convert("RGB")
    heatmap = Image.fromarray((grid * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    heatmap_colored = ImageOps.colorize(heatmap, black="black", mid="red", white="yellow")

    heatmap_strength = heatmap.point(lambda p: int(p * OVERLAY_ALPHA))  # scale by pixel's own probability, not a flat blend
    overlay = Image.composite(heatmap_colored, image, heatmap_strength)

    mask = Image.open(MASKS_DIR / f"{image_id}_segmentation.png").convert("L").resize(image.size)
    mask_edges = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(OUTLINE_THICKNESS))
    mask_outline = ImageOps.colorize(mask_edges, black="black", white="lime")
    overlay = Image.composite(mask_outline, overlay, mask_edges)  # draw the outline, leave the rest of overlay untouched

    overlay.save(OUT_DIR / f"{image_id}_heatmap.png")
