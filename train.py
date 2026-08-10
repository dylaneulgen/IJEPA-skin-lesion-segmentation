import pathlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


EMBEDDINGS_PATH = "embeddings/isic2018_task1.npz"
MASKS_PATH = pathlib.Path("data/isic2018_task1/ISIC2018_Task1_Training_GroundTruth")
WEIGHTS_PATH = "weights/head.pt"
GRID_SIZE = 16
VAL_FRACTION = 0.2
SEED = 0

EPOCHS = 10
BATCH_SIZE = 4096
LR = 1e-3

data = np.load(EMBEDDINGS_PATH)
embeddings = data["embeddings"]
image_ids = data["image_ids"]
IMAGE_COUNT = len(image_ids)

masks = np.zeros((IMAGE_COUNT, GRID_SIZE * GRID_SIZE), dtype=np.float32)
for i, image_id in enumerate(image_ids):
    mask = Image.open(MASKS_PATH / f"{image_id}_segmentation.png").convert("L")
    mask = mask.resize((224, 224), resample=Image.BILINEAR)
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    block_size = 224 // GRID_SIZE

    # fraction of mask per patch
    mask = mask.reshape(GRID_SIZE, block_size, GRID_SIZE, block_size).mean(axis=(1, 3))
    masks[i] = mask.flatten()

# split into train/val
train_idx, val_idx = train_test_split(
    np.arange(IMAGE_COUNT), test_size=VAL_FRACTION, random_state=SEED
)

x_train = torch.from_numpy(embeddings[train_idx].reshape(-1, embeddings.shape[-1])) # (IMAGE_COUNT * 256, 1280)
y_train = torch.from_numpy(masks[train_idx].reshape(-1)) # (IMAGE_COUNT * 256,)
x_val = torch.from_numpy(embeddings[val_idx].reshape(-1, embeddings.shape[-1])) # (IMAGE_COUNT * 256, 1280)
y_val = torch.from_numpy(masks[val_idx].reshape(-1)) # (IMAGE_COUNT * 256,)

head = nn.Linear(embeddings.shape[-1], 1)
optimizer = torch.optim.Adam(head.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss() # avoids log(0) instability from doing them separately

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = head(x_batch).squeeze(-1) # (batch, 1) -> (batch,) to match y_batch
        loss = loss_fn(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x_batch) # accumulate total loss for the epoch relative to size

    with torch.no_grad():
        val_out = head(x_val).squeeze(-1) # (batch, 1) -> (batch,) to match y_batch
        val_loss = loss_fn(val_out, y_val).item()

    print(f"epoch {epoch + 1}/{EPOCHS}  train_loss={total_loss / len(x_train):.4f}  val_loss={val_loss:.4f}")

pathlib.Path("weights").mkdir(exist_ok=True)
torch.save(head.state_dict(), WEIGHTS_PATH)
