# =========================
# TRAIN.PY
# =========================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import segmentation_models_pytorch as smp
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data"

# -------------------------
# LOAD SPLITS
# -------------------------
def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

train_ids = load_split(f"{DATA_DIR}/split/train.txt")
val_ids   = load_split(f"{DATA_DIR}/split/val.txt")
all_ids   = train_ids + val_ids


# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    hh, hv = img[0], img[1]
    green, red, nir, swir = img[2], img[3], img[4], img[5]

    eps = 1e-6

    ndwi  = (green - nir) / (green + nir + eps)
    mndwi = (green - swir) / (green + swir + eps)
    ndvi  = (nir - red) / (nir + red + eps)
    sar_diff = hh - hv

    bands = [hh, hv, green, red, nir, swir, ndwi, mndwi, ndvi, sar_diff]

    normed = []
    for b in bands:
        p2, p98 = np.percentile(b, [2,98])
        b = np.clip(b, p2, p98)
        b = (b - p2)/(p98 - p2 + eps)
        normed.append(b)

    return np.stack(normed).astype(np.float32)


# -------------------------
# DATASET
# -------------------------
class FloodDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        with rasterio.open(f"{DATA_DIR}/image/{pid}_image.tif") as src:
            img = preprocess(src.read().astype(np.float32))

        with rasterio.open(f"{DATA_DIR}/label/{pid}_label.tif") as src:
            mask = src.read(1).astype(np.int64)

        return torch.from_numpy(img), torch.from_numpy(mask)


# -------------------------
# MODEL
# -------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=10,
    classes=3,
    decoder_attention_type="scse"
).to(device)


# -------------------------
# LOSS
# -------------------------
weights = torch.tensor([0.3, 5.0, 1.0]).to(device)

ce = nn.CrossEntropyLoss(weight=weights)
dice = smp.losses.DiceLoss(mode="multiclass")

def loss_fn(logits, targets):
    return 0.5 * ce(logits, targets) + 0.5 * dice(logits, targets)


# -------------------------
# TRAIN
# -------------------------
train_loader = DataLoader(FloodDataset(all_ids), batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(imgs), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model2.pth")
print("Model saved!")
