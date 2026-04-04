# =========================
# INFERENCE.PY (ENSEMBLE)
# =========================

import os
import numpy as np
import pandas as pd
import torch
import rasterio
from tqdm import tqdm
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "/kaggle/input/competitions/anrfaisehack-theme-1-phase2/data"


# -------------------------
# PREPROCESS (SAME)
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
# LOAD MODELS
# -------------------------
model1 = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b5",
    encoder_weights=None,
    in_channels=10,
    classes=3,
    decoder_attention_type="scse"
).to(device)

model2 = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=10,
    classes=3,
    decoder_attention_type="scse"
).to(device)

model1.load_state_dict(torch.load("best.pth", map_location=device))
model2.load_state_dict(torch.load("model2.pth", map_location=device))

model1.eval()
model2.eval()


# -------------------------
# RLE
# -------------------------
def rle(mask):
    pixels=(mask==1).astype(np.uint8).flatten(order="F")
    pixels=np.concatenate([[0],pixels,[0]])
    runs=np.where(pixels[1:]!=pixels[:-1])[0]+1
    runs[1::2]-=runs[::2]
    return " ".join(map(str,runs)) if len(runs)>0 else "0 0"


# -------------------------
# TEST IDS
# -------------------------
test_ids = [f.replace("_image.tif","")
            for f in os.listdir(f"{DATA_DIR}/prediction/image")]


# -------------------------
# INFERENCE
# -------------------------
rows = []

for pid in tqdm(test_ids):

    with rasterio.open(f"{DATA_DIR}/prediction/image/{pid}_image.tif") as src:
        img = preprocess(src.read().astype(np.float32))

    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        p1 = torch.softmax(model1(img_tensor), dim=1)[0].cpu().numpy()
        p2 = torch.softmax(model2(img_tensor), dim=1)[0].cpu().numpy()

    # 🔥 BEST WEIGHT (YOUR RESULT)
    final_probs = 0.8 * p1 + 0.2 * p2

    pred = final_probs.argmax(0)

    rows.append({"id": pid, "rle_mask": rle(pred)})

df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)

print("Submission saved!")
