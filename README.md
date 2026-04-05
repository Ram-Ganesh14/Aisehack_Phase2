# AI-Powered Flood Segmentation using Multi-Modal Satellite Data

A deep learning pipeline for accurate flood segmentation using multi-modal satellite imagery, combining optical and SAR data with advanced architectures and ensemble strategies.

---

## 📂 Pipeline Overview

The project follows a structured end-to-end pipeline:

### 1. Data Loading
- Raster **TIFF** satellite imagery  
- Multi-band inputs (optical + SAR)

### 2. Feature Engineering
- Computation of spectral indices:
  - **NDWI** (Normalized Difference Water Index)
  - **NDVI** (Normalized Difference Vegetation Index)
  - Additional custom indices  
- Channel stacking for enriched input representation

### 3. Model Training
- **UNet (ResNet34 backbone)**
- **UNet++ (EfficientNet-B5 backbone)**
- Stratified training with imbalance handling

### 4. Ensemble & Inference
- Model ensembling for improved generalization  
- **Test Time Augmentation (TTA)** for robustness  

### 5. Post-processing
- Mask refinement  
- **RLE Encoding** for submission  

---

## 🚀 Key Innovations

- 🔗 **Multi-modal fusion** (Optical + SAR data)  
- 🌈 **Custom spectral indices** for better water detection  
- ⚖️ **Weighted loss functions** to handle class imbalance  
- 🧠 **Model ensembling** for higher accuracy  
- 🔁 **Test Time Augmentation (TTA)** for stable predictions  

---

## 📊 Models

| Model File | Architecture | Backbone |
|-----------|-------------|----------|
| `model2.pth` | UNet | ResNet34 |
| `best.pth` | UNet++ | EfficientNet-B5 |

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- segmentation-models-pytorch  
- Rasterio  
- NumPy  
- Pandas  
- OpenCV  

---

## 📈 Workflow Summary
Satellite Data → Feature Engineering → Model Training
→ Ensemble + TTA → Post-processing → Submission (RLE)


---

## ⚖️ License

This project follows the **ANRF Open License**.  
Please refer to the `LICENSE` file for full details.

---

## 📌 Notes

- Designed for high-performance flood detection tasks  
- Optimized for competitions like Kaggle / real-world disaster response  
- Easily extendable to other segmentation problems  
