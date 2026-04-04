# AI-Powered Flood Segmentation using Multi-Modal Satellite Data

This project implements a deep learning pipeline for flood segmentation using multi-modal satellite imagery combining SAR and optical data.

## 🔍 Overview
The system processes satellite images and predicts flood regions using semantic segmentation models. The approach integrates feature engineering and ensemble learning to improve performance on limited datasets.

## 🧠 Methodology

### Data Processing
- Multi-modal input: SAR (HH, HV) + Optical bands
- Feature engineering:
  - NDWI (Normalized Difference Water Index)
  - MNDWI
  - NDVI
  - SAR difference (HH - HV)
- Percentile-based normalization (2–98%) to handle noise and outliers

### Models Used
- **UNet (ResNet34 encoder)** — stable baseline model
- **UNet++ (EfficientNet-B5 encoder)** — captures complex spatial features

### Training Strategy
- Loss Function:
  - Weighted CrossEntropy Loss (for class imbalance)
  - Dice Loss (for segmentation accuracy)
- Optimizer: AdamW
- Batch size: 4
- Multi-epoch training

### Ensemble Technique
Predictions from both models are combined using weighted averaging:
Final Prediction = 0.8 × Model1 + 0.2 × Model2

This improves generalization and boosts IoU performance.

## 📊 Results
- Significant improvement over single-model baseline
- Ensemble strategy improves robustness and accuracy

## 📁 Structure
- Training pipelines for both models
- Inference and submission generation
- Preprocessing utilities

## 📜 License
This project follows the ANRF Open License.
